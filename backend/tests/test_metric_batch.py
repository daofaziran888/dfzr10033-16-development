"""分组指标批次：整组公式锁定同一立方批次、批量结算、缺项挂起与续算测试。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.metric_batch import MetricBatchService
from app.services.metric_factory import MetricFactoryService
from app.services.olap_cube import OlapCubeService
from app.store import store

client = TestClient(app)


def _make_cube(cube_no: str, measures: str, cached_rows: list[dict]) -> int:
    """构建一个已确认（可用）立方，并写入带确定度量明细的聚合缓存。"""
    cube_svc = OlapCubeService()
    entry, missing = cube_svc.create_entry({
        "立方编号": cube_no, "立方名称": f"测试立方-{cube_no}",
        "维度列表": "region", "度量列表": measures,
        "聚合类型": "SUM", "分区策略": "dt", "刷新周期": "3600", "数据行数": "0"})
    assert entry is not None, missing
    cube_id = entry["id"]
    build = cube_svc.build_cube(cube_id)
    assert build["ok"]
    # 用确定的度量明细覆盖随机预热数据，保证指标值可断言
    agg_rows = store.rows("aggregate_cache")
    for cache in agg_rows:
        if cache.get("目标立方") == f"olap_cube:{cube_id}" and cache.get("status") == "生效中":
            cache["cached_data"] = [dict(row) for row in cached_rows]
    return cube_id


def _make_metric(metric_no: str, name: str, formula: str, cube_id: int) -> int:
    svc = MetricFactoryService()
    entry, missing = svc.create_entry({
        "指标编号": metric_no, "指标名称": name, "数据来源": f"olap_cube:{cube_id}",
        "计算公式": formula, "聚合窗口": "daily", "精度模式": "exact",
        "依赖立方": str(cube_id), "最后产出": "-"})
    assert entry is not None, missing
    return int(entry["id"])


def _make_batch(cube_id: int, metric_ids: list[int]) -> int:
    svc = MetricBatchService()
    entry, missing = svc.create_entry({
        "批次编号": f"GMBC-T-{cube_id}-{len(metric_ids)}", "批次名称": "分组指标批次测试",
        "锁定立方": f"olap_cube:{cube_id}", "指标编号清单": ",".join(map(str, metric_ids)),
        "交付策略": "尽快交付可算部分"})
    assert entry is not None, missing
    return int(entry["id"])


@pytest.fixture(autouse=True)
def reset_inflight():
    MetricBatchService._inflight.clear()
    yield
    MetricBatchService._inflight.clear()


class TestCubeBatchLock:
    def test_lock_requires_confirmed_cube(self):
        cube_svc = OlapCubeService()
        entry, _ = cube_svc.create_entry({
            "立方编号": "TEST-LOCK-CUBE-1", "立方名称": "未确认立方",
            "维度列表": "region", "度量列表": "revenue",
            "聚合类型": "SUM", "分区策略": "dt", "刷新周期": "3600", "数据行数": "0"})
        batch_svc = MetricBatchService()
        batch, _ = batch_svc.create_entry({
            "批次编号": "GMBC-LOCK-1", "批次名称": "锁定校验",
            "锁定立方": f"olap_cube:{entry['id']}"})
        result = batch_svc.lock_cube_batch(batch["id"])
        assert not result["ok"]
        assert "已确认" in result["message"]

    def test_lock_binds_all_formulas_to_same_fingerprint(self):
        cube_id = _make_cube("TEST-LOCK-CUBE-2", "revenue,cost",
                             [{"region": "CN", "revenue": 100.0, "cost": 60.0}])
        m1 = _make_metric("METR-LOCK-1", "销售额", "SUM(revenue)", cube_id)
        m2 = _make_metric("METR-LOCK-2", "成本额", "SUM(cost)", cube_id)
        batch_id = _make_batch(cube_id, [m1, m2])
        result = MetricBatchService().lock_cube_batch(batch_id)
        assert result["ok"] and result["metric_count"] == 2
        batch = store.find("metric_batch", batch_id)
        assert batch["status"] == "已锁定"
        assert batch["立方批次指纹"].startswith("CUBEBATCH-")
        lines = batch["settlement_lines"]
        assert {line["metric_id"] for line in lines} == {m1, m2}
        # 一整组公式锁定同一个已确认立方批次：去重键共享同一指纹前缀
        assert all(line["去重键"].startswith(batch["立方批次指纹"]) for line in lines)


class TestBatchSettle:
    def test_sum_metric_settled_and_cached(self):
        cube_id = _make_cube("TEST-SETTLE-CUBE-1", "revenue",
                             [{"region": "CN", "revenue": 120.0},
                              {"region": "US", "revenue": 80.0}])
        m1 = _make_metric("METR-SETTLE-1", "销售额求和", "SUM(revenue)", cube_id)
        batch_id = _make_batch(cube_id, [m1])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)
        result = svc.settle_batch(batch_id)
        assert result["ok"] and result["settled"] == 1
        line = store.find("metric_batch", batch_id)["settlement_lines"][0]
        assert line["指标值"] == pytest.approx(200.0)
        assert line["分子"] == pytest.approx(200.0)
        assert line["缓存凭据"]
        # 结果缓存按批次接收到结算值
        cache = next(c for c in store.rows("result_cache") if c.get("结果哈希") == line["缓存凭据"])
        assert cache["status"] == "有效"
        assert cache["result_data"][0]["value"] == pytest.approx(200.0)
        assert cache["result_data"][0]["fingerprint"] == store.find("metric_batch", batch_id)["立方批次指纹"]

    def test_ratio_missing_denominator_suspended_not_zero(self):
        # 暂缺分母 total_revenue：只有 sales 字段有值
        cube_id = _make_cube("TEST-RATIO-CUBE-1", "sales,total_revenue",
                             [{"region": "CN", "sales": 30.0},
                              {"region": "US", "sales": 20.0}])
        sales_metric = _make_metric("METR-RATIO-1", "选中销售额求和", "SUM(sales)", cube_id)
        margin_metric = _make_metric(
            "METR-RATIO-2", "毛利率", "RATIO(SUM(sales),SUM(total_revenue))", cube_id)
        batch_id = _make_batch(cube_id, [sales_metric, margin_metric])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)
        result = svc.settle_batch(batch_id)

        lines = {line["metric_id"]: line for line in result["lines"]}
        # 可算部分尽快交付
        assert lines[sales_metric]["结算状态"] == "已结算"
        assert lines[sales_metric]["指标值"] == pytest.approx(50.0)
        # 缺分母的比率必须挂起，绝不能当作零计算
        assert lines[margin_metric]["结算状态"] == "缺项挂起"
        assert "total_revenue" in lines[margin_metric]["缺失依赖"]
        assert lines[margin_metric]["指标值"] is None
        assert lines[margin_metric]["缓存凭据"] is None
        batch = store.find("metric_batch", batch_id)
        assert batch["status"] == "部分交付"

        # 报表核对公式依赖是否齐全
        report = svc.dependency_report(batch_id)
        report_lines = {line["metric_id"]: line for line in report["lines"]}
        assert report_lines[sales_metric]["依赖齐全"] is True
        assert report_lines[margin_metric]["依赖齐全"] is False
        assert report_lines[margin_metric]["凭据可回查"] is False
        assert report_lines[sales_metric]["凭据可回查"] is True

    def test_ratio_uses_matched_numerator_and_denominator(self):
        cube_id = _make_cube("TEST-RATIO-CUBE-2", "sales,total_revenue",
                             [{"region": "CN", "sales": 30.0, "total_revenue": 200.0},
                              {"region": "US", "sales": 20.0, "total_revenue": 50.0}])
        margin = _make_metric(
            "METR-RATIO-3", "毛利率", "RATIO(SUM(sales),SUM(total_revenue))", cube_id)
        batch_id = _make_batch(cube_id, [margin])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)
        result = svc.settle_batch(batch_id)
        line = result["lines"][0]
        assert line["结算状态"] == "已结算"
        assert line["分子"] == pytest.approx(50.0)
        assert line["分母"] == pytest.approx(250.0)
        assert line["指标值"] == pytest.approx(0.2)

    def test_ratio_can_reference_sibling_metric_value(self):
        cube_id = _make_cube("TEST-RATIO-CUBE-3", "sales,total_revenue",
                             [{"region": "CN", "sales": 30.0, "total_revenue": 200.0}])
        sales_metric = _make_metric("METR-RATIO-4", "选中销售额求和", "SUM(sales)", cube_id)
        margin_metric = _make_metric(
            "METR-RATIO-5", "引用毛利率", f"RATIO(METRIC({sales_metric}),SUM(total_revenue))", cube_id)
        batch_id = _make_batch(cube_id, [sales_metric, margin_metric])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)
        result = svc.settle_batch(batch_id)
        lines = {line["metric_id"]: line for line in result["lines"]}
        assert lines[sales_metric]["结算状态"] == "已结算"
        assert lines[margin_metric]["结算状态"] == "已结算"
        assert lines[margin_metric]["指标值"] == pytest.approx(30.0 / 200.0)
        assert lines[margin_metric]["分子"] == pytest.approx(30.0)
        assert lines[margin_metric]["分母"] == pytest.approx(200.0)

    def test_zero_denominator_is_failure_not_zero_ratio(self):
        cube_id = _make_cube("TEST-RATIO-CUBE-4", "sales,total_revenue",
                             [{"region": "CN", "sales": 30.0, "total_revenue": 0.0}])
        margin = _make_metric(
            "METR-RATIO-6", "零分母毛利率", "RATIO(SUM(sales),SUM(total_revenue))", cube_id)
        batch_id = _make_batch(cube_id, [margin])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)
        result = svc.settle_batch(batch_id)
        assert result["failed"] == 1
        assert result["lines"][0]["结算状态"] == "结算失败"
        assert "分母" in result["lines"][0]["失败原因"]


class TestResumeAndDedup:
    def test_resume_reuses_credentials_and_settles_suspended(self):
        cube_id = _make_cube("TEST-RESUME-CUBE-1", "sales,total_revenue",
                             [{"region": "CN", "sales": 30.0},
                              {"region": "US", "sales": 20.0}])
        sales_metric = _make_metric("METR-RESUME-1", "选中销售额求和", "SUM(sales)", cube_id)
        margin_metric = _make_metric(
            "METR-RESUME-2", "毛利率", "RATIO(SUM(sales),SUM(total_revenue))", cube_id)
        batch_id = _make_batch(cube_id, [sales_metric, margin_metric])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)
        first = svc.settle_batch(batch_id)
        assert first["settled"] == 1 and first["pending"] == 1

        # 暂缺分母时先核对各项缓存凭据：销售额凭据可回查，毛利率无凭据
        sales_line = next(l for l in first["lines"] if l["metric_id"] == sales_metric)
        margin_line = next(l for l in first["lines"] if l["metric_id"] == margin_metric)
        assert sales_line["缓存凭据"] and margin_line["缓存凭据"] is None
        cache_before = next(
            c for c in store.rows("result_cache") if c.get("结果哈希") == sales_line["缓存凭据"])
        hits_before = int(cache_before["缓存命中"])

        # 补齐输入：同一立方批次补入 total_revenue 度量明细（指纹不变）
        for cache in store.rows("aggregate_cache"):
            if cache.get("目标立方") == f"olap_cube:{cube_id}" and cache.get("status") == "生效中":
                for row in cache["cached_data"]:
                    row["total_revenue"] = 100.0

        # 续跑原批
        resumed = svc.resume_batch(batch_id)
        assert resumed["ok"]
        assert resumed["reused_credentials"] == 1
        assert sales_metric in resumed["reused_metric_ids"]
        batch = store.find("metric_batch", batch_id)
        assert batch["status"] == "已齐套"
        lines = {line["metric_id"]: line for line in batch["settlement_lines"]}
        assert lines[margin_metric]["结算状态"] == "已结算"
        assert lines[margin_metric]["分母"] == pytest.approx(200.0)
        assert lines[margin_metric]["指标值"] == pytest.approx(50.0 / 200.0)
        # 已结算指标值保持不变，凭据被复用并增加一次命中
        assert lines[sales_metric]["指标值"] == pytest.approx(50.0)
        cache_after = next(
            c for c in store.rows("result_cache") if c.get("结果哈希") == sales_line["缓存凭据"])
        assert int(cache_after["缓存命中"]) == hits_before + 1

        # 逐条核算入库的指标值、分子分母和公式覆盖
        report = svc.dependency_report(batch_id)
        assert report["formula_coverage"] == 1.0
        assert report["all_credentials_verifiable"] is True
        for line in report["lines"]:
            assert line["依赖齐全"] and line["凭据可回查"]

    def test_resume_rejects_changed_cube_fingerprint(self):
        cube_id = _make_cube("TEST-RESUME-CUBE-2", "sales",
                             [{"region": "CN", "sales": 30.0}])
        m1 = _make_metric("METR-RESUME-3", "销售额", "SUM(sales)", cube_id)
        batch_id = _make_batch(cube_id, [m1])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)
        svc.settle_batch(batch_id)
        # 立方重建导致指纹变化
        cube = store.find("olap_cube", cube_id)
        cube["数据行数"] = str(int(cube["数据行数"]) + 999)
        result = svc.resume_batch(batch_id)
        assert not result["ok"] and "指纹" in result["message"]

    def test_concurrent_settle_dedupes_by_batch_and_metric(self):
        cube_id = _make_cube("TEST-CONC-CUBE-1", "revenue",
                             [{"region": "CN", "revenue": 100.0},
                              {"region": "US", "revenue": 50.0}])
        m1 = _make_metric("METR-CONC-1", "销售额", "SUM(revenue)", cube_id)
        batch_id = _make_batch(cube_id, [m1])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: svc.settle_batch(batch_id), range(4)))
        assert all(r["ok"] for r in results)
        # 去重 + 凭据复用：同一批次同一指标只产生一条结果缓存
        entries = [
            c for c in store.rows("result_cache")
            if c.get("查询摘要") == f"metric_batch_{batch_id}:metric_{m1}"]
        assert len(entries) == 1
        line = store.find("metric_batch", batch_id)["settlement_lines"][0]
        assert line["指标值"] == pytest.approx(150.0)

    def test_repeated_settle_reuses_written_credential(self):
        cube_id = _make_cube("TEST-CONC-CUBE-2", "revenue",
                             [{"region": "CN", "revenue": 77.0}])
        m1 = _make_metric("METR-CONC-2", "销售额", "SUM(revenue)", cube_id)
        batch_id = _make_batch(cube_id, [m1])
        svc = MetricBatchService()
        svc.lock_cube_batch(batch_id)
        first = svc.settle_batch(batch_id)
        second = svc.settle_batch(batch_id)
        assert first["settled"] == 1 and second["settled"] == 1
        assert second["reused_credentials"] == 1
        entries = [
            c for c in store.rows("result_cache")
            if c.get("查询摘要") == f"metric_batch_{batch_id}:metric_{m1}"]
        assert len(entries) == 1


class TestMetricBatchApi:
    def test_full_pipeline_api_lock_settle_report_resume(self):
        cube_id = _make_cube("TEST-API-CUBE-1", "sales,total_revenue",
                             [{"region": "CN", "sales": 40.0}])
        sales = _make_metric("METR-API-1", "选中销售额求和", "SUM(sales)", cube_id)
        margin = _make_metric(
            "METR-API-2", "毛利率", "RATIO(SUM(sales),SUM(total_revenue))", cube_id)

        lock = client.post("/api/pipelines/metric-batch/lock", params={
            "batch_no": "GMBC-API-1", "batch_name": "接口批次",
            "cube_id": cube_id, "metric_ids": f"{sales},{margin}"})
        assert lock.status_code == 200
        batch_id = lock.json()["batch_id"]

        settle = client.post(f"/api/pipelines/metric-batch/{batch_id}/settle")
        assert settle.status_code == 200
        body = settle.json()
        assert body["settled"] == 1 and body["pending"] == 1
        assert body["entry"]["status"] == "部分交付"

        report = client.get(f"/api/pipelines/metric-batch/{batch_id}/report")
        assert report.status_code == 200
        missing_line = next(line for line in report.json()["lines"] if line["metric_id"] == margin)
        assert missing_line["依赖齐全"] is False

        # 补齐分母后续跑
        for cache in store.rows("aggregate_cache"):
            if cache.get("目标立方") == f"olap_cube:{cube_id}" and cache.get("status") == "生效中":
                for row in cache["cached_data"]:
                    row["total_revenue"] = 160.0
        resume = client.post(f"/api/pipelines/metric-batch/{batch_id}/resume")
        assert resume.status_code == 200
        assert resume.json()["entry"]["status"] == "已齐套"
        final_report = client.get(f"/api/pipelines/metric-batch/{batch_id}/report").json()
        assert final_report["formula_coverage"] == 1.0

    def test_generic_resource_list_and_action_flow(self):
        cube_id = _make_cube("TEST-API-CUBE-2", "revenue",
                             [{"region": "CN", "revenue": 10.0}])
        m1 = _make_metric("METR-API-3", "销售额", "SUM(revenue)", cube_id)
        batch_id = _make_batch(cube_id, [m1])

        listed = client.get("/api/metric_batch")
        assert listed.status_code == 200
        assert listed.json()["total"] >= 1

        locked = client.post(f"/api/metric_batch/{batch_id}/actions", json={"action": "锁定批次"})
        assert locked.status_code == 200 and locked.json()["ok"]
        settled = client.post(f"/api/metric_batch/{batch_id}/actions", json={"action": "并发结算"})
        assert settled.status_code == 200 and settled.json()["ok"]
        assert settled.json()["entry"]["status"] == "已齐套"

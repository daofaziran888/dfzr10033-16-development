"""分组指标批次领域服务。

同一立方的多个指标原本只能逐条计算，本模块把一整组指标公式锁定到同一个
已确认的立方批次上，基于同一份立方度量明细一次批量结算：

- ``lock_cube_batch``   把整组公式绑定到同一个已确认（可用）立方批次，
  生成只读的立方批次指纹与初始指标结算清单。
- ``settle_batch``      并发结算，用「批次 + 指标编号」去重；已经写入结果
  缓存的凭据直接复用，不重复计算。
- ``resume_batch``      补齐输入后续跑原批：已结算指标复用缓存凭据，仅重算
  挂起/失败指标，逐条回写指标值、分子分母与公式覆盖。
- ``dependency_report`` 报表核对每条公式的依赖字段是否齐全、缓存凭据是否
  仍然可回查。

硬性口径：任意比率（RATIO）必须使用匹配的分子分母，缺项一律挂起，
绝不把缺失项当作零参与计算。
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.services.base import BaseModuleService
from app.store import store


class MetricBatchService(BaseModuleService):
    MODULE = "metric_batch"
    ENTITY = "分组指标批次"
    OPERATION_TERM = "结算"
    LIST_FIELDS = [
        "批次编号", "批次名称", "锁定立方", "立方批次指纹", "指标项数",
        "已结算项数", "交付策略", "最后结算时间",
    ]
    REQUIRED_FIELDS = ["批次编号", "批次名称", "锁定立方"]
    STATUS_ORDER = ["草稿", "已锁定", "部分交付", "已齐套", "结算异常"]
    ACTION_RULES = {
        "锁定批次": "已锁定",
        "并发结算": "部分交付",
        "续算原批": "已齐套",
        "标记异常": "结算异常",
    }
    NEGATIVE_ACTIONS = ["标记异常"]

    # 同一进程内的并发结算去重锁：key = (批次id, 指标id)
    _settle_lock = threading.RLock()
    _inflight: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------ #
    # 登记
    # ------------------------------------------------------------------ #
    def create_entry(self, values: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        entry, missing = super().create_entry(values)
        if entry is None:
            return entry, missing
        cube_id = _parse_cube_id(values.get("锁定立方"))
        metric_ids = _parse_id_list(values.get("指标编号清单", ""))
        entry["指标编号清单"] = ",".join(str(mid) for mid in metric_ids)
        entry["交付策略"] = values.get("交付策略", "尽快交付可算部分")
        entry["锁定立方"] = f"olap_cube:{cube_id}" if cube_id is not None else values.get("锁定立方")
        entry["指标项数"] = "0"
        entry["已结算项数"] = "0"
        entry["最后结算时间"] = "-"
        entry["立方批次指纹"] = "-"
        entry["settlement_lines"] = []
        return entry, []

    def run_action(self, entry_id: int, action: str) -> tuple[dict[str, Any] | None, str]:
        if action == "锁定批次":
            result = self.lock_cube_batch(entry_id)
            if not result["ok"]:
                return None, result["message"]
            return store.find(self.MODULE, entry_id), result["message"]
        if action == "并发结算":
            result = self.settle_batch(entry_id)
            if not result["ok"]:
                return None, result["message"]
            return store.find(self.MODULE, entry_id), result["message"]
        if action == "续算原批":
            result = self.resume_batch(entry_id)
            if not result["ok"]:
                return None, result["message"]
            return store.find(self.MODULE, entry_id), result["message"]
        return super().run_action(entry_id, action)

    # ------------------------------------------------------------------ #
    # 锁定：整组公式绑定同一个已确认立方批次
    # ------------------------------------------------------------------ #
    def lock_cube_batch(self, batch_id: int) -> dict[str, Any]:
        batch = store.find(self.MODULE, batch_id)
        if batch is None:
            return {"ok": False, "message": f"分组指标批次 {batch_id} 不存在或已归档"}

        cube_id = _parse_cube_id(batch.get("锁定立方", ""))
        cube = store.find("olap_cube", cube_id) if cube_id is not None else None
        if cube is None:
            return {"ok": False, "message": "锁定失败：目标立方不存在或引用格式无法识别"}
        # 只允许锁定已确认（可用）的立方批次
        if cube.get("status") != "可用":
            return {
                "ok": False,
                "message": f"锁定失败：立方 {cube.get('立方编号')} 状态为「{cube.get('status')}」，"
                           "仅已确认（可用）的立方批次可被分组指标锁定",
            }

        metric_ids = _parse_id_list(batch.get("指标编号清单", ""))
        if not metric_ids:
            # 未显式指定时，默认收拢所有依赖该立方的指标公式
            metric_ids = [
                int(m["id"]) for m in store.rows("metric_factory")
                if str(cube_id) in str(m.get("依赖立方", ""))
            ]
        metrics = [store.find("metric_factory", mid) for mid in metric_ids]
        missing_metrics = [mid for mid, m in zip(metric_ids, metrics) if m is None]
        if missing_metrics:
            return {"ok": False, "message": f"锁定失败：指标 {missing_metrics} 不存在"}

        measures = _split_csv(cube.get("度量列表", ""))
        fingerprint = _cube_fingerprint(cube)
        batch["锁定立方"] = f"olap_cube:{cube_id}"
        batch["立方批次指纹"] = fingerprint
        batch["指标编号清单"] = ",".join(str(mid) for mid in metric_ids)
        batch["指标项数"] = str(len(metric_ids))
        batch["已结算项数"] = "0"
        batch["locked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        batch["cube_snapshot"] = {
            "cube_id": cube_id,
            "立方编号": cube.get("立方编号"),
            "status": cube.get("status"),
            "数据行数": cube.get("数据行数"),
            "度量列表": measures,
            "fingerprint": fingerprint,
        }
        # 读取多个指标公式，生成指标结算清单
        batch["settlement_lines"] = [
            {
                "metric_id": int(metric["id"]),
                "指标编号": metric.get("指标编号"),
                "指标名称": metric.get("指标名称"),
                "计算公式": metric.get("计算公式", ""),
                "结算状态": "待结算",
                "指标值": None,
                "分子": None,
                "分母": None,
                "缺失依赖": _formula_dependencies(metric.get("计算公式", "")),
                "缓存凭据": None,
                "缓存条目id": None,
                "去重键": f"{fingerprint}:{metric['id']}",
                "结算时间": "-",
                "失败原因": "",
            }
            for metric in metrics
        ]
        batch["status"] = "已锁定"
        batch["pending"] = True
        batch["abnormal"] = False
        _write_audit("LOCK", batch_id, len(metric_ids), fingerprint)
        return {
            "ok": True,
            "batch_id": batch_id,
            "cube_id": cube_id,
            "fingerprint": fingerprint,
            "metric_count": len(metric_ids),
            "message": f"批次已锁定到立方批次 {fingerprint}，生成 {len(metric_ids)} 条指标结算清单",
        }

    # ------------------------------------------------------------------ #
    # 并发结算：批次 + 指标编号去重，已写入的缓存凭据直接复用
    # ------------------------------------------------------------------ #
    def settle_batch(self, batch_id: int) -> dict[str, Any]:
        batch = store.find(self.MODULE, batch_id)
        if batch is None:
            return {"ok": False, "message": f"分组指标批次 {batch_id} 不存在或已归档"}
        if batch.get("status") not in ("已锁定", "部分交付", "结算异常", "已齐套", "草稿"):
            return {"ok": False, "message": f"当前状态「{batch.get('status')}」不允许并发结算"}
        if not batch.get("settlement_lines"):
            locked = self.lock_cube_batch(batch_id)
            if not locked["ok"]:
                return locked
            batch = store.find(self.MODULE, batch_id)

        cube_id = _parse_cube_id(batch.get("锁定立方", ""))
        measure_rows = _load_cube_measure_rows(cube_id)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(batch["settlement_lines"])))) as pool:
            futures = [
                pool.submit(self._settle_line, batch_id, line, measure_rows)
                for line in batch["settlement_lines"]
            ]
            line_results = [f.result() for f in futures]
        elapsed_ms = (time.time() - t0) * 1000

        self._refresh_batch_state(batch, elapsed_ms)
        settled = [r for r in line_results if r["结算状态"] == "已结算"]
        reused = [r for r in line_results if r.get("reused")]
        return {
            "ok": True,
            "batch_id": batch_id,
            "fingerprint": batch.get("立方批次指纹"),
            "total": len(line_results),
            "settled": len(settled),
            "pending": sum(1 for r in line_results if r["结算状态"] == "缺项挂起"),
            "failed": sum(1 for r in line_results if r["结算状态"] == "结算失败"),
            "reused_credentials": len(reused),
            "elapsed_ms": elapsed_ms,
            "lines": line_results,
            "message": _settle_message(batch, len(settled), len(line_results)),
        }

    def _settle_line(
        self, batch_id: int, line: dict[str, Any], measure_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        metric_id = int(line["metric_id"])
        dedupe_key = (batch_id, metric_id)

        # 续算/重复结算：已经写入结果缓存的凭据直接复用，不重算
        existing = _reuse_credential(line)
        if existing is not None:
            return {**_line_public(line), "reused": True}

        with self._settle_lock:
            if dedupe_key in self._inflight:
                # 并发去重：同一批次同一指标已有结算在途，等待其落库后复用凭据
                waited = 0
                while dedupe_key in self._inflight and waited < 50:
                    time.sleep(0.01)
                    waited += 1
                reused = _reuse_credential(line)
                if reused is not None:
                    return {**_line_public(line), "reused": True, "deduped": True}
            self._inflight.add(dedupe_key)

        try:
            metric = store.find("metric_factory", metric_id)
            if metric is None:
                line["结算状态"] = "结算失败"
                line["失败原因"] = "指标定义不存在"
                return _line_public(line)

            formula = str(metric.get("计算公式", ""))
            try:
                outcome = _evaluate_formula(formula, measure_rows, batch_lines=_batch_lines(batch_id))
            except _MissingDependency as exc:
                # 缺项不能当作零：比率/聚合缺字段一律挂起，等待补齐后续算
                line["结算状态"] = "缺项挂起"
                line["缺失依赖"] = exc.missing
                line["失败原因"] = str(exc)
                line["结算时间"] = time.strftime("%Y-%m-%d %H:%M:%S")
                return _line_public(line)
            except ZeroDenominatorError as exc:
                line["结算状态"] = "结算失败"
                line["失败原因"] = str(exc)
                line["结算时间"] = time.strftime("%Y-%m-%d %H:%M:%S")
                return _line_public(line)

            value = round(float(outcome["value"]), 6)
            # 结果缓存按分组指标批次接收已结算的指标值
            cache_entry = _write_result_cache(batch_id, line, value, outcome)
            # 缓存写入结果回写指标结算清单（可回查凭据）
            line["指标值"] = value
            line["分子"] = outcome.get("numerator")
            line["分母"] = outcome.get("denominator")
            line["缺失依赖"] = []
            line["失败原因"] = ""
            line["缓存凭据"] = cache_entry["凭据"]
            line["缓存条目id"] = cache_entry["cache_id"]
            line["结算状态"] = "已结算"
            line["结算时间"] = time.strftime("%Y-%m-%d %H:%M:%S")

            metric["最后产出"] = line["结算时间"]
            metric["status"] = "已产出"
            metric["pending"] = False
            return _line_public(line)
        finally:
            with self._settle_lock:
                self._inflight.discard(dedupe_key)

    # ------------------------------------------------------------------ #
    # 续算：补齐输入后跑原批，复用已写入缓存凭据，仅重算挂起/失败项
    # ------------------------------------------------------------------ #
    def resume_batch(self, batch_id: int) -> dict[str, Any]:
        batch = store.find(self.MODULE, batch_id)
        if batch is None:
            return {"ok": False, "message": f"分组指标批次 {batch_id} 不存在或已归档"}
        fingerprint = batch.get("立方批次指纹")
        cube_id = _parse_cube_id(batch.get("锁定立方", ""))
        cube = store.find("olap_cube", cube_id) if cube_id is not None else None
        if cube is None or _cube_fingerprint(cube) != fingerprint:
            return {"ok": False, "message": "续算拒绝：立方批次指纹已变化，必须重新锁定，不能混用不同立方批次"}
        if batch.get("status") not in ("部分交付", "结算异常", "已锁定", "已齐套"):
            return {"ok": False, "message": f"当前状态「{batch.get('status')}」不允许续算"}

        measure_rows = _load_cube_measure_rows(cube_id)
        t0 = time.time()
        reused, recomputed = [], []
        for line in batch["settlement_lines"]:
            if line.get("结算状态") == "已结算" and _reuse_credential(line) is not None:
                reused.append(line["metric_id"])
                continue
            recomputed.append(self._settle_line(batch_id, line, measure_rows))
        elapsed_ms = (time.time() - t0) * 1000

        self._refresh_batch_state(batch, elapsed_ms)
        return {
            "ok": True,
            "batch_id": batch_id,
            "fingerprint": fingerprint,
            "reused_credentials": len(reused),
            "reused_metric_ids": reused,
            "recomputed": recomputed,
            "elapsed_ms": elapsed_ms,
            "lines": [_line_public(line) for line in batch["settlement_lines"]],
            "message": _settle_message(
                batch,
                sum(1 for line in batch["settlement_lines"] if line["结算状态"] == "已结算"),
                len(batch["settlement_lines"]),
            ),
        }

    # ------------------------------------------------------------------ #
    # 报表：核对公式依赖是否齐全、凭据是否可回查
    # ------------------------------------------------------------------ #
    def dependency_report(self, batch_id: int) -> dict[str, Any]:
        batch = store.find(self.MODULE, batch_id)
        if batch is None:
            return {"ok": False, "message": f"分组指标批次 {batch_id} 不存在或已归档"}
        cube_id = _parse_cube_id(batch.get("锁定立方", ""))
        available = _available_measure_fields(cube_id)
        lines_report = []
        for line in batch.get("settlement_lines", []):
            deps = _formula_dependencies(line.get("计算公式", ""))
            missing = [d for d in deps if d not in available]
            credential_ok = _credential_verifiable(line)
            lines_report.append({
                "metric_id": line.get("metric_id"),
                "指标编号": line.get("指标编号"),
                "指标名称": line.get("指标名称"),
                "计算公式": line.get("计算公式"),
                "依赖字段": deps,
                "立方可用字段": available,
                "缺失依赖": missing,
                "依赖齐全": not missing,
                "结算状态": line.get("结算状态"),
                "指标值": line.get("指标值"),
                "分子": line.get("分子"),
                "分母": line.get("分母"),
                "缓存凭据": line.get("缓存凭据"),
                "凭据可回查": credential_ok,
            })
        total = len(lines_report)
        complete = sum(1 for r in lines_report if r["依赖齐全"])
        settled = sum(1 for r in lines_report if r["结算状态"] == "已结算")
        return {
            "ok": True,
            "batch_id": batch_id,
            "批次编号": batch.get("批次编号"),
            "fingerprint": batch.get("立方批次指纹"),
            "cube_snapshot": batch.get("cube_snapshot"),
            "total": total,
            "dependency_complete": complete,
            "settled": settled,
            "formula_coverage": round(settled / total, 4) if total else 0.0,
            "all_credentials_verifiable": all(r["凭据可回查"] for r in lines_report if r["结算状态"] == "已结算"),
            "lines": lines_report,
        }

    # ------------------------------------------------------------------ #
    def _refresh_batch_state(self, batch: dict[str, Any], elapsed_ms: float) -> None:
        lines = batch["settlement_lines"]
        settled = sum(1 for line in lines if line["结算状态"] == "已结算")
        failed = sum(1 for line in lines if line["结算状态"] == "结算失败")
        pending = sum(1 for line in lines if line["结算状态"] == "缺项挂起")
        batch["已结算项数"] = str(settled)
        batch["最后结算时间"] = time.strftime("%Y-%m-%d %H:%M:%S")
        batch["last_elapsed_ms"] = round(elapsed_ms, 3)
        if failed:
            batch["status"] = "结算异常"
            batch["abnormal"] = True
            batch["pending"] = True
        elif pending or settled < len(lines):
            # 尽快交付可算部分：能算的先落缓存，其余挂起
            batch["status"] = "部分交付"
            batch["abnormal"] = False
            batch["pending"] = True
        else:
            batch["status"] = "已齐套"
            batch["abnormal"] = False
            batch["pending"] = False
        _write_audit("SETTLE", int(batch["id"]), settled, batch.get("立方批次指纹", "-"))


# ---------------------------------------------------------------------- #
# 公式解析与计算
# ---------------------------------------------------------------------- #
class _MissingDependency(Exception):
    def __init__(self, missing: list[str], context: str = "") -> None:
        self.missing = missing
        super().__init__(f"缺少依赖字段：{'、'.join(missing)}" + (f"（{context}）" if context else ""))


class ZeroDenominatorError(Exception):
    pass


_AGG_FUNCS = ("SUM", "AVG", "COUNT", "MAX", "MIN")


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    """返回 (函数名, 参数列表)。支持 RATIO(a,b)、METRIC(id)、SUM(field) 等。"""
    text = (formula or "").strip()
    if "(" not in text or ")" not in text:
        return "RAW", [text] if text else []
    func = text[: text.index("(")].strip().upper()
    args_text = text[text.index("(") + 1: text.rindex(")")]
    args = [arg.strip() for arg in args_text.split(",") if arg.strip()]
    return func, args


def _formula_dependencies(formula: str) -> list[str]:
    """静态提取公式直接依赖的度量字段（METRIC 引用是同批指标，不属于立方字段）。"""
    func, args = _parse_formula(formula)
    if not args:
        return []
    deps: list[str] = []
    for arg in args:
        sub_func, sub_args = _parse_formula(arg)
        if sub_func in _AGG_FUNCS:
            deps.extend(sub_args)
        elif sub_func == "METRIC":
            continue
        elif sub_func == "RAW" and arg:
            deps.append(arg)
    return deps


def _column_values(field: str, rows: list[dict[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        if field not in row or row[field] is None:
            continue
        value = row[field]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _aggregate(func: str, field: str, rows: list[dict[str, Any]]) -> float:
    values = _column_values(field, rows)
    present = any(field in row and row[field] is not None for row in rows)
    if not present:
        raise _MissingDependency([field], func)
    if not values:
        raise _MissingDependency([field], func)
    if func == "SUM":
        return float(sum(values))
    if func == "AVG":
        return sum(values) / len(values)
    if func == "COUNT":
        return float(len(values))
    if func == "MAX":
        return float(max(values))
    if func == "MIN":
        return float(min(values))
    raise _MissingDependency([field], func)


def _resolve_operand(token: str, rows: list[dict[str, Any]], batch_lines: list[dict[str, Any]]) -> float:
    func, args = _parse_formula(token)
    if func in _AGG_FUNCS:
        return _aggregate(func, args[0], rows)
    if func == "METRIC":
        ref_id = int(args[0])
        for line in batch_lines:
            if int(line["metric_id"]) == ref_id:
                if line.get("结算状态") != "已结算" or line.get("指标值") is None:
                    raise _MissingDependency([], f"同批指标 {ref_id} 尚未结算")
                return float(line["指标值"])
        raise _MissingDependency([], f"同批指标 {ref_id} 不在本批次内")
    if func == "RAW":
        # 裸字段按 SUM 口径取值
        return _aggregate("SUM", token, rows)
    raise _MissingDependency([], f"无法解析的公式片段：{token}")


def _evaluate_formula(
    formula: str, rows: list[dict[str, Any]], batch_lines: list[dict[str, Any]]
) -> dict[str, Any]:
    func, args = _parse_formula(formula)
    if func == "RATIO":
        if len(args) != 2:
            raise ZeroDenominatorError("RATIO 公式必须同时提供分子与分母两个参数")
        num_token, den_token = args
        # 先静态检查分子分母依赖字段是否齐全，缺项绝不当零
        missing = [d for d in _formula_dependencies(formula) if not _field_available(d, rows)]
        if missing:
            raise _MissingDependency(missing, "RATIO")
        numerator = _resolve_operand(num_token, rows, batch_lines)
        denominator = _resolve_operand(den_token, rows, batch_lines)
        if denominator == 0:
            raise ZeroDenominatorError("分母为零，比率不可计算（缺项不得当作零）")
        return {"value": numerator / denominator, "numerator": numerator, "denominator": denominator}
    if func in _AGG_FUNCS:
        value = _aggregate(func, args[0], rows)
        return {"value": value, "numerator": value, "denominator": None}
    if func == "METRIC":
        value = _resolve_operand(formula, rows, batch_lines)
        return {"value": value, "numerator": value, "denominator": None}
    if func == "RAW" and args:
        value = _aggregate("SUM", args[0], rows)
        return {"value": value, "numerator": value, "denominator": None}
    raise _MissingDependency([], f"无法解析公式：{formula}")


def _field_available(field: str, rows: list[dict[str, Any]]) -> bool:
    return any(field in row and row[field] is not None for row in rows)


# ---------------------------------------------------------------------- #
# 立方批次与度量明细
# ---------------------------------------------------------------------- #
def _parse_cube_id(ref: Any) -> int | None:
    text = str(ref or "").strip()
    if not text:
        return None
    if ":" in text:
        text = text.split(":", 1)[1]
    try:
        return int(text)
    except ValueError:
        return None


def _split_csv(text: Any) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _parse_id_list(text: Any) -> list[int]:
    ids = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def _cube_fingerprint(cube: dict[str, Any]) -> str:
    """同一已确认立方批次的指纹：立方身份 + 状态 + 行数 + 度量清单的哈希。"""
    raw = "|".join([
        str(cube.get("id")),
        str(cube.get("立方编号")),
        str(cube.get("status")),
        str(cube.get("数据行数")),
        str(cube.get("度量列表")),
    ])
    return "CUBEBATCH-" + hashlib.md5(raw.encode()).hexdigest()[:12]


def _load_cube_measure_rows(cube_id: int | None) -> list[dict[str, Any]]:
    """读取锁定立方的度量明细。

    优先读取生效中的聚合缓存；当立方一致性检查把缓存标记为「淘汰中」但尚未
    物理清除时，回退读取同一锁定立方批次的明细，避免结算被缓存生命周期误杀。
    结果缓存（立方构建产物）作为第二回退来源。
    """
    rows: list[dict[str, Any]] = []
    stale_rows: list[dict[str, Any]] = []
    for cache in store.rows("aggregate_cache"):
        if f"olap_cube:{cube_id}" != str(cache.get("目标立方", "")).strip():
            continue
        data = cache.get("cached_data") or []
        if cache.get("status") == "生效中" and data:
            rows.extend(dict(item) for item in data)
        elif cache.get("status") == "淘汰中" and data:
            stale_rows.extend(dict(item) for item in data)
    if rows:
        return rows
    if stale_rows:
        return stale_rows
    for cache in store.rows("result_cache"):
        if cache.get("status") not in ("有效", "已过期"):
            continue
        if f"cube_build_{cube_id}" != str(cache.get("查询摘要", "")).strip():
            continue
        data = cache.get("result_data") or cache.get("cached_data") or []
        rows.extend(dict(item) for item in data)
    return rows


def _available_measure_fields(cube_id: int | None) -> list[str]:
    rows = _load_cube_measure_rows(cube_id)
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key.startswith("_") or key in ("group_count", "aggregated_value"):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in fields:
                fields.append(key)
    return fields


# ---------------------------------------------------------------------- #
# 结果缓存凭据
# ---------------------------------------------------------------------- #
def _write_result_cache(
    batch_id: int, line: dict[str, Any], value: float, outcome: dict[str, Any]
) -> dict[str, Any]:
    rows = store.rows("result_cache")
    cache_id = max((int(r.get("id", 0)) for r in rows), default=0) + 1
    fingerprint = _batch(batch_id).get("立方批次指纹", "-")
    metric_id = int(line["metric_id"])
    payload = {
        "batch_id": batch_id,
        "fingerprint": fingerprint,
        "metric_id": metric_id,
        "value": value,
        "numerator": outcome.get("numerator"),
        "denominator": outcome.get("denominator"),
        "formula": line.get("计算公式"),
    }
    credential = "GMBC-" + hashlib.md5(str(payload).encode()).hexdigest()[:16]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "id": cache_id,
        "条目编号": f"RESU-B{batch_id:04d}-M{metric_id:04d}",
        "查询摘要": f"metric_batch_{batch_id}:metric_{metric_id}",
        "结果哈希": credential,
        "缓存命中": "1",
        "过期时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + 3600)),
        "存储大小": str(len(str(payload))),
        "创建时间": now,
        "最后命中": now,
        "status": "有效",
        "pending": True,
        "abnormal": False,
        "result_data": [payload],
        "batch_id": batch_id,
        "metric_id": metric_id,
        "fingerprint": fingerprint,
        "凭据": credential,
    }
    rows.append(entry)
    return {"cache_id": cache_id, "凭据": credential}


def _reuse_credential(line: dict[str, Any]) -> dict[str, Any] | None:
    """续算/并发去重：结算清单里已写入的凭据若仍能在结果缓存回查，则复用。"""
    credential = line.get("缓存凭据")
    if not credential or line.get("结算状态") != "已结算":
        return None
    for entry in store.rows("result_cache"):
        if entry.get("结果哈希") == credential and entry.get("status") == "有效":
            entry["缓存命中"] = str(int(entry.get("缓存命中", "0")) + 1)
            entry["最后命中"] = time.strftime("%Y-%m-%d %H:%M:%S")
            payload = (entry.get("result_data") or [{}])[0]
            return {
                "cache_id": entry["id"],
                "凭据": credential,
                "value": payload.get("value"),
            }
    return None


def _credential_verifiable(line: dict[str, Any]) -> bool:
    if line.get("结算状态") != "已结算":
        return False
    credential = line.get("缓存凭据")
    if not credential:
        return False
    return any(
        entry.get("结果哈希") == credential and entry.get("status") == "有效"
        for entry in store.rows("result_cache")
    )


# ---------------------------------------------------------------------- #
# 辅助
# ---------------------------------------------------------------------- #
def _batch(batch_id: int) -> dict[str, Any]:
    return store.find("metric_batch", batch_id) or {}


def _batch_lines(batch_id: int) -> list[dict[str, Any]]:
    return _batch(batch_id).get("settlement_lines", [])


def _line_public(line: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric_id": line.get("metric_id"),
        "指标编号": line.get("指标编号"),
        "指标名称": line.get("指标名称"),
        "计算公式": line.get("计算公式"),
        "结算状态": line.get("结算状态"),
        "指标值": line.get("指标值"),
        "分子": line.get("分子"),
        "分母": line.get("分母"),
        "缺失依赖": line.get("缺失依赖"),
        "缓存凭据": line.get("缓存凭据"),
        "缓存条目id": line.get("缓存条目id"),
        "失败原因": line.get("失败原因"),
        "结算时间": line.get("结算时间"),
    }


def _settle_message(batch: dict[str, Any], settled: int, total: int) -> str:
    status = batch.get("status")
    if status == "已齐套":
        return f"整组 {total} 条指标已一次性出齐并写入结果缓存"
    if status == "结算异常":
        return f"已交付 {settled}/{total}，存在结算失败项，详情见指标结算清单"
    return f"已尽快交付可算部分 {settled}/{total}，缺项指标挂起等待补齐输入续算"


def _write_audit(action: str, batch_id: int, count: int, fingerprint: str) -> None:
    rows = store.rows("query_audit")
    audit = {
        "id": max((int(r.get("id", 0)) for r in rows), default=0) + 1,
        "记录编号": f"AUDIT-{uuid.uuid4().hex[:8]}",
        "查询语句": f"METRIC_BATCH.{action} batch_id={batch_id} fp={fingerprint} lines={count}",
        "执行耗时": "0ms",
        "扫描行数": str(count),
        "返回行数": str(count),
        "数据来源": "metric_batch",
        "操作用户": "metric_batch_engine",
        "查询时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "正常",
        "pending": False,
        "abnormal": False,
    }
    rows.append(audit)

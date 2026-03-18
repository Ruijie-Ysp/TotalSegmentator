"""
FastAPI 后端 - 专业医学影像分割 API
提供 TotalSegmentator 分割、切片提取、元数据查询等功能
支持 2D 图像、DICOM ZIP 智能选择、调试模式等高级功能
"""
import os
from pathlib import Path

# === 加载 .env 环境变量（必须在其他导入之前）===
from dotenv import load_dotenv
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
    print(f"✓ 已加载环境配置: {_ENV_FILE}")

# === MPS (Apple Silicon) 环境变量 - 必须在导入 torch 之前设置 ===
# 设置 MPS 内存分配策略（仅在 macOS 上生效）
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # 禁用限制，让系统自动管理
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # 启用 fallback 以处理不支持的操作
os.environ["DICOM2NIFTI_ALLOW_MISSING_SLICES"] = "True"  # 设置 DICOM 缺失切片处理

import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Literal
import json
import hashlib
import gc
import time
from datetime import datetime, timezone

import numpy as np
import nibabel as nib
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import io
from PIL import Image
import torch

# TotalSegmentator imports
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map

# LLM 问答生成器
from api.llm_qa_generator import generate_qa_pairs, generate_batch_qa, get_organ_cn

# 全局调试模式（通过环境变量启用）
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

# 检测 MPS 可用性
MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if MPS_AVAILABLE:
    print("✓ MPS (Apple Silicon GPU) 已启用")

# 导入数据库和缓存模块
from contextlib import asynccontextmanager
from api.database import init_db, check_db_connection
from api.cache import check_redis_connection

@asynccontextmanager
async def lifespan(app):
    """应用生命周期管理 - 启动时初始化数据库和缓存连接"""
    print("🚀 正在初始化服务...")

    # 初始化 PostgreSQL 数据库（启用数据库模式时）
    postgres_port = os.getenv('POSTGRES_PORT', '5432')
    if USE_DATABASE:
        try:
            await init_db()
            if await check_db_connection():
                print(f"✓ PostgreSQL 已连接 (localhost:{postgres_port})")
            else:
                _disable_database_mode(f"postgres unreachable localhost:{postgres_port}")
        except Exception as e:
            _disable_database_mode(f"postgres init failed localhost:{postgres_port}", e)
    else:
        print("ℹ 数据库模式未启用，使用本地会话存储")

    # 检查 Redis 连接
    redis_port = os.getenv('REDIS_PORT', '6379')
    try:
        if await check_redis_connection():
            print(f"✓ Redis 已连接 (localhost:{redis_port})")
        else:
            print(f"⚠ Redis 连接失败 (localhost:{redis_port})")
    except Exception as e:
        print(f"⚠ Redis 初始化失败: {e}")

    print("✓ 服务初始化完成")

    yield

    # 关闭时清理
    print("👋 服务正在关闭...")

app = FastAPI(
    title="TotalSegmentator API",
    description="专业医学影像分割 API - 支持 117+ 解剖结构自动分割",
    version="2.0.0",
    debug=DEBUG_MODE,
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入配置模块
from api.config import (
    load_config, save_config, get_config, update_llm_config,
    load_sessions, save_sessions, save_single_session, delete_session_file,
    AppConfig, LLMConfig, LabelStudioConfig,
    update_labelstudio_config, get_labelstudio_config,
    # QA 模板配置持久化
    get_qa_template_state, update_template_selection, update_template_selection_batch,
    update_custom_templates, reset_qa_templates_config, get_template_selection_state,
    get_custom_templates, get_all_template_selection_states
)

# 获取项目根目录（api 目录的父目录）
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent

# 全局存储（使用绝对路径确保一致性）
DATA_DIR = _PROJECT_ROOT / "api_data"
DATA_DIR.mkdir(exist_ok=True)
DEBUG_DIR = _PROJECT_ROOT / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# 数据持久化模式
USE_DATABASE = os.getenv("USE_DATABASE", "false").lower() == "true"

# 运行时数据库可用性（支持自动降级）
DATABASE_RUNTIME_AVAILABLE = USE_DATABASE

if USE_DATABASE:
    # 数据库模式 - 使用 PostgreSQL + Redis
    from api.task_repository import (
        TaskRepository, get_task, task_exists, create_task, update_task, delete_task
    )
    # 内存缓存仅用于运行时临时数据（如 _orig_data, _seg_data）
    SESSIONS: Dict[str, Any] = {}
    print("✓ 数据库模式已启用 (PostgreSQL + Redis)")
else:
    # 兼容模式 - 使用本地 JSON 文件
    SESSIONS = load_sessions()
    print(f"✓ 兼容模式: 已加载 {len(SESSIONS)} 个持久化 session")


def _get_env_int(name: str, default: int) -> int:
    """读取整型环境变量，读取失败时回退默认值"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"⚠ 环境变量 {name}={value} 非法，使用默认值 {default}")
        return default


MAX_RUNTIME_CACHE_SESSIONS = max(0, _get_env_int("MAX_RUNTIME_CACHE_SESSIONS", 1))
UPLOAD_READ_CHUNK_BYTES = max(256 * 1024, _get_env_int("UPLOAD_READ_CHUNK_BYTES", 4 * 1024 * 1024))

# 任务服务（兼容旧代码）
from api.task_service import TaskService, get_task_service

# Label Studio ML Backend
from api.labelstudio_ml_backend import router as ml_backend_router
app.include_router(ml_backend_router)


def _safe_filename(name: str) -> str:
    """生成安全的文件名"""
    keep = []
    for ch in (name or ""):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "input"


def _database_mode_enabled() -> bool:
    """当前请求是否使用数据库模式（运行时可降级）"""
    return USE_DATABASE and DATABASE_RUNTIME_AVAILABLE


def _disable_database_mode(reason: str, error: Optional[Exception] = None) -> None:
    """数据库不可用时降级到本地模式，避免业务接口直接 500"""
    global DATABASE_RUNTIME_AVAILABLE

    if not USE_DATABASE:
        return

    was_enabled = DATABASE_RUNTIME_AVAILABLE
    DATABASE_RUNTIME_AVAILABLE = False

    if was_enabled:
        msg = f"⚠ 数据库模式已降级为本地模式: {reason}"
        if error is not None:
            msg = f"{msg}; error={error}"
        print(msg)

    # 若内存为空，尝试加载本地持久化会话作为回退
    if not SESSIONS:
        try:
            fallback_sessions = load_sessions()
            if fallback_sessions:
                SESSIONS.update(fallback_sessions)
                print(f"✓ 回退模式: 已加载 {len(fallback_sessions)} 个本地 session")
        except Exception as load_err:
            print(f"⚠ 回退模式加载本地 session 失败: {load_err}")


async def _get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """获取会话数据（支持数据库和内存两种模式）"""
    if _database_mode_enabled():
        # 先检查内存缓存（运行时数据）
        if session_id in SESSIONS:
            session = SESSIONS[session_id]
            _touch_session(session)
            return session
        # 从数据库加载
        try:
            task_data = await get_task(session_id)
            if task_data:
                SESSIONS[session_id] = task_data
                _touch_session(task_data)
                return task_data
            return None
        except Exception as e:
            _disable_database_mode("get_task failed", e)

    session = SESSIONS.get(session_id)
    if session:
        _touch_session(session)
    return session


async def _session_exists(session_id: str) -> bool:
    """检查会话是否存在"""
    if _database_mode_enabled():
        if session_id in SESSIONS:
            return True
        try:
            return await task_exists(session_id)
        except Exception as e:
            _disable_database_mode("task_exists failed", e)

    return session_id in SESSIONS


async def _save_session(session_id: str, session: Dict[str, Any]) -> None:
    """保存会话数据（排除运行时数据如 _orig_data, _seg_data）"""
    # 过滤掉以 _ 开头的运行时数据
    data_to_save = {k: v for k, v in session.items() if not k.startswith("_")}
    _touch_session(session)

    if _database_mode_enabled():
        try:
            await update_task(session_id, data_to_save)
            return
        except Exception as e:
            _disable_database_mode("update_task failed", e)

    save_single_session(session_id, data_to_save)


async def _delete_session(session_id: str) -> bool:
    """删除会话"""
    session = SESSIONS.pop(session_id, None)
    if session:
        _release_runtime_arrays(session)
        gc.collect()
        _release_torch_memory()

    if _database_mode_enabled():
        try:
            return await delete_task(session_id)
        except Exception as e:
            _disable_database_mode("delete_task failed", e)

    delete_session_file(session_id)
    return True


def _touch_session(session: Dict[str, Any]) -> None:
    """更新会话最近访问时间"""
    if session is not None:
        session["_last_access_ts"] = time.time()


def _release_runtime_arrays(session: Dict[str, Any]) -> bool:
    """释放会话中占用内存较大的运行时数组"""
    released = False
    if not session:
        return released

    if session.pop("_orig_data", None) is not None:
        released = True
    if session.pop("_seg_data", None) is not None:
        released = True
    if session.pop("_orig_img", None) is not None:
        released = True
    if session.pop("_seg_img", None) is not None:
        released = True
    return released


def _release_torch_memory() -> None:
    """释放 Torch 设备缓存，避免长期占用系统/GPU 内存"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        mps = getattr(torch, "mps", None)
        if mps and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        pass


def _prune_runtime_session_cache() -> None:
    """将运行时数组缓存限制在可控数量，防止会话越多内存越涨"""
    _prune_runtime_session_cache_with_limit(
        max_cached_sessions=MAX_RUNTIME_CACHE_SESSIONS,
        dry_run=False,
        run_gc=True,
        release_torch=True,
    )


def _estimate_obj_nbytes(obj: Any) -> int:
    """估算对象占用字节数（优先 numpy 数组）"""
    if obj is None:
        return 0

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)

    nbytes = getattr(obj, "nbytes", None)
    if isinstance(nbytes, (int, float)):
        return int(nbytes)

    return 0


def _format_bytes(num_bytes: int) -> str:
    """将字节数格式化为可读字符串"""
    if num_bytes is None:
        return "unknown"

    size = float(max(0, num_bytes))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{int(size)} B"


def _get_process_rss_bytes() -> Tuple[Optional[int], str]:
    """获取当前进程 RSS（常驻内存）"""
    # 1) 优先 psutil（最准确）
    try:
        import psutil

        rss = int(psutil.Process(os.getpid()).memory_info().rss)
        return rss, "psutil"
    except Exception:
        pass

    # 2) Linux /proc
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
            if len(parts) >= 2:
                rss_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return rss_pages * page_size, "proc_statm"
    except Exception:
        pass

    # 3) resource（回退；在部分系统上是 maxrss）
    try:
        import resource

        ru_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS 通常是 bytes，Linux 通常是 KB
        if os.uname().sysname == "Darwin":
            return ru_maxrss, "resource_ru_maxrss_bytes"
        return ru_maxrss * 1024, "resource_ru_maxrss_kb"
    except Exception:
        return None, "unavailable"


def _session_runtime_memory_stats(session: Dict[str, Any]) -> Dict[str, int]:
    """统计单个会话在运行时缓存中的内存占用"""
    orig_bytes = _estimate_obj_nbytes(session.get("_orig_data"))
    seg_bytes = _estimate_obj_nbytes(session.get("_seg_data"))
    total_bytes = orig_bytes + seg_bytes
    return {
        "orig_bytes": orig_bytes,
        "seg_bytes": seg_bytes,
        "total_bytes": total_bytes,
    }


def _prune_runtime_session_cache_with_limit(
    max_cached_sessions: int,
    dry_run: bool = False,
    run_gc: bool = True,
    release_torch: bool = True,
) -> Dict[str, Any]:
    """按指定上限清理运行时数组缓存，返回清理统计信息"""
    cached_entries = []
    for sid, session in SESSIONS.items():
        stats = _session_runtime_memory_stats(session)
        total_bytes = stats["total_bytes"]
        if total_bytes <= 0:
            continue
        cached_entries.append(
            {
                "session_id": sid,
                "last_access_ts": float(session.get("_last_access_ts", 0.0)),
                "total_bytes": total_bytes,
            }
        )

    before_cached_sessions = len(cached_entries)
    before_total_bytes = sum(item["total_bytes"] for item in cached_entries)

    if before_cached_sessions == 0:
        return {
            "dry_run": dry_run,
            "max_cached_sessions": max_cached_sessions,
            "before": {
                "cached_sessions": 0,
                "total_bytes": 0,
                "total_human": _format_bytes(0),
            },
            "after": {
                "cached_sessions": 0,
                "total_bytes": 0,
                "total_human": _format_bytes(0),
            },
            "released": {
                "count": 0,
                "bytes": 0,
                "bytes_human": _format_bytes(0),
                "session_ids": [],
            },
            "kept_session_ids": [],
        }

    cached_entries.sort(key=lambda item: item["last_access_ts"], reverse=True)
    if max_cached_sessions <= 0:
        keep_ids = set()
    else:
        keep_ids = {item["session_id"] for item in cached_entries[:max_cached_sessions]}

    candidate_to_release = [item for item in cached_entries if item["session_id"] not in keep_ids]
    candidate_release_bytes = sum(item["total_bytes"] for item in candidate_to_release)

    released_session_ids: List[str] = []
    released_bytes = 0

    if not dry_run:
        for item in candidate_to_release:
            sid = item["session_id"]
            session = SESSIONS.get(sid)
            if not session:
                continue

            before_bytes = _session_runtime_memory_stats(session)["total_bytes"]
            released = _release_runtime_arrays(session)
            after_bytes = _session_runtime_memory_stats(session)["total_bytes"]

            if released:
                released_session_ids.append(sid)
                released_bytes += max(0, before_bytes - after_bytes)

        if released_session_ids and run_gc:
            gc.collect()
        if released_session_ids and release_torch:
            _release_torch_memory()

    after_entries = []
    for sid, session in SESSIONS.items():
        total_bytes = _session_runtime_memory_stats(session)["total_bytes"]
        if total_bytes > 0:
            after_entries.append({"session_id": sid, "total_bytes": total_bytes})

    after_cached_sessions = len(after_entries)
    after_total_bytes = sum(item["total_bytes"] for item in after_entries)

    return {
        "dry_run": dry_run,
        "max_cached_sessions": max_cached_sessions,
        "before": {
            "cached_sessions": before_cached_sessions,
            "total_bytes": before_total_bytes,
            "total_human": _format_bytes(before_total_bytes),
        },
        "after": {
            "cached_sessions": after_cached_sessions if not dry_run else max(before_cached_sessions - len(candidate_to_release), 0),
            "total_bytes": after_total_bytes if not dry_run else max(before_total_bytes - candidate_release_bytes, 0),
            "total_human": _format_bytes(after_total_bytes if not dry_run else max(before_total_bytes - candidate_release_bytes, 0)),
        },
        "released": {
            "count": len(released_session_ids) if not dry_run else len(candidate_to_release),
            "bytes": released_bytes if not dry_run else candidate_release_bytes,
            "bytes_human": _format_bytes(released_bytes if not dry_run else candidate_release_bytes),
            "session_ids": released_session_ids if not dry_run else [item["session_id"] for item in candidate_to_release],
        },
        "kept_session_ids": [item["session_id"] for item in cached_entries if item["session_id"] in keep_ids],
    }


def _collect_dicom_candidates(root: Path, max_files: int = 20000) -> tuple:
    """收集 DICOM 文件候选"""
    files: list = []
    truncated = False
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if "__MACOSX" in str(p):
                continue
            if p.suffix.lower() in (".dcm", ".dicom"):
                files.append(p)
            elif len(files) < max_files:
                files.append(p)
            if len(files) >= max_files:
                truncated = True
                break
    except Exception:
        pass
    return files, truncated


def _choose_best_dicom_subset(dicom_root: Path, max_files: int = 20000) -> Optional[Dict]:
    """智能选择最佳 DICOM 子集（多 Series/Timepoint 时）"""
    try:
        import pydicom
    except ImportError:
        return None

    files, _ = _collect_dicom_candidates(dicom_root, max_files=max_files)

    def get_tag(ds, name, default=None):
        try:
            v = getattr(ds, name, default)
            return str(v) if v is not None else default
        except Exception:
            return default

    studies: dict = {}
    parsed_any = False

    for f in files:
        try:
            if "__MACOSX" in f.parts:
                continue
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            sop = get_tag(ds, "SOPClassUID", None)
            modality = get_tag(ds, "Modality", None)
            if sop is None and modality is None:
                continue
            parsed_any = True

            study_uid = get_tag(ds, "StudyInstanceUID", "UNKNOWN_STUDY")
            series_uid = get_tag(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")
            series_desc = get_tag(ds, "SeriesDescription", "")
            manufacturer = get_tag(ds, "Manufacturer", "")

            temporal_pos = get_tag(ds, "TemporalPositionIdentifier", None)
            acquisition_num = get_tag(ds, "AcquisitionNumber", None)
            trigger_time = get_tag(ds, "TriggerTime", None)
            tp_key = temporal_pos or acquisition_num or trigger_time or "NA"

            instance_number = None
            try:
                instance_number_raw = getattr(ds, "InstanceNumber", None)
                if instance_number_raw is not None:
                    instance_number = int(instance_number_raw)
            except Exception:
                pass

            study = studies.setdefault(study_uid, {})
            series = study.setdefault(series_uid, {
                "modality": modality,
                "series_description": series_desc,
                "manufacturer": manufacturer,
                "timepoints": {}
            })
            tp = series["timepoints"].setdefault(tp_key, {
                "files": [],
                "inst_set": set(),
                "inst_min": None,
                "inst_max": None,
            })
            tp["files"].append(f)
            if instance_number is not None:
                tp["inst_set"].add(instance_number)
                if tp["inst_min"] is None or instance_number < tp["inst_min"]:
                    tp["inst_min"] = instance_number
                if tp["inst_max"] is None or instance_number > tp["inst_max"]:
                    tp["inst_max"] = instance_number
        except Exception:
            continue

    if not parsed_any or not studies:
        return None

    def series_desc_bonus(desc: str) -> float:
        d = (desc or "").lower()
        bonus = 1.0
        if "std" in d:
            bonus *= 1.10
        if "lung" in d:
            bonus *= 0.95
        if "localizer" in d or "scout" in d:
            bonus *= 0.10
        if "summary" in d:
            bonus *= 0.30
        return bonus

    candidates: list = []
    for study_uid, series_map in studies.items():
        for series_uid, series in series_map.items():
            tps = series.get("timepoints", {})
            if not tps:
                continue
            tp_candidates = []
            for tp_key, tp in tps.items():
                n_files = len(tp.get("files", []))
                inst_min = tp.get("inst_min")
                inst_max = tp.get("inst_max")
                inst_unique = len(tp.get("inst_set", set()))
                missing_est = None
                if inst_min is not None and inst_max is not None and inst_unique > 0:
                    expected = inst_max - inst_min + 1
                    missing_est = max(0, expected - inst_unique)
                score = float(n_files)
                if missing_est is not None and (inst_max - inst_min + 1) > 0:
                    expected = (inst_max - inst_min + 1)
                    miss_ratio = missing_est / expected if expected else 0.0
                    score *= (1.0 - min(0.9, miss_ratio))
                else:
                    score *= 0.85
                tp_candidates.append((score, tp_key, missing_est, n_files))

            tp_candidates.sort(reverse=True, key=lambda x: x[0])
            best_tp_score, best_tp_key, best_tp_missing, best_tp_files = tp_candidates[0]

            desc = series.get("series_description", "")
            tp_count = len(tps)
            score = best_tp_score
            if tp_count == 1:
                score *= 1.20
            else:
                score *= 0.90
            score *= series_desc_bonus(desc)

            candidates.append({
                "score": score,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "series_description": desc,
                "modality": series.get("modality"),
                "manufacturer": series.get("manufacturer"),
                "timepoint_key": best_tp_key,
                "files": tps[best_tp_key]["files"],
                "timepoint_count": tp_count,
                "files_count": best_tp_files,
            })

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x.get("score", 0))
    chosen = candidates[0]
    chosen["reason"] = (
        f"auto_selected: files={chosen.get('files_count')} "
        f"timepoint_count={chosen.get('timepoint_count')} "
        f"tp={chosen.get('timepoint_key')} "
        f"desc={chosen.get('series_description')!r}"
    )
    return chosen


def _stage_dicom_files(files: list, dst_dir: Path) -> tuple:
    """将选中的 DICOM 文件复制到目标目录"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in files:
        try:
            dst = dst_dir / src.name
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except Exception:
                shutil.copy2(src, dst)
            count += 1
        except Exception:
            continue
    return dst_dir, count


def _dicom_series_summary(dicom_root: Path, out_json_path: Path, max_files: int = 20000) -> dict:
    """生成 DICOM 系列统计报告（用于调试）"""
    summary: dict = {
        "dicom_root": str(dicom_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "max_files": max_files,
        "file_count_seen": 0,
        "file_count_parsed": 0,
        "truncated": False,
        "studies": {},
        "best_candidates": [],
        "warnings": []
    }

    try:
        import pydicom
    except Exception as e:
        summary["warnings"].append(f"pydicom_not_available: {e}")
        out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    files, truncated = _collect_dicom_candidates(dicom_root, max_files=max_files)
    summary["file_count_seen"] = len(files)
    summary["truncated"] = bool(truncated)

    def get_tag(ds, name, default=None):
        try:
            v = getattr(ds, name, default)
            return str(v) if v is not None else default
        except Exception:
            return default

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            sop = get_tag(ds, "SOPClassUID", None)
            modality = get_tag(ds, "Modality", None)
            if sop is None and modality is None:
                continue
            summary["file_count_parsed"] += 1

            study_uid = get_tag(ds, "StudyInstanceUID", "UNKNOWN_STUDY")
            series_uid = get_tag(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")

            studies = summary["studies"]
            study = studies.setdefault(study_uid, {"series": {}, "notes": []})
            series = study["series"].setdefault(series_uid, {
                "modality": modality,
                "series_description": get_tag(ds, "SeriesDescription", None),
                "files": 0,
            })
            series["files"] += 1
        except Exception:
            continue

    try:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        summary["warnings"].append(f"write_summary_failed: {e}")

    return summary


def _convert_2d_to_nifti(input_path: Path, output_path: Path) -> Path:
    """将 2D 图像 (JPG/PNG) 转换为 NIfTI 格式"""
    img_2d = Image.open(input_path).convert("L")
    data_2d = np.array(img_2d)
    data_3d = data_2d.T[:, :, np.newaxis]
    affine = np.eye(4)
    new_img = nib.Nifti1Image(data_3d.astype(np.float32), affine)
    nib.save(new_img, str(output_path))
    return output_path


def _get_default_device() -> str:
    """自动检测最佳运行设备 - MPS 优先于 CPU"""
    # 优先级: CUDA GPU > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        return "gpu"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_session_path(path_str: str) -> Path:
    """将 session 中存储的相对路径转换为绝对路径"""
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    # 相对路径相对于项目根目录
    return _PROJECT_ROOT / path


def _ensure_session_data_loaded(session: dict) -> tuple:
    """
    确保 session 的数据已加载到内存中。
    如果内存中没有数据，从磁盘加载。
    返回 (orig_data, seg_data)，如果无法加载返回 (None, None)
    """
    _touch_session(session)
    orig_data = session.get("_orig_data")
    seg_data = session.get("_seg_data")

    orig_img = session.get("_orig_img")
    seg_img = session.get("_seg_img")

    # 如果数据已在内存中，直接返回
    if orig_data is not None and seg_data is not None:
        return orig_data, seg_data

    # 如果已经缓存 nibabel 图像对象，则从 dataobj 读取（避免 get_fdata 默认 float64 膨胀）
    if seg_data is None and seg_img is not None:
        try:
            seg_data = np.asarray(seg_img.dataobj)
            session["_seg_data"] = seg_data
        except Exception as e:
            print(f"[Data] Failed to read seg_data from cached img: {e}")

    if orig_data is None and orig_img is not None:
        try:
            orig_data = np.asarray(orig_img.dataobj)
            session["_orig_data"] = orig_data
        except Exception as e:
            print(f"[Data] Failed to read orig_data from cached img: {e}")

    if orig_data is not None and seg_data is not None:
        _prune_runtime_session_cache()
        return orig_data, seg_data

    # 尝试从磁盘加载分割数据
    if seg_data is None:
        output_path = _resolve_session_path(session.get("output_path"))
        if output_path and output_path.exists():
            try:
                seg_data = np.asarray(nib.load(str(output_path)).dataobj)
                session["_seg_data"] = seg_data
                print(f"[Data] Loaded seg_data from {output_path}")
            except Exception as e:
                print(f"[Data] Failed to load seg_data: {e}")
                return None, None

    # 尝试从磁盘加载原始数据
    if orig_data is None:
        nifti_path = None
        # 优先使用 converted_nifti
        candidate = _resolve_session_path(session.get("converted_nifti"))
        if candidate and candidate.exists():
            nifti_path = candidate

        if not nifti_path:
            # 其次使用 processed_input（如果是 nifti）
            candidate = _resolve_session_path(session.get("processed_input"))
            if candidate and candidate.exists():
                if str(candidate).endswith('.nii') or str(candidate).endswith('.nii.gz'):
                    nifti_path = candidate

        if not nifti_path:
            # 最后使用 input_path
            candidate = _resolve_session_path(session.get("input_path"))
            if candidate and candidate.exists():
                if str(candidate).endswith('.nii') or str(candidate).endswith('.nii.gz'):
                    nifti_path = candidate

        if nifti_path and nifti_path.exists():
            try:
                orig_data = np.asarray(nib.load(str(nifti_path)).dataobj)
                session["_orig_data"] = orig_data
                print(f"[Data] Loaded orig_data from {nifti_path}")
            except Exception as e:
                print(f"[Data] Failed to load orig_data: {e}")
                return None, None

    _prune_runtime_session_cache()

    return orig_data, seg_data


def _get_device_info() -> Dict[str, Any]:
    """获取详细的设备信息"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "recommended_device": _get_default_device(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        # 获取 GPU 内存信息
        if torch.cuda.device_count() > 0:
            info["cuda_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)

    if info["mps_available"]:
        info["mps_memory_limit"] = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        info["mps_fallback_enabled"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") == "1"

    return info


class SegmentRequest(BaseModel):
    task: str = "total"
    fast: bool = True
    device: str = "auto"  # auto, gpu, cpu, mps
    debug_mode: bool = False  # 调试模式：保留临时文件


class SliceRequest(BaseModel):
    session_id: str
    axis: str = "axial"  # axial, sagittal, coronal
    slice_idx: int = 0
    window_center: int = 40
    window_width: int = 400
    overlay_organs: List[str] = []
    overlay_opacity: float = 0.5


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/system")
async def get_system_info():
    """获取系统信息和推荐设备"""
    device_info = _get_device_info()
    return {
        "status": "ok",
        "device": device_info,
        "supported_formats": ["nifti", "dicom", "zip", "2d_image"],
        "supported_2d_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/debug/memory")
async def debug_memory(
    top_n: int = Query(20, ge=1, le=200, description="返回占用最大的前 N 个会话"),
    include_idle: bool = Query(True, description="是否包含未缓存数组的会话"),
):
    """内存诊断：进程 RSS + 每个 session 运行时缓存占用"""
    process_rss_bytes, process_rss_source = _get_process_rss_bytes()

    items = []
    runtime_total_bytes = 0
    cached_session_count = 0

    for sid, session in SESSIONS.items():
        stats = _session_runtime_memory_stats(session)
        total_bytes = stats["total_bytes"]
        if total_bytes > 0:
            cached_session_count += 1
            runtime_total_bytes += total_bytes

        if (not include_idle) and total_bytes <= 0:
            continue

        items.append(
            {
                "session_id": sid,
                "status": session.get("status"),
                "task": session.get("task"),
                "shape": session.get("shape"),
                "has_orig": session.get("_orig_data") is not None,
                "has_seg": session.get("_seg_data") is not None,
                "orig_bytes": stats["orig_bytes"],
                "seg_bytes": stats["seg_bytes"],
                "total_bytes": total_bytes,
                "total_human": _format_bytes(total_bytes),
                "last_access_ts": session.get("_last_access_ts"),
                "last_access_iso": datetime.fromtimestamp(
                    float(session.get("_last_access_ts", 0) or 0), tz=timezone.utc
                ).isoformat()
                if session.get("_last_access_ts")
                else None,
            }
        )

    items.sort(key=lambda item: item["total_bytes"], reverse=True)

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_cache_policy": {
            "max_runtime_cache_sessions": MAX_RUNTIME_CACHE_SESSIONS,
            "upload_read_chunk_bytes": UPLOAD_READ_CHUNK_BYTES,
        },
        "process": {
            "pid": os.getpid(),
            "rss_bytes": process_rss_bytes,
            "rss_human": _format_bytes(process_rss_bytes) if process_rss_bytes is not None else "unknown",
            "rss_source": process_rss_source,
        },
        "sessions": {
            "total": len(SESSIONS),
            "cached_with_arrays": cached_session_count,
            "runtime_arrays_total_bytes": runtime_total_bytes,
            "runtime_arrays_total_human": _format_bytes(runtime_total_bytes),
        },
        "top": items[:top_n],
    }


@app.post("/api/debug/memory/prune")
async def debug_memory_prune(
    keep_latest_sessions: Optional[int] = Query(None, ge=0, le=500, description="清理后保留的最新缓存会话数；为空则使用默认策略"),
    dry_run: bool = Query(False, description="仅预览清理结果，不实际释放"),
    run_gc: bool = Query(True, description="清理后是否执行 gc.collect()"),
    release_torch: bool = Query(True, description="清理后是否释放 Torch 设备缓存"),
):
    """手动触发运行时数组缓存清理（应急使用）"""
    target_keep = MAX_RUNTIME_CACHE_SESSIONS if keep_latest_sessions is None else keep_latest_sessions

    rss_before, rss_before_source = _get_process_rss_bytes()
    summary = _prune_runtime_session_cache_with_limit(
        max_cached_sessions=target_keep,
        dry_run=dry_run,
        run_gc=run_gc,
        release_torch=release_torch,
    )
    rss_after, rss_after_source = _get_process_rss_bytes()

    rss_released = None
    if rss_before is not None and rss_after is not None:
        rss_released = max(0, rss_before - rss_after)

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "requested_keep_latest_sessions": keep_latest_sessions,
            "effective_keep_latest_sessions": target_keep,
            "dry_run": dry_run,
            "run_gc": run_gc,
            "release_torch": release_torch,
        },
        "prune": summary,
        "process_rss": {
            "before_bytes": rss_before,
            "before_human": _format_bytes(rss_before) if rss_before is not None else "unknown",
            "before_source": rss_before_source,
            "after_bytes": rss_after,
            "after_human": _format_bytes(rss_after) if rss_after is not None else "unknown",
            "after_source": rss_after_source,
            "released_bytes": rss_released,
            "released_human": _format_bytes(rss_released) if rss_released is not None else "unknown",
        },
    }


@app.get("/api/task-types")
async def get_available_tasks():
    """获取所有可用的分割任务类型及其器官列表"""
    tasks_info = {}
    for task_name, organs in class_map.items():
        tasks_info[task_name] = {
            "organ_count": len(organs),
            "organs": {idx: name for idx, name in organs.items()}
        }
    return tasks_info


@app.get("/api/organs/{task}")
async def get_task_organs(task: str):
    """获取指定任务的器官列表"""
    if task not in class_map:
        raise HTTPException(status_code=404, detail=f"Task '{task}' not found")
    return {
        "task": task,
        "organs": class_map[task]
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传医学影像文件，返回 session_id
    支持格式: NIfTI (.nii, .nii.gz), DICOM (.dcm), ZIP, 2D图像 (JPG/PNG)
    """
    session_id = hashlib.md5(f"{file.filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # 保存上传文件
    original_filename = file.filename or "upload"
    input_path = session_dir / original_filename
    with open(input_path, "wb") as f:
        while True:
            chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
            if not chunk:
                break
            f.write(chunk)

    # 检测文件类型
    suffix = input_path.suffix.lower()
    file_type = "unknown"
    if suffix in [".nii"] or original_filename.endswith(".nii.gz"):
        file_type = "nifti"
    elif suffix in [".dcm", ".dicom"]:
        file_type = "dicom"
    elif suffix == ".zip":
        file_type = "zip"
    elif suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        file_type = "2d_image"

    task_data = {
        "input_path": str(input_path),
        "filename": original_filename,
        "file_type": file_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "uploaded",
        "session_dir": str(session_dir)
    }

    if _database_mode_enabled():
        # 数据库模式
        try:
            await create_task(session_id, task_data)
            # 内存中保留运行时数据引用
            SESSIONS[session_id] = task_data.copy()
        except Exception as e:
            _disable_database_mode("create_task failed in upload", e)
            SESSIONS[session_id] = task_data
            save_single_session(session_id, SESSIONS[session_id])
    else:
        # 兼容模式
        SESSIONS[session_id] = task_data
        save_single_session(session_id, SESSIONS[session_id])

    return {
        "session_id": session_id,
        "filename": original_filename,
        "file_type": file_type
    }


@app.post("/api/segment/{session_id}")
async def run_segmentation(session_id: str, request: SegmentRequest):
    """
    运行分割任务
    支持: NIfTI, DICOM, ZIP (含多 Series 智能选择), 2D 图像
    """
    # 使用辅助函数检查并获取会话
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 确保内存中有会话引用
    SESSIONS[session_id] = session
    session_dir = Path(session.get("session_dir", DATA_DIR / session_id))
    output_path = session_dir / "segmentation.nii.gz"
    debug_out_dir = None

    try:
        session["status"] = "processing"
        session["progress"] = 0
        session["progress_message"] = "准备处理..."
        file_type = session.get("file_type", "unknown")
        input_path = Path(session["input_path"])
        current_input = str(input_path)
        nifti_input_path = session_dir / "input_converted.nii.gz"

        # === 处理 2D 图像 ===
        if file_type == "2d_image":
            print(f"Converting 2D image {input_path.name} to NIfTI...")
            _convert_2d_to_nifti(input_path, nifti_input_path)
            current_input = str(nifti_input_path)
            session["converted_nifti"] = str(nifti_input_path)

        # === 处理 ZIP 文件 ===
        elif file_type == "zip":
            print(f"Extracting ZIP file {input_path.name}...")
            zip_extract_dir = session_dir / "extracted_zip"
            zip_extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(zip_extract_dir)

            # 智能选择最佳 DICOM Series/Timepoint
            chosen = _choose_best_dicom_subset(zip_extract_dir)
            if chosen and chosen.get("files"):
                selected_dir = session_dir / "selected_dicom"
                _, staged = _stage_dicom_files(chosen["files"], selected_dir)
                current_input = str(selected_dir)
                session["dicom_selection"] = {
                    "series_description": chosen.get("series_description"),
                    "series_uid": chosen.get("series_uid"),
                    "timepoint": chosen.get("timepoint_key"),
                    "files_count": staged,
                    "reason": chosen.get("reason")
                }
                print(f"Selected DICOM subset: {chosen.get('reason')}")
            else:
                # 回退：选择包含最多 .dcm 文件的目录
                dicom_dirs = []
                for root, dirs, files_ in os.walk(zip_extract_dir):
                    if any(f.lower().endswith(('.dcm', '.dicom')) for f in files_):
                        dicom_dirs.append(root)
                if dicom_dirs:
                    current_input = sorted(
                        dicom_dirs,
                        key=lambda d: len([f for f in os.listdir(d) if f.lower().endswith(('.dcm', '.dicom'))]),
                        reverse=True
                    )[0]
                    print(f"Found DICOM directory (fallback): {current_input}")
                else:
                    current_input = str(zip_extract_dir)

        # === 处理单个 DICOM ===
        elif file_type == "dicom":
            dicom_dir = session_dir / "single_dicom"
            dicom_dir.mkdir(exist_ok=True)
            shutil.copy(input_path, dicom_dir / input_path.name)
            current_input = str(dicom_dir)

        # === 确定运行设备 ===
        device = request.device
        if device == "auto":
            device = _get_default_device()
        session["device_used"] = device

        # 更新进度：准备运行分割
        session["progress"] = 20
        session["progress_message"] = f"准备在 {device} 上运行分割..."

        print(f"Running task {request.task} on {device}...")

        # 更新进度：开始分割
        session["progress"] = 30
        session["progress_message"] = "正在运行 TotalSegmentator 分割模型..."

        # === 运行 TotalSegmentator ===
        result = await run_in_threadpool(
            totalsegmentator,
            input=current_input,
            output=str(output_path),
            task=request.task,
            fast=request.fast,
            device=device,
            ml=True,
        )

        # 更新进度：分割完成，处理结果
        session["progress"] = 80
        session["progress_message"] = "分割完成，正在处理结果..."

        if isinstance(result, tuple):
            result_img = result[0]
        else:
            result_img = result

        # 保存分割结果
        nib.save(result_img, str(output_path))

        # 用 dataobj 保持原始 dtype，避免 get_fdata() 默认 float64 造成内存翻倍
        seg_data = np.asarray(result_img.dataobj)

        # 尝试加载原始图像
        orig_data = np.zeros_like(seg_data)
        ct_nifti_path = None

        try:
            converted_path = session.get("converted_nifti")
            if converted_path and Path(converted_path).exists():
                orig_data = np.asarray(nib.load(converted_path).dataobj)
                ct_nifti_path = converted_path
            elif input_path.suffix in [".gz", ".nii"] or str(input_path).endswith(".nii.gz"):
                orig_data = np.asarray(nib.load(str(input_path)).dataobj)
                ct_nifti_path = str(input_path)
            else:
                # 对于 DICOM 输入，尝试使用 dicom2nifti 转换
                dicom_input = Path(current_input)
                if dicom_input.is_dir():
                    ct_nifti_output = session_dir / "ct_original.nii.gz"
                    try:
                        import dicom2nifti
                        dicom2nifti.convert_directory(str(dicom_input), str(session_dir), compression=True, reorient=True)
                        # 查找转换后的文件
                        nifti_files = list(session_dir.glob("*.nii.gz"))
                        if nifti_files:
                            # 选择最大的文件（通常是主序列）
                            largest_nifti = max(nifti_files, key=lambda f: f.stat().st_size)
                            largest_nifti.rename(ct_nifti_output)
                            orig_data = np.asarray(nib.load(str(ct_nifti_output)).dataobj)
                            ct_nifti_path = str(ct_nifti_output)
                            print(f"Converted DICOM to NIfTI: {ct_nifti_path}")
                    except Exception as conv_err:
                        print(f"DICOM to NIfTI conversion failed: {conv_err}")
                        # 回退：从分割结果的 affine 创建一个假的 CT NIfTI
                        ct_nifti_output = session_dir / "ct_original.nii.gz"
                        fake_ct = nib.Nifti1Image(orig_data, result_img.affine)
                        nib.save(fake_ct, str(ct_nifti_output))
                        ct_nifti_path = str(ct_nifti_output)
        except Exception as e:
            print(f"Warning: Could not load original image: {e}")

        # 保存 CT NIfTI 路径到会话
        if ct_nifti_path:
            session["converted_nifti"] = ct_nifti_path

        # 更新进度：完成
        session["progress"] = 100
        session["progress_message"] = "处理完成！"

        session.update({
            "status": "completed",
            "task": request.task,
            "output_path": str(output_path),
            "shape": list(seg_data.shape),
            "organs": list(class_map.get(request.task, {}).values()),
            "processed_input": current_input
        })

        # 缓存数据用于切片提取（不保存到文件，运行时缓存）
        session["_orig_data"] = orig_data
        session["_seg_data"] = seg_data
        session["_orig_img"] = None
        session["_seg_img"] = None
        _touch_session(session)
        _prune_runtime_session_cache()

        # 持久化保存（使用辅助函数，自动选择数据库或文件模式）
        await _save_session(session_id, session)

        # 调试模式：保留临时文件和生成报告
        debug_info = None
        if request.debug_mode:
            debug_prefix = f"debug_{_safe_filename(input_path.stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            debug_out_dir = DEBUG_DIR / debug_prefix
            debug_out_dir.mkdir(parents=True, exist_ok=True)

            # 保存 DICOM 统计报告
            if file_type == "zip":
                zip_extract_dir = session_dir / "extracted_zip"
                if zip_extract_dir.exists():
                    summary_path = debug_out_dir / "dicom_series_summary.json"
                    _dicom_series_summary(zip_extract_dir, summary_path)
                    debug_info = {"debug_dir": str(debug_out_dir), "summary": str(summary_path)}

            session["debug_info"] = debug_info

        return {
            "session_id": session_id,
            "status": "completed",
            "shape": session["shape"],
            "organ_count": len(session["organs"]),
            "organs": session["organs"],
            "file_type": file_type,
            "device_used": device,
            "debug_info": debug_info
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        _release_torch_memory()
        gc.collect()
        session["status"] = "error"
        session["error"] = str(e)
        session["progress"] = 0
        session["progress_message"] = f"处理失败: {str(e)[:100]}"

        # 调试模式：即使出错也保存信息
        if request.debug_mode:
            try:
                debug_prefix = f"debug_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                debug_out_dir = DEBUG_DIR / debug_prefix
                debug_out_dir.mkdir(parents=True, exist_ok=True)

                # 保存错误信息
                error_info = {
                    "error": str(e),
                    "session_id": session_id,
                    "file_type": session.get("file_type"),
                    "input_path": session.get("input_path"),
                    "timestamp": datetime.now().isoformat()
                }
                error_path = debug_out_dir / "error_info.json"
                error_path.write_text(json.dumps(error_info, ensure_ascii=False, indent=2))

                # 保存 DICOM 统计
                zip_extract_dir = session_dir / "extracted_zip"
                if zip_extract_dir.exists():
                    summary_path = debug_out_dir / "dicom_series_summary.json"
                    _dicom_series_summary(zip_extract_dir, summary_path)

                session["debug_info"] = {"debug_dir": str(debug_out_dir)}
            except Exception as debug_e:
                print(f"Debug save failed: {debug_e}")

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metadata/{session_id}")
async def get_metadata(session_id: str):
    """获取影像元数据"""
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 检查关键数据是否还可用
    data_available = False
    if session.get("status") == "completed":
        # 检查分割数据是否存在
        output_path = session.get("output_path")
        if output_path and Path(output_path).exists():
            data_available = True
        # 检查内存缓存的数据
        elif session.get("_seg_data") is not None:
            data_available = True

    return {
        "session_id": session_id,
        "filename": session.get("filename"),
        "status": session.get("status") if data_available else "expired",
        "shape": session.get("shape"),
        "task": session.get("task"),
        "organs": session.get("organs", []),
        "data_available": data_available
    }


def apply_window(data: np.ndarray, center: int, width: int, output_uint8: bool = True) -> np.ndarray:
    """
    应用窗宽窗位（优化版：保留精度）

    Args:
        data: 输入数据（通常是 CT 值）
        center: 窗位
        width: 窗宽
        output_uint8: 是否输出 uint8（True 兼容旧代码，False 保留 float32 精度）

    Returns:
        窗口化后的数据（uint8 或 float32）
    """
    min_val = center - width / 2
    max_val = center + width / 2
    windowed = np.clip(data, min_val, max_val)

    if output_uint8:
        # 兼容旧代码：输出 uint8
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        # ✅ 优化：保留 float32 精度，归一化到 0-1 范围
        windowed = ((windowed - min_val) / (max_val - min_val)).astype(np.float32)

    return windowed


def get_organ_colors():
    """生成器官颜色映射 - 使用专业的医学影像颜色"""
    colors = {
        # 实体器官 - 暖色系
        "liver": [139, 69, 19],       # 褐色
        "spleen": [148, 0, 211],      # 紫色
        "pancreas": [255, 165, 0],    # 橙色
        "kidney_left": [220, 20, 60], # 深红
        "kidney_right": [255, 99, 71],# 番茄红
        "gallbladder": [0, 128, 0],   # 绿色
        "stomach": [255, 192, 203],   # 粉色

        # 肺部 - 蓝色系
        "lung_upper_lobe_left": [70, 130, 180],
        "lung_lower_lobe_left": [100, 149, 237],
        "lung_upper_lobe_right": [65, 105, 225],
        "lung_middle_lobe_right": [30, 144, 255],
        "lung_lower_lobe_right": [0, 191, 255],

        # 心血管 - 红色系
        "heart": [220, 20, 60],
        "aorta": [255, 0, 0],
        "inferior_vena_cava": [0, 0, 139],
        "portal_vein_and_splenic_vein": [75, 0, 130],

        # 骨骼 - 白/灰色系
        "skull": [245, 245, 245],
        "sternum": [211, 211, 211],
        "vertebrae_L1": [192, 192, 192],
        "vertebrae_L2": [169, 169, 169],

        # 肌肉 - 棕色系
        "gluteus_maximus_left": [139, 90, 43],
        "gluteus_maximus_right": [160, 82, 45],
    }
    return colors


ORGAN_COLORS = get_organ_colors()


# 器官中文名称映射
ORGAN_NAMES_CN = {
    # 腹部器官
    "liver": "肝脏", "spleen": "脾脏", "pancreas": "胰腺", "gallbladder": "胆囊",
    "stomach": "胃", "small_bowel": "小肠", "duodenum": "十二指肠", "colon": "结肠",
    # 泌尿系统
    "kidney_left": "左肾", "kidney_right": "右肾", "urinary_bladder": "膀胱",
    "kidney_cyst_left": "左肾囊肿", "kidney_cyst_right": "右肾囊肿", "prostate": "前列腺",
    # 肺部
    "lung_upper_lobe_left": "左肺上叶", "lung_lower_lobe_left": "左肺下叶",
    "lung_upper_lobe_right": "右肺上叶", "lung_middle_lobe_right": "右肺中叶",
    "lung_lower_lobe_right": "右肺下叶", "lung_left": "左肺", "lung_right": "右肺",
    "trachea": "气管", "esophagus": "食管",
    # 心血管系统
    "heart": "心脏", "aorta": "主动脉", "pulmonary_vein": "肺静脉",
    "inferior_vena_cava": "下腔静脉", "superior_vena_cava": "上腔静脉",
    "portal_vein_and_splenic_vein": "门静脉和脾静脉", "brachiocephalic_trunk": "头臂干",
    "subclavian_artery_right": "右锁骨下动脉", "subclavian_artery_left": "左锁骨下动脉",
    "common_carotid_artery_right": "右颈总动脉", "common_carotid_artery_left": "左颈总动脉",
    "iliac_artery_left": "左髂动脉", "iliac_artery_right": "右髂动脉",
    "iliac_vena_left": "左髂静脉", "iliac_vena_right": "右髂静脉",
    "pulmonary_artery": "肺动脉", "heart_myocardium": "心肌",
    "heart_atrium_left": "左心房", "heart_ventricle_left": "左心室",
    "heart_atrium_right": "右心房", "heart_ventricle_right": "右心室",
    # 骨骼系统
    "skull": "颅骨", "humerus_left": "左肱骨", "humerus_right": "右肱骨",
    "scapula_left": "左肩胛骨", "scapula_right": "右肩胛骨",
    "clavicula_left": "左锁骨", "clavicula_right": "右锁骨", "sternum": "胸骨",
    "femur_left": "左股骨", "femur_right": "右股骨",
    "hip_left": "左髋骨", "hip_right": "右髋骨", "sacrum": "骶骨",
    # 脊柱
    "vertebrae_S1": "骶椎S1", "vertebrae_L5": "腰椎L5", "vertebrae_L4": "腰椎L4",
    "vertebrae_L3": "腰椎L3", "vertebrae_L2": "腰椎L2", "vertebrae_L1": "腰椎L1",
    "vertebrae_T12": "胸椎T12", "vertebrae_T11": "胸椎T11", "vertebrae_T10": "胸椎T10",
    "vertebrae_T9": "胸椎T9", "vertebrae_T8": "胸椎T8", "vertebrae_T7": "胸椎T7",
    "vertebrae_T6": "胸椎T6", "vertebrae_T5": "胸椎T5", "vertebrae_T4": "胸椎T4",
    "vertebrae_T3": "胸椎T3", "vertebrae_T2": "胸椎T2", "vertebrae_T1": "胸椎T1",
    "vertebrae_C7": "颈椎C7", "vertebrae_C6": "颈椎C6", "vertebrae_C5": "颈椎C5",
    "vertebrae_C4": "颈椎C4", "vertebrae_C3": "颈椎C3", "vertebrae_C2": "颈椎C2",
    "vertebrae_C1": "颈椎C1", "vertebrae": "椎骨", "intervertebral_discs": "椎间盘",
    "spinal_cord": "脊髓", "spinal_canal": "椎管",
    # 肋骨
    "rib_left_1": "左第1肋", "rib_left_2": "左第2肋", "rib_left_3": "左第3肋",
    "rib_left_4": "左第4肋", "rib_left_5": "左第5肋", "rib_left_6": "左第6肋",
    "rib_left_7": "左第7肋", "rib_left_8": "左第8肋", "rib_left_9": "左第9肋",
    "rib_left_10": "左第10肋", "rib_left_11": "左第11肋", "rib_left_12": "左第12肋",
    "rib_right_1": "右第1肋", "rib_right_2": "右第2肋", "rib_right_3": "右第3肋",
    "rib_right_4": "右第4肋", "rib_right_5": "右第5肋", "rib_right_6": "右第6肋",
    "rib_right_7": "右第7肋", "rib_right_8": "右第8肋", "rib_right_9": "右第9肋",
    "rib_right_10": "右第10肋", "rib_right_11": "右第11肋", "rib_right_12": "右第12肋",
    # 肌肉
    "gluteus_maximus_left": "左臀大肌", "gluteus_maximus_right": "右臀大肌",
    "gluteus_medius_left": "左臀中肌", "gluteus_medius_right": "右臀中肌",
    "iliopsoas_left": "左髂腰肌", "iliopsoas_right": "右髂腰肌",
    "autochthon_left": "左竖脊肌", "autochthon_right": "右竖脊肌",
    # 腺体
    "adrenal_gland_right": "右肾上腺", "adrenal_gland_left": "左肾上腺",
    "thyroid_gland": "甲状腺",
    # 脑部
    "brain": "脑", "face": "面部",
    "eyeball_left": "左眼球", "eyeball_right": "右眼球",
    # 软骨
    "costal_cartilages": "肋软骨",
}


def get_organ_cn_name(organ: str) -> str:
    """获取器官中文名称"""
    return ORGAN_NAMES_CN.get(organ, organ.replace("_", " "))


def extract_contour(mask: np.ndarray, thickness: int = 2, use_precise: bool = False) -> np.ndarray:
    """
    提取 mask 的轮廓线（优化版：支持专业算法）

    Args:
        mask: 二值掩码
        thickness: 轮廓线粗细（1-3 推荐）
        use_precise: 是否使用精确算法（Marching Squares，True 更精确但稍慢）

    Returns:
        轮廓二值图像
    """
    if use_precise:
        # ✅ 优化：使用 Marching Squares 算法（更精确）
        try:
            from skimage import measure
            contour_img = np.zeros_like(mask, dtype=np.uint8)

            # 查找轮廓
            contours = measure.find_contours(mask.astype(float), 0.5)

            for contour in contours:
                # 转换为整数坐标并绘制
                contour = contour.astype(int)
                # 限制在图像范围内
                contour[:, 0] = np.clip(contour[:, 0], 0, mask.shape[0] - 1)
                contour[:, 1] = np.clip(contour[:, 1], 0, mask.shape[1] - 1)

                # 绘制精细轮廓线
                for y, x in contour:
                    y_min = max(0, y - thickness // 2)
                    y_max = min(mask.shape[0], y + thickness // 2 + 1)
                    x_min = max(0, x - thickness // 2)
                    x_max = min(mask.shape[1], x + thickness // 2 + 1)
                    contour_img[y_min:y_max, x_min:x_max] = 1

            return contour_img > 0
        except ImportError:
            # 如果 scikit-image 不可用，回退到形态学方法
            pass

    # 兼容旧代码：使用形态学操作
    from scipy import ndimage
    dilated = ndimage.binary_dilation(mask, iterations=thickness)
    eroded = ndimage.binary_erosion(mask, iterations=thickness)
    contour = dilated.astype(np.uint8) - eroded.astype(np.uint8)
    return contour > 0


def smart_resize_image(img: np.ndarray, max_size: int = 1024, min_size: int = 256) -> np.ndarray:
    """
    智能图像缩放（优化版：根据尺寸自动调整）

    Args:
        img: 输入图像（H, W, C）
        max_size: 最大边长
        min_size: 最小边长

    Returns:
        缩放后的图像
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    min_dim = min(h, w)

    # 如果图像尺寸在合理范围内，不缩放
    if min_dim <= max_size and max_dim >= min_size:
        return img

    # 计算缩放因子
    if max_dim > max_size:
        scale = max_size / max_dim
    elif min_dim < min_size:
        scale = min_size / min_dim
    else:
        return img

    new_h = int(h * scale)
    new_w = int(w * scale)

    # ✅ 使用 LANCZOS 高质量插值
    from PIL import Image
    if len(img.shape) == 3:
        pil_img = Image.fromarray(img.astype(np.uint8))
    else:
        pil_img = Image.fromarray(img.astype(np.uint8))

    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(resized)


def generate_overlay_image(
    orig_slice: np.ndarray,
    seg_slice: np.ndarray,
    organs: List[str],
    class_map_task: Dict[int, str],
    window_center: int = 40,
    window_width: int = 400,
    overlay_opacity: float = 1.0,
    overlay_mode: str = "fill_contour",
    contour_thickness: int = 1,
    draw_labels: bool = False
) -> str:
    """
    生成分割叠加图（单独的叠加图，不是并排对比）

    Args:
        orig_slice: 原始CT切片数据
        seg_slice: 分割标签切片数据
        organs: 要叠加显示的器官列表
        class_map_task: 标签到器官名的映射
        window_center: 窗位
        window_width: 窗宽
        overlay_opacity: 叠加透明度 (0-1)
        overlay_mode: 叠加模式 fill(填充), contour(轮廓), fill_contour(填充+轮廓)
        contour_thickness: 轮廓线粗细 (1-5)
        draw_labels: 是否绘制器官中文标签

    Returns:
        分割叠加图的 Base64 编码
    """
    import base64
    from PIL import ImageDraw, ImageFont

    # 应用窗宽窗位
    windowed = apply_window(orig_slice, window_center, window_width)

    # 灰度转 RGB
    overlay_image = np.stack([windowed, windowed, windowed], axis=-1).astype(np.float32)

    # 叠加器官分割
    name_to_idx = {v: k for k, v in class_map_task.items()}

    # 存储器官标签位置（用于后续绘制）
    label_positions = []

    for organ in organs:
        if organ in name_to_idx:
            organ_idx = name_to_idx[organ]
            mask = seg_slice == organ_idx
            if np.any(mask):
                color = np.array(ORGAN_COLORS.get(organ, [255, 255, 0]), dtype=np.float32)

                if overlay_mode in ("fill", "fill_contour"):
                    # 填充模式 - 使用较低的填充透明度
                    fill_opacity = overlay_opacity * 0.4 if overlay_mode == "fill_contour" else overlay_opacity
                    for c in range(3):
                        overlay_image[:, :, c] = np.where(
                            mask,
                            (1 - fill_opacity) * overlay_image[:, :, c] + fill_opacity * color[c],
                            overlay_image[:, :, c]
                        )

                if overlay_mode in ("contour", "fill_contour"):
                    # 轮廓线模式
                    contour = extract_contour(mask, thickness=contour_thickness)
                    for c in range(3):
                        overlay_image[:, :, c] = np.where(
                            contour,
                            color[c],  # 轮廓线完全不透明
                            overlay_image[:, :, c]
                        )

                # 计算器官中心位置（用于标签）
                if draw_labels:
                    y_coords, x_coords = np.where(mask)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        center_x = int(np.mean(x_coords))
                        center_y = int(np.mean(y_coords))
                        cn_name = get_organ_cn_name(organ)
                        label_positions.append({
                            "organ": organ,
                            "cn_name": cn_name,
                            "x": center_x,
                            "y": center_y,
                            "color": tuple(int(c) for c in color)
                        })

    # 翻转图像
    overlay_image = np.flipud(overlay_image).astype(np.uint8)

    # 转换为 PIL Image
    img = Image.fromarray(overlay_image)

    # 绘制器官标签
    if draw_labels and label_positions:
        draw = ImageDraw.Draw(img)
        img_height = img.height

        # 尝试加载中文字体
        font = None
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "C:\\Windows\\Fonts\\msyh.ttc",  # Windows 微软雅黑
            "C:\\Windows\\Fonts\\simhei.ttf",  # Windows 黑体
        ]
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 12)
                break
            except (IOError, OSError):
                continue

        if font is None:
            font = ImageFont.load_default()

        for label in label_positions:
            # 翻转 Y 坐标（因为图像已翻转）
            label_y = img_height - label["y"]
            label_x = label["x"]
            text = label["cn_name"]

            # 获取文本尺寸
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(text, font=font)

            # 绘制文本背景（不透明黑色）
            padding = 3
            bg_x1 = label_x - text_width // 2 - padding
            bg_y1 = label_y - text_height // 2 - padding
            bg_x2 = label_x + text_width // 2 + padding
            bg_y2 = label_y + text_height // 2 + padding
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0))

            # 绘制文本（白色，清晰可见）
            text_x = label_x - text_width // 2
            text_y = label_y - text_height // 2
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    img_buffer = io.BytesIO()
    # ✅ 优化：使用 PNG 压缩
    img.save(img_buffer, format='PNG', compress_level=9, optimize=True)
    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')


def generate_comparison_image(
    orig_slice: np.ndarray,
    seg_slice: np.ndarray,
    organs: List[str],
    class_map_task: Dict[int, str],
    window_center: int = 40,
    window_width: int = 400,
    overlay_opacity: float = 0.5
) -> str:
    """
    生成并排对比图（左原始CT + 右分割叠加）

    Args:
        orig_slice: 原始CT切片数据
        seg_slice: 分割标签切片数据
        organs: 要叠加显示的器官列表
        class_map_task: 标签到器官名的映射
        window_center: 窗位
        window_width: 窗宽
        overlay_opacity: 叠加透明度

    Returns:
        并排对比图的 Base64 编码
    """
    import base64

    # 应用窗宽窗位
    windowed = apply_window(orig_slice, window_center, window_width)

    # 左侧：原始 CT 图像（灰度转 RGB）
    left_image = np.stack([windowed, windowed, windowed], axis=-1).astype(np.float32)

    # 右侧：分割叠加图
    right_image = np.stack([windowed, windowed, windowed], axis=-1).astype(np.float32)

    # 在右侧图像上叠加器官分割
    name_to_idx = {v: k for k, v in class_map_task.items()}

    for organ in organs:
        if organ in name_to_idx:
            organ_idx = name_to_idx[organ]
            mask = seg_slice == organ_idx
            if np.any(mask):
                color = np.array(ORGAN_COLORS.get(organ, [255, 255, 0]), dtype=np.float32)
                # 填充模式
                for c in range(3):
                    right_image[:, :, c] = np.where(
                        mask,
                        (1 - overlay_opacity) * right_image[:, :, c] + overlay_opacity * color[c],
                        right_image[:, :, c]
                    )
                # 添加轮廓线增强边界
                contour = extract_contour(mask, thickness=2)
                for c in range(3):
                    right_image[:, :, c] = np.where(
                        contour,
                        color[c],  # 轮廓线完全不透明
                        right_image[:, :, c]
                    )

    # 翻转图像（与 get_slice 保持一致）
    left_image = np.flipud(left_image).astype(np.uint8)
    right_image = np.flipud(right_image).astype(np.uint8)

    # 并排拼接：左原始 + 中间分隔 + 右叠加
    separator = np.ones((left_image.shape[0], 4, 3), dtype=np.uint8) * 128  # 灰色分隔线
    combined = np.concatenate([left_image, separator, right_image], axis=1)

    # 在图像上添加标签说明
    from PIL import Image, ImageDraw
    img = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)

    # 添加标题文字（简单的像素字体）
    try:
        # 尝试加载字体
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    # 左上角标注"原始CT"
    draw.text((5, 5), "Original CT", fill=(255, 255, 0), font=font)
    # 右侧标注"分割叠加"
    draw.text((left_image.shape[1] + 10, 5), "Segmentation Overlay", fill=(255, 255, 0), font=font)

    # 转换为 Base64
    buffer = io.BytesIO()
    # ✅ 优化：使用 PNG 压缩
    img.save(buffer, format='PNG', compress_level=9, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.get("/api/slice/{session_id}")
async def get_slice(
    session_id: str,
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="视图方向: axial, sagittal, coronal"),
    slice_idx: int = Query(0, description="切片索引"),
    window_center: int = Query(40, description="窗位"),
    window_width: int = Query(400, gt=0, description="窗宽"),
    overlay_organs: str = Query("", description="叠加显示的器官，逗号分隔"),
    overlay_opacity: float = Query(0.5, description="叠加透明度 0-1"),
    overlay_mode: str = Query("fill", description="叠加模式: fill(填充), contour(轮廓), fill_contour(填充+轮廓)"),
    contour_thickness: int = Query(2, description="轮廓线粗细 1-5")
):
    """获取指定切片的图像"""
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载（从内存或磁盘）
    orig_data, seg_data = _ensure_session_data_loaded(session)

    if orig_data is None or seg_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Please re-run segmentation.")

    # 限制轮廓粗细范围
    contour_thickness = max(1, min(5, contour_thickness))

    # 获取切片
    shape = orig_data.shape
    if axis == "axial":
        slice_idx = min(max(0, slice_idx), shape[2] - 1)
        orig_slice = orig_data[:, :, slice_idx].T
        seg_slice = seg_data[:, :, slice_idx].T
    elif axis == "sagittal":
        slice_idx = min(max(0, slice_idx), shape[0] - 1)
        orig_slice = orig_data[slice_idx, :, :].T
        seg_slice = seg_data[slice_idx, :, :].T
    elif axis == "coronal":
        slice_idx = min(max(0, slice_idx), shape[1] - 1)
        orig_slice = orig_data[:, slice_idx, :].T
        seg_slice = seg_data[:, slice_idx, :].T
    else:
        raise HTTPException(status_code=400, detail="Invalid axis")

    # 应用窗宽窗位（兼容原有功能：使用 uint8）
    windowed = apply_window(orig_slice, window_center, window_width, output_uint8=True)

    # 可选：轻微 Gamma 校正增强 CT 对比度（仅在需要时启用）
    # gamma = 0.95  # 轻微增强
    # windowed = np.power(windowed / 255.0, gamma) * 255.0

    # 创建 RGB 图像（使用 float 以便混合计算）
    rgb_image = np.stack([windowed, windowed, windowed], axis=-1).astype(np.float32)

    # 叠加器官分割
    if overlay_organs:
        organs_list = [o.strip() for o in overlay_organs.split(",") if o.strip()]
        task = session.get("task", "total")
        class_map_task = class_map.get(task, {})
        name_to_idx = {v: k for k, v in class_map_task.items()}

        for organ in organs_list:
            if organ in name_to_idx:
                organ_idx = name_to_idx[organ]
                mask = seg_slice == organ_idx
                if np.any(mask):
                    color = np.array(ORGAN_COLORS.get(organ, [255, 255, 0]), dtype=np.float32)

                    if overlay_mode == "contour":
                        # 兼容原有功能：使用形态学轮廓（快速）
                        contour = extract_contour(mask, contour_thickness, use_precise=False)
                        for c in range(3):
                            rgb_image[:, :, c] = np.where(
                                contour,
                                color[c],  # 轮廓线完全不透明
                                rgb_image[:, :, c]
                            )
                    elif overlay_mode == "fill_contour":
                        # 填充 + 轮廓线模式
                        # 兼容原有权重
                        fill_opacity = overlay_opacity * 0.4
                        for c in range(3):
                            rgb_image[:, :, c] = np.where(
                                mask,
                                (1 - fill_opacity) * rgb_image[:, :, c] + fill_opacity * color[c],
                                rgb_image[:, :, c]
                            )
                        # 再画轮廓（高对比度）
                        contour = extract_contour(mask, contour_thickness, use_precise=False)
                        for c in range(3):
                            rgb_image[:, :, c] = np.where(
                                contour,
                                color[c],  # 轮廓线完全不透明
                                rgb_image[:, :, c]
                            )
                    else:
                        # 默认填充模式（兼容原有权重）
                        for c in range(3):
                            rgb_image[:, :, c] = np.where(
                                mask,
                                (1 - overlay_opacity) * rgb_image[:, :, c] + overlay_opacity * color[c],
                                rgb_image[:, :, c]
                            )

    # 转换为 PNG
    rgb_image = np.flipud(rgb_image).astype(np.uint8)

    # ✅ 保留 PNG 优化（压缩）
    img = Image.fromarray(rgb_image)

    buffer = io.BytesIO()
    # ✅ 优化：使用 PNG 压缩（不影响质量，仅减少文件大小）
    img.save(buffer, format="PNG", compress_level=9, optimize=True)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


@app.get("/api/slice_info/{session_id}")
async def get_slice_info(session_id: str):
    """获取各轴向的切片数量"""
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    shape = session.get("shape", [0, 0, 0])

    return {
        "axial": {"max": shape[2] if len(shape) > 2 else 0, "default": shape[2] // 2 if len(shape) > 2 else 0},
        "sagittal": {"max": shape[0] if len(shape) > 0 else 0, "default": shape[0] // 2 if len(shape) > 0 else 0},
        "coronal": {"max": shape[1] if len(shape) > 1 else 0, "default": shape[1] // 2 if len(shape) > 1 else 0}
    }


@app.get("/api/organ_labels/{session_id}")
async def get_organ_labels(
    session_id: str,
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="视图方向: axial, sagittal, coronal"),
    slice_idx: int = Query(0, description="切片索引"),
    organs: str = Query("", description="器官列表，逗号分隔")
):
    """获取指定切片上器官的中心位置，用于显示标签"""
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载（从内存或磁盘）
    _, seg_data = _ensure_session_data_loaded(session)
    if seg_data is None:
        raise HTTPException(status_code=400, detail="Segmentation data not loaded")

    # 获取切片
    shape = seg_data.shape
    if axis == "axial":
        slice_idx = min(max(0, slice_idx), shape[2] - 1)
        seg_slice = seg_data[:, :, slice_idx].T
    elif axis == "sagittal":
        slice_idx = min(max(0, slice_idx), shape[0] - 1)
        seg_slice = seg_data[slice_idx, :, :].T
    elif axis == "coronal":
        slice_idx = min(max(0, slice_idx), shape[1] - 1)
        seg_slice = seg_data[:, slice_idx, :].T
    else:
        raise HTTPException(status_code=400, detail="Invalid axis")

    # 翻转以匹配图像显示
    seg_slice = np.flipud(seg_slice)
    slice_height, slice_width = seg_slice.shape

    # 解析器官列表
    organs_list = [o.strip() for o in organs.split(",") if o.strip()]
    if not organs_list:
        return {"labels": []}

    task = session.get("task", "total")
    class_map_task = class_map.get(task, {})
    name_to_idx = {v: k for k, v in class_map_task.items()}

    labels = []
    for organ in organs_list:
        if organ in name_to_idx:
            organ_idx = name_to_idx[organ]
            mask = seg_slice == organ_idx
            if np.any(mask):
                # 计算质心
                coords = np.where(mask)
                center_y = float(np.mean(coords[0]))
                center_x = float(np.mean(coords[1]))
                # 计算面积占比
                area_ratio = float(np.sum(mask)) / (slice_height * slice_width)
                # 获取颜色
                color = ORGAN_COLORS.get(organ, [255, 255, 0])

                labels.append({
                    "organ": organ,
                    "x": center_x / slice_width,  # 归一化到 0-1
                    "y": center_y / slice_height,
                    "area_ratio": area_ratio,
                    "color": color
                })

    return {"labels": labels, "slice_size": [slice_width, slice_height]}



@app.get("/api/export/labelstudio/{session_id}")
async def export_label_studio(session_id: str):
    """导出为 Label Studio 兼容的 JSON 格式"""
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    task = session.get("task", "total")
    organs = session.get("organs", [])
    shape = session.get("shape", [0, 0, 0])

    # 生成 Label Studio 任务格式
    label_studio_task = {
        "data": {
            "image": f"/api/slice/{session_id}?axis=axial&slice_idx={shape[2]//2}",
            "session_id": session_id,
            "original_filename": session.get("filename", "unknown")
        },
        "predictions": [{
            "model_version": f"TotalSegmentator-{task}",
            "result": []
        }]
    }

    # 为每个器官创建标签
    for organ in organs:
        label_studio_task["predictions"][0]["result"].append({
            "type": "labels",
            "value": {
                "labels": [organ]
            },
            "from_name": "label",
            "to_name": "image"
        })

    return label_studio_task


@app.get("/api/export/nifti/{session_id}")
async def export_nifti(session_id: str):
    """下载分割结果 NIfTI 文件"""
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    output_path = session.get("output_path")

    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Segmentation file not found")

    return FileResponse(
        path=output_path,
        media_type="application/gzip",
        filename=f"segmentation_{session_id}.nii.gz",
    )


@app.get("/api/volume/ct/{session_id}")
async def get_ct_volume(session_id: str):
    """获取原始 CT 数据用于 3D 渲染（NiiVue 兼容格式）"""
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 调试日志
    print(f"[CT Volume] session_id={session_id}")
    print(f"  converted_nifti: {session.get('converted_nifti')}")
    print(f"  processed_input: {session.get('processed_input')}")
    print(f"  input_path: {session.get('input_path')}")

    # 优先使用转换后的 NIfTI 文件，其次使用处理后的输入，最后使用原始输入
    nifti_path = None

    # 1. 优先检查 converted_nifti（2D 图像或需要转换的格式）
    converted_nifti = session.get("converted_nifti")
    if converted_nifti and Path(converted_nifti).exists():
        nifti_path = converted_nifti
        print(f"  -> using converted_nifti: {nifti_path}")
    else:
        # 2. 检查 processed_input（ZIP 解压后选择的 DICOM 转换结果）
        processed_input = session.get("processed_input")
        if processed_input and Path(processed_input).exists() and (
            processed_input.endswith('.nii') or processed_input.endswith('.nii.gz')
        ):
            nifti_path = processed_input
            print(f"  -> using processed_input: {nifti_path}")
        else:
            # 3. 检查原始输入是否为 NIfTI 格式
            input_path = session.get("input_path")
            if input_path and Path(input_path).exists():
                if input_path.endswith('.nii') or input_path.endswith('.nii.gz'):
                    nifti_path = input_path
                    print(f"  -> using input_path: {nifti_path}")

    if not nifti_path or not Path(nifti_path).exists():
        print("  -> ERROR: No valid NIfTI path found")
        raise HTTPException(status_code=404, detail="CT NIfTI data not found")

    # NiiVue 需要正确的 Content-Type 和 CORS 头
    return FileResponse(
        path=nifti_path,
        media_type="application/octet-stream",
        filename=f"ct_{session_id}.nii.gz",
        headers={
            "Content-Disposition": f"inline; filename=ct_{session_id}.nii.gz",
            "Access-Control-Expose-Headers": "Content-Length, Content-Range",
        },
    )


@app.get("/api/volume/seg/{session_id}")
async def get_seg_volume(session_id: str):
    """获取分割结果用于 3D 渲染（NiiVue 兼容格式）"""
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    output_path = session.get("output_path")

    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Segmentation data not found")

    return FileResponse(
        path=output_path,
        media_type="application/octet-stream",
        filename=f"seg_{session_id}.nii.gz",
        headers={
            "Content-Disposition": f"inline; filename=seg_{session_id}.nii.gz",
            "Access-Control-Expose-Headers": "Content-Length, Content-Range",
        },
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话及其数据"""
    exists = await _session_exists(session_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Session not found")

    _release_runtime_arrays(SESSIONS.get(session_id, {}))
    _release_torch_memory()
    gc.collect()

    session_dir = DATA_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    # 使用辅助函数删除（支持数据库和文件两种模式）
    await _delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# 预设窗宽窗位
WINDOW_PRESETS = {
    "soft_tissue": {"center": 40, "width": 400, "name": "软组织窗"},
    "lung": {"center": -600, "width": 1500, "name": "肺窗"},
    "bone": {"center": 400, "width": 1800, "name": "骨窗"},
    "brain": {"center": 40, "width": 80, "name": "脑窗"},
    "liver": {"center": 60, "width": 150, "name": "肝窗"},
    "mediastinum": {"center": 50, "width": 350, "name": "纵隔窗"}
}


@app.get("/api/window_presets")
async def get_window_presets():
    """获取窗宽窗位预设"""
    return WINDOW_PRESETS


@app.get("/api/colormap/{session_id}")
async def get_colormap(session_id: str):
    """
    获取器官颜色映射表，用于 NiiVue 3D 渲染
    返回格式兼容 NiiVue 的 setColormapLabel
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    task = session.get("task", "total")
    class_map_task = class_map.get(task, {})

    # 构建 NiiVue 兼容的 colormap
    # 格式: { R: [], G: [], B: [], A: [], I: [], labels: [] }
    max_idx = max(class_map_task.keys()) if class_map_task else 0

    # 初始化数组，索引 0 为背景（透明）
    r_values = [0]  # 背景
    g_values = [0]
    b_values = [0]
    a_values = [0]  # 背景透明
    i_values = [0]  # 索引值
    labels = ["background"]

    # 按索引顺序添加器官颜色
    for idx in range(1, max_idx + 1):
        organ_name = class_map_task.get(idx, f"unknown_{idx}")
        color = ORGAN_COLORS.get(organ_name, [128, 128, 128])  # 默认灰色

        r_values.append(color[0])
        g_values.append(color[1])
        b_values.append(color[2])
        a_values.append(255)  # 不透明
        i_values.append(idx)
        labels.append(organ_name)

    return {
        "R": r_values,
        "G": g_values,
        "B": b_values,
        "A": a_values,
        "I": i_values,
        "labels": labels
    }


# ==================== LLaMA-Factory 训练数据导出 ====================

import random
from typing import List, Dict, Any

# 问答模板 - 用于数据增强
QA_TEMPLATES = {
    # 器官识别类问题模板
    "organ_identification": [
        {"q": "这张CT图像中能看到哪些器官？", "type": "list_organs"},
        {"q": "请识别图像中所有可见的解剖结构。", "type": "list_organs"},
        {"q": "图中显示了哪些人体器官和组织？", "type": "list_organs"},
        {"q": "分析这张医学影像，列出所有识别到的器官。", "type": "list_organs"},
    ],
    # 特定器官定位
    "organ_location": [
        {"q": "请描述{organ}在图像中的位置。", "type": "locate_organ"},
        {"q": "{organ}在这张CT切片中位于哪里？", "type": "locate_organ"},
        {"q": "能看到{organ}吗？它在图像的什么位置？", "type": "locate_organ"},
    ],
    # 器官状态描述
    "organ_description": [
        {"q": "请描述{organ}的形态特征。", "type": "describe_organ"},
        {"q": "{organ}在这张图像中表现如何？", "type": "describe_organ"},
        {"q": "分析{organ}的影像学特征。", "type": "describe_organ"},
    ],
    # 区域分析
    "region_analysis": [
        {"q": "分析图像中{region}区域的器官分布。", "type": "analyze_region"},
        {"q": "{region}区域可见哪些结构？", "type": "analyze_region"},
    ],
    # 解剖关系
    "anatomical_relation": [
        {"q": "{organ1}和{organ2}的位置关系是什么？", "type": "relation"},
        {"q": "描述{organ1}与{organ2}之间的解剖关系。", "type": "relation"},
    ],
    # 临床相关
    "clinical_relevance": [
        {"q": "从临床角度分析这张图像显示的主要结构。", "type": "clinical"},
        {"q": "这张CT切片对诊断有什么参考价值？", "type": "clinical"},
    ],
    # 诊断分析类问题模板（新增）
    "diagnostic_analysis": [
        {"q": "请对这张CT图像进行初步诊断分析。", "type": "diagnosis"},
        {"q": "基于这张图像，有哪些需要关注的影像学表现？", "type": "diagnosis"},
        {"q": "从放射科医生角度，分析这张CT图像的诊断要点。", "type": "diagnosis"},
        {"q": "这张CT切片显示的器官有无明显异常？请分析。", "type": "diagnosis"},
        {"q": "请对图像中可见的器官进行系统性评估。", "type": "diagnosis"},
        {"q": "这张影像的诊断报告应该包含哪些要点？", "type": "diagnosis"},
    ],
    # 综合报告类问题模板（新增）
    "comprehensive_report": [
        {"q": "请为这张CT图像生成一份简要的影像学报告。", "type": "report"},
        {"q": "作为放射科医生，请描述这张图像的主要发现。", "type": "report"},
        {"q": "请用专业术语描述这张CT切片的影像学表现。", "type": "report"},
    ],
}

# 自定义 QA 模板（用户可覆盖默认模板）
CUSTOM_QA_TEMPLATES: Dict[str, List[Dict[str, str]]] = {}

# 诊断分析回答模板
DIAGNOSTIC_ANSWER_TEMPLATES = {
    "normal": [
        "本层面CT图像显示{organs_summary}。各器官形态、大小、密度均在正常范围内，边界清晰，未见明显占位性病变、积液或异常信号。",
        "影像分析：本切面可见{organs_summary}。上述结构显示正常，无明显病理改变。建议结合临床及其他影像学检查综合评估。",
        "放射学评估：{organs_summary}在本层面显示清晰。各解剖结构位置正常，密度均匀，未见异常增强或信号改变。",
    ],
    "report": [
        "影像学报告：本CT切片层面显示{organs_summary}。检查所见：各器官形态规整，大小正常，密度均匀。未见明显异常发现。",
        "CT检查报告：扫描层面可见{organs_summary}。影像表现：各解剖结构清晰可辨，边界光整，实质密度正常，未见局灶性病变。",
    ],
}

# 区域分类
ANATOMICAL_REGIONS = {
    "腹部": ["liver", "spleen", "pancreas", "gallbladder", "stomach", "small_bowel", "duodenum", "colon",
             "kidney_left", "kidney_right", "adrenal_gland_left", "adrenal_gland_right"],
    "胸部": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
             "lung_lower_lobe_right", "heart", "aorta", "trachea", "esophagus"],
    "盆腔": ["urinary_bladder", "prostate", "hip_left", "hip_right", "sacrum"],
    "脊柱": ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
             "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
             "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10",
             "vertebrae_T11", "vertebrae_T12", "vertebrae_C1", "vertebrae_C2", "vertebrae_C3",
             "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7", "spinal_cord"],
}


def get_region_for_organ(organ: str) -> str:
    """获取器官所属的解剖区域"""
    for region, organs in ANATOMICAL_REGIONS.items():
        if organ in organs:
            return region
    return "其他"


def generate_organ_list_answer(organs: List[str], include_details: bool = True) -> str:
    """生成器官列表的回答"""
    if not organs:
        return "在这张图像中未能识别到明显的器官结构。"

    organ_names = [get_organ_cn_name(o) for o in organs]

    if len(organs) <= 3:
        answer = f"在这张CT图像中，可以识别到以下器官：{', '.join(organ_names)}。"
    else:
        # 按区域分组
        regions = {}
        for organ in organs:
            region = get_region_for_organ(organ)
            if region not in regions:
                regions[region] = []
            regions[region].append(get_organ_cn_name(organ))

        parts = []
        for region, names in regions.items():
            if names:
                parts.append(f"{region}区域可见{', '.join(names)}")

        answer = f"在这张CT图像中，共识别到{len(organs)}个解剖结构。" + "；".join(parts) + "。"

    return answer


def generate_location_answer(organ: str, position_info: dict = None) -> str:
    """生成器官定位的回答"""
    organ_cn = get_organ_cn_name(organ)

    # 基于器官生成位置描述
    location_hints = {
        "liver": "位于右上腹部，膈肌下方",
        "spleen": "位于左上腹部，胃的后外侧",
        "heart": "位于胸腔中央偏左，两肺之间",
        "kidney_left": "位于左后腹膜后间隙",
        "kidney_right": "位于右后腹膜后间隙",
        "stomach": "位于左上腹部，肝脏左侧",
        "pancreas": "位于腹膜后，横跨脊柱前方",
    }

    base_location = location_hints.get(organ, "在图像中可见")

    if position_info:
        x, y = position_info.get("x", 0.5), position_info.get("y", 0.5)
        if x < 0.4:
            h_pos = "左侧"
        elif x > 0.6:
            h_pos = "右侧"
        else:
            h_pos = "中央"

        if y < 0.4:
            v_pos = "上部"
        elif y > 0.6:
            v_pos = "下部"
        else:
            v_pos = "中部"

        return f"是的，{organ_cn}在图像中可见，位于图像的{h_pos}{v_pos}区域。解剖上，{organ_cn}{base_location}。"

    return f"{organ_cn}在这张切片中{base_location}，可以清晰识别其轮廓和边界。"


def generate_diagnostic_answer(organs: List[str]) -> str:
    """生成诊断分析回答"""
    if not organs:
        return "本层面未见明显器官结构，建议参考其他层面综合评估。"

    organ_names = [get_organ_cn_name(o) for o in organs]
    if len(organ_names) <= 3:
        organs_summary = "、".join(organ_names)
    else:
        organs_summary = f"{organ_names[0]}、{organ_names[1]}等{len(organ_names)}个解剖结构"

    template = random.choice(DIAGNOSTIC_ANSWER_TEMPLATES["normal"])
    return template.format(organs_summary=organs_summary)


def generate_report_answer(organs: List[str]) -> str:
    """生成影像报告回答"""
    if not organs:
        return "本层面CT扫描未见明显器官结构，建议结合其他层面综合评估。"

    organ_names = [get_organ_cn_name(o) for o in organs]
    if len(organ_names) <= 4:
        organs_summary = "、".join(organ_names)
    else:
        organs_summary = f"{organ_names[0]}、{organ_names[1]}、{organ_names[2]}等{len(organ_names)}个解剖结构"

    template = random.choice(DIAGNOSTIC_ANSWER_TEMPLATES["report"])
    return template.format(organs_summary=organs_summary)


def generate_training_qa_pairs(
    organs: List[str],
    slice_info: dict,
    num_samples: int = 5,
    include_augmentation: bool = True
) -> List[Dict[str, str]]:
    """生成训练问答对

    根据模板选择状态和优先级生成问答对：
    - 优先使用标记为"重点"的模板类别
    - 其次使用"已启用"的模板类别
    - 如果都没有选择，使用所有模板

    生成多种类型的问答对：
    1. 器官识别问答
    2. 器官定位问答
    3. 器官描述问答
    4. 诊断分析问答
    5. 综合报告问答
    """
    qa_pairs = []

    if not organs:
        return qa_pairs

    # 获取按优先级排序的启用模板
    enabled_templates = get_all_enabled_templates()
    enabled_categories = list(enabled_templates.keys())

    # 辅助函数：安全获取模板（考虑优先级）
    def safe_get_template(category: str) -> Dict[str, str]:
        # 如果该类别在启用列表中，使用它
        if category in enabled_templates and enabled_templates[category]:
            return random.choice(enabled_templates[category])
        # 否则回退到默认
        templates = get_effective_templates(category)
        if not templates:
            templates = QA_TEMPLATES.get(category, [{"q": "请描述这张图像", "type": "default"}])
        return random.choice(templates)

    # 检查类别是否启用
    def is_category_enabled(category: str) -> bool:
        state = get_template_selection_state(category)
        return state.get("enabled", True)

    # 1. 基础器官列表问答（如果启用）
    if is_category_enabled("organ_identification"):
        q_template = safe_get_template("organ_identification")
        qa_pairs.append({
            "question": q_template["q"],
            "answer": generate_organ_list_answer(organs),
            "type": "organ_identification"
        })

    # 2. 特定器官定位问答（如果启用）
    if is_category_enabled("organ_location") and len(organs) >= 1:
        sample_organs = random.sample(organs, min(2, len(organs)))
        for organ in sample_organs:
            if len(qa_pairs) >= num_samples:
                break
            q_template = safe_get_template("organ_location")
            qa_pairs.append({
                "question": q_template["q"].format(organ=get_organ_cn_name(organ)),
                "answer": generate_location_answer(organ),
                "type": "organ_location"
            })

    # 3. 诊断分析问答（如果启用）
    if is_category_enabled("diagnostic_analysis") and len(qa_pairs) < num_samples:
        q_template = safe_get_template("diagnostic_analysis")
        qa_pairs.append({
            "question": q_template["q"],
            "answer": generate_diagnostic_answer(organs),
            "type": "diagnostic_analysis"
        })

    # 4. 综合报告问答（如果启用）
    if is_category_enabled("comprehensive_report") and len(qa_pairs) < num_samples:
        q_template = safe_get_template("comprehensive_report")
        qa_pairs.append({
            "question": q_template["q"],
            "answer": generate_report_answer(organs),
            "type": "comprehensive_report"
        })

    # 5. 数据增强 - 器官描述（如果启用）
    if include_augmentation and is_category_enabled("organ_description") and len(qa_pairs) < num_samples:
        for organ in random.sample(organs, min(2, len(organs))):
            if len(qa_pairs) >= num_samples:
                break
            q_template = safe_get_template("organ_description")
            organ_cn = get_organ_cn_name(organ)
            qa_pairs.append({
                "question": q_template["q"].format(organ=organ_cn),
                "answer": f"{organ_cn}在图像中显示正常形态，边界清晰，密度均匀，未见明显异常信号。",
                "type": "organ_description"
            })

    # 如果没有生成任何问答（所有类别都禁用），至少生成一个基础问答
    if not qa_pairs:
        q_template = get_effective_templates("organ_identification")
        if q_template:
            qa_pairs.append({
                "question": q_template[0]["q"],
                "answer": generate_organ_list_answer(organs),
                "type": "organ_identification"
            })

    return qa_pairs[:num_samples]


@app.get("/api/export/llama-factory/{session_id}")
async def export_llama_factory(
    session_id: str,
    format: str = Query("sharegpt", description="数据格式: sharegpt, alpaca"),
    num_samples: int = Query(5, description="每个切片生成的问答对数量"),
    slice_step: int = Query(10, ge=1, description="切片采样步长"),
    include_augmentation: bool = Query(True, description="是否包含数据增强"),
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="切片轴向: axial, sagittal, coronal"),
    use_llm: bool = Query(False, description="是否使用 LLM 生成问答（需配置 LLM）")
):
    """
    导出 LLaMA-Factory 兼容的多模态训练数据

    支持两种格式:
    - sharegpt: 对话格式，适合微调
    - alpaca: 指令格式，适合监督学习

    问答生成模式:
    - use_llm=false: 使用模板生成（快速）
    - use_llm=true: 使用配置的 LLM 生成（需先配置 /api/config/llm）
    """
    import time as time_module
    export_start_time = time_module.time()

    print(f"\n{'#'*70}")
    print("[导出] 🚀 开始导出 LLaMA-Factory 训练数据")
    print(f"[导出] 📋 Session: {session_id}")
    print(f"[导出] 🔧 参数: format={format}, samples={num_samples}, step={slice_step}, axis={axis}")
    print(f"[导出] 🤖 LLM模式: {'启用' if use_llm else '禁用'}")
    print(f"{'#'*70}\n")

    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载（从内存或磁盘）
    orig_data, seg_data = _ensure_session_data_loaded(session)

    if orig_data is None or seg_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Please re-run segmentation or check session files.")

    task = session.get("task", "total")
    class_map_task = class_map.get(task, {})
    shape = orig_data.shape
    detected_organs = session.get("organs", [])

    print(f"[导出] 📊 数据维度: {shape}")
    print(f"[导出] 🫀 检测器官数: {len(detected_organs)}")

    training_data = []

    # 根据轴向确定切片范围
    if axis == "axial":
        max_slice = shape[2]
    elif axis == "sagittal":
        max_slice = shape[0]
    else:  # coronal
        max_slice = shape[1]

    # 计算预计处理的切片数
    slice_indices = list(range(0, max_slice, slice_step))
    total_slices_to_process = len(slice_indices)
    processed_count = 0

    print(f"[导出] 📊 切片范围: 0-{max_slice}, 步长={slice_step}, 预计处理 {total_slices_to_process} 个切片")

    # 采样切片
    for slice_idx in slice_indices:
        processed_count += 1
        print(f"\n[导出] 🔄 处理切片 {processed_count}/{total_slices_to_process} (index={slice_idx})")

        # 获取该切片的原始数据和分割数据（两者必须保持相同的转置操作）
        if axis == "axial":
            orig_slice = orig_data[:, :, slice_idx].T
            seg_slice = seg_data[:, :, slice_idx].T
        elif axis == "sagittal":
            orig_slice = orig_data[slice_idx, :, :].T
            seg_slice = seg_data[slice_idx, :, :].T
        else:
            orig_slice = orig_data[:, slice_idx, :].T
            seg_slice = seg_data[:, slice_idx, :].T

        # 找出该切片中存在的器官
        unique_labels = np.unique(seg_slice)
        slice_organs = []
        for label in unique_labels:
            if label > 0 and label in class_map_task:
                organ_name = class_map_task[label]
                if organ_name in detected_organs:
                    slice_organs.append(organ_name)

        if not slice_organs:
            continue

        # 图像路径（相对路径，用于训练时配置）
        image_path = f"images/{session_id}_{axis}_{slice_idx}.png"

        # 生成问答对
        slice_info = {
            "axis": axis,
            "slice_idx": slice_idx,
            "total_slices": max_slice
        }

        if use_llm:
            # 生成并排对比图用于多模态 LLM 分析
            comparison_image_base64 = generate_comparison_image(
                orig_slice=orig_slice,
                seg_slice=seg_slice,
                organs=slice_organs,
                class_map_task=class_map_task,
                window_center=40,  # 使用默认窗位
                window_width=400,  # 使用默认窗宽
                overlay_opacity=0.5
            )

            # 使用 LLM 生成问答对（传递图像和模板选择状态）
            enabled_categories, priority_categories = get_template_selection_for_llm()
            qa_pairs_list = await generate_qa_pairs(
                organs=slice_organs,
                slice_info=slice_info,
                num_pairs=num_samples,
                use_llm=True,
                image_base64=comparison_image_base64,
                enabled_categories=enabled_categories,
                priority_categories=priority_categories
            )
        else:
            # 使用模板生成问答对
            qa_pairs_list = generate_training_qa_pairs(
                slice_organs,
                slice_info,
                num_samples=num_samples,
                include_augmentation=include_augmentation
            )

        for qa in qa_pairs_list:
            if format == "sharegpt":
                # ShareGPT 格式
                training_data.append({
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{qa['question']}"},
                        {"from": "gpt", "value": qa['answer']}
                    ],
                    "images": [image_path],
                    "source": qa.get("source", "template")
                })
            else:
                # Alpaca 格式
                training_data.append({
                    "instruction": qa['question'],
                    "input": "",
                    "output": qa['answer'],
                    "images": [image_path],
                    "source": qa.get("source", "template")
                })

    # 返回训练数据和元信息
    export_elapsed = time_module.time() - export_start_time
    config_obj = get_config()

    print(f"\n{'#'*70}")
    print("[导出] ✅ 导出完成!")
    print(f"[导出] 📊 总样本数: {len(training_data)}")
    print(f"[导出] ⏱️ 总耗时: {export_elapsed:.2f}s")
    print(f"[导出] 📈 平均: {export_elapsed/max(len(training_data),1):.2f}s/样本")
    print(f"{'#'*70}\n")

    return {
        "format": format,
        "total_samples": len(training_data),
        "session_id": session_id,
        "task": task,
        "detected_organs": detected_organs,
        "config": {
            "slice_step": slice_step,
            "num_samples_per_slice": num_samples,
            "axis": axis,
            "include_augmentation": include_augmentation,
            "use_llm": use_llm,
            "llm_enabled": config_obj.llm.enabled,
            "llm_model": config_obj.llm.model if use_llm else None
        },
        "data": training_data
    }


@app.get("/api/export/llama-factory/{session_id}/preview")
async def preview_llama_factory(
    session_id: str,
    format: str = Query("sharegpt", description="数据格式: sharegpt, alpaca"),
    num_samples: int = Query(5, description="每个切片生成的问答对数量"),
    slice_step: int = Query(10, ge=1, description="切片采样步长"),
    include_augmentation: bool = Query(True, description="是否包含数据增强"),
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="切片轴向: axial, sagittal, coronal"),
    use_llm: bool = Query(False, description="是否使用 LLM 生成问答"),
    window_center: int = Query(40, description="窗位"),
    window_width: int = Query(400, gt=0, description="窗宽"),
    max_preview: int = Query(50, description="最大预览数量")
):
    """
    生成训练数据预览，包含切片图像的 Base64 数据供前端展示和审核
    """
    import base64
    import time as time_module

    preview_start_time = time_module.time()

    print(f"\n{'#'*70}")
    print("[预览] 🔍 开始生成训练数据预览")
    print(f"[预览] 📋 Session: {session_id}")
    print(f"[预览] 🔧 参数: samples={num_samples}, step={slice_step}, axis={axis}, max={max_preview}")
    print(f"[预览] 🤖 LLM模式: {'启用' if use_llm else '禁用'}")
    print(f"{'#'*70}\n")

    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载
    orig_data, seg_data = _ensure_session_data_loaded(session)

    if orig_data is None or seg_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")

    task = session.get("task", "total")
    class_map_task = class_map.get(task, {})
    shape = orig_data.shape
    detected_organs = session.get("organs", [])

    print(f"[预览] 📊 数据维度: {shape}")
    print(f"[预览] 🫀 检测器官数: {len(detected_organs)}")

    preview_data = []

    # 根据轴向确定切片范围
    if axis == "axial":
        max_slice = shape[2]
    elif axis == "sagittal":
        max_slice = shape[0]
    else:
        max_slice = shape[1]

    print(f"[预览] 📊 切片范围: 0-{max_slice}, 步长={slice_step}")

    # 采样切片
    preview_count = 0
    for slice_idx in range(0, max_slice, slice_step):
        if preview_count >= max_preview:
            print(f"[预览] ⏹️ 达到最大预览数 {max_preview}，停止")
            break

        print(f"\n[预览] 🔄 处理切片 {preview_count+1}/{max_preview} (index={slice_idx})")

        # 获取该切片的原始图像和分割数据（两者必须保持相同的转置操作）
        if axis == "axial":
            orig_slice = orig_data[:, :, slice_idx].T
            seg_slice = seg_data[:, :, slice_idx].T
        elif axis == "sagittal":
            orig_slice = orig_data[slice_idx, :, :].T
            seg_slice = seg_data[slice_idx, :, :].T
        else:
            orig_slice = orig_data[:, slice_idx, :].T
            seg_slice = seg_data[:, slice_idx, :].T

        # 找出该切片中存在的器官
        unique_labels = np.unique(seg_slice)
        slice_organs = []
        for label in unique_labels:
            if label > 0 and label in class_map_task:
                organ_name = class_map_task[label]
                if organ_name in detected_organs:
                    slice_organs.append(organ_name)

        if not slice_organs:
            continue

        # 生成原始CT切片图像的 Base64
        windowed = apply_window(orig_slice, window_center, window_width)
        windowed_flipped = np.flipud(windowed)  # 翻转以保持一致
        img = Image.fromarray(windowed_flipped)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # 生成分割叠加图（用于前端预览显示）- 使用主界面相同的效果
        seg_overlay_base64 = generate_overlay_image(
            orig_slice=orig_slice,
            seg_slice=seg_slice,
            organs=slice_organs,
            class_map_task=class_map_task,
            window_center=window_center,
            window_width=window_width,
            overlay_opacity=1.0,
            overlay_mode="fill_contour",
            contour_thickness=1,
            draw_labels=True
        )

        # 图像路径（用于最终导出）
        image_path = f"images/{session_id}_{axis}_{slice_idx}.png"

        # 生成问答对
        slice_info = {
            "axis": axis,
            "slice_idx": slice_idx,
            "total_slices": max_slice
        }

        if use_llm:
            # 生成并排对比图用于 LLM 分析
            comparison_image_base64 = generate_comparison_image(
                orig_slice=orig_slice,
                seg_slice=seg_slice,
                organs=slice_organs,
                class_map_task=class_map_task,
                window_center=window_center,
                window_width=window_width,
                overlay_opacity=0.5
            )

            # 使用 LLM 生成问答对（传递图像和模板选择状态）
            try:
                enabled_categories, priority_categories = get_template_selection_for_llm()
                qa_pairs = await generate_qa_pairs(
                    organs=slice_organs,
                    slice_info=slice_info,
                    num_pairs=num_samples,
                    use_llm=True,
                    image_base64=comparison_image_base64,
                    enabled_categories=enabled_categories,
                    priority_categories=priority_categories
                )
            except Exception as e:
                print(f"LLM 生成问答失败，回退到模板: {e}")
                qa_pairs = generate_training_qa_pairs(
                    slice_organs, slice_info, num_samples, include_augmentation
                )
        else:
            # 使用模板生成问答对
            qa_pairs = generate_training_qa_pairs(
                slice_organs, slice_info, num_samples, include_augmentation
            )

        # 转换为预览格式
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")

            if format == "sharegpt":
                item = {
                    "id": f"{session_id}_{axis}_{slice_idx}_{preview_count}",
                    "slice_idx": slice_idx,
                    "axis": axis,
                    "image_path": image_path,
                    "image_base64": img_base64,
                    "seg_overlay_base64": seg_overlay_base64,  # 新增分割叠加图
                    "organs": slice_organs,
                    "question": question,
                    "answer": answer,
                    "source": qa.get("source", "template"),
                    "selected": True
                }
            else:
                # Alpaca 格式
                item = {
                    "id": f"{session_id}_{axis}_{slice_idx}_{preview_count}",
                    "slice_idx": slice_idx,
                    "axis": axis,
                    "image_path": image_path,
                    "image_base64": img_base64,
                    "seg_overlay_base64": seg_overlay_base64,  # 新增分割叠加图
                    "organs": slice_organs,
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "source": qa.get("source", "template"),
                    "selected": True
                }

            preview_data.append(item)
            preview_count += 1

            if preview_count >= max_preview:
                break

    preview_elapsed = time_module.time() - preview_start_time

    print(f"\n{'#'*70}")
    print("[预览] ✅ 预览生成完成!")
    print(f"[预览] 📊 预览项数: {len(preview_data)}")
    print(f"[预览] ⏱️ 总耗时: {preview_elapsed:.2f}s")
    if len(preview_data) > 0:
        print(f"[预览] 📈 平均: {preview_elapsed/len(preview_data):.2f}s/项")
    print(f"{'#'*70}\n")

    result = {
        "session_id": session_id,
        "format": format,
        "total_available": len(preview_data),
        "max_preview": max_preview,
        "config": {
            "slice_step": slice_step,
            "num_samples_per_slice": num_samples,
            "axis": axis,
            "include_augmentation": include_augmentation,
            "use_llm": use_llm
        },
        "preview": preview_data
    }

    # 持久化预览数据到文件
    preview_cache_path = DATA_DIR / session_id / "preview_cache.json"
    try:
        # 保存预览数据（不包含 base64 图像以节省空间，图像可以重新生成）
        cache_data = {
            "session_id": session_id,
            "format": format,
            "config": {
                "slice_step": slice_step,
                "num_samples_per_slice": num_samples,
                "axis": axis,
                "include_augmentation": include_augmentation,
                "use_llm": use_llm
            },
            "preview": [
                {
                    "id": item["id"],
                    "slice_idx": item["slice_idx"],
                    "axis": item["axis"],
                    "image_path": item["image_path"],
                    "organs": item["organs"],
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "source": item.get("source", "template"),
                    "selected": item.get("selected", True),
                }
                for item in preview_data
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        preview_cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[预览] ✓ 预览数据已缓存到 {preview_cache_path}")
    except Exception as e:
        print(f"[预览] ⚠ 缓存预览数据失败: {e}")

    # 更新会话的活动状态
    session["activity_status"] = "idle"
    session["last_preview_at"] = datetime.now().isoformat()
    await _save_session(session_id, session)

    return result


# ============ 预览数据持久化 API ============

import base64  # 用于图像编码

@app.get("/api/preview/{session_id}")
async def get_cached_preview(session_id: str):
    """
    获取缓存的预览数据（如果存在）
    用于恢复之前的预览状态，避免重新生成
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    preview_cache_path = DATA_DIR / session_id / "preview_cache.json"

    if not preview_cache_path.exists():
        return {"exists": False, "preview": None}

    try:
        cache_data = json.loads(preview_cache_path.read_text(encoding="utf-8"))

        # 重新生成 base64 图像（从磁盘读取或重新渲染）
        orig_data, seg_data = _ensure_session_data_loaded(session)

        if orig_data is None or seg_data is None:
            return {"exists": True, "preview": cache_data, "images_available": False}

        # 获取任务类型
        task = session.get("task", "total")
        class_map_task = class_map.get(task, {})
        detected_organs = session.get("organs", [])
        window_center = 40
        window_width = 400

        # 为每个预览项重新生成图像
        for item in cache_data.get("preview", []):
            try:
                slice_idx = item["slice_idx"]
                axis = item["axis"]

                # 获取切片数据（两者必须保持相同的转置操作）
                if axis == "axial":
                    orig_slice = orig_data[:, :, slice_idx].T
                    seg_slice = seg_data[:, :, slice_idx].T
                elif axis == "sagittal":
                    orig_slice = orig_data[slice_idx, :, :].T
                    seg_slice = seg_data[slice_idx, :, :].T
                else:  # coronal
                    orig_slice = orig_data[:, slice_idx, :].T
                    seg_slice = seg_data[:, slice_idx, :].T

                # 使用标准窗宽窗位处理函数
                windowed = apply_window(orig_slice, window_center, window_width)
                windowed_flipped = np.flipud(windowed)
                img = Image.fromarray(windowed_flipped)
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                item["image_base64"] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                # 生成分割叠加图 - 使用主界面相同的效果
                slice_organs = item.get("organs", [])
                seg_overlay_base64 = generate_overlay_image(
                    orig_slice=orig_slice,
                    seg_slice=seg_slice,
                    organs=slice_organs,
                    class_map_task=class_map_task,
                    window_center=window_center,
                    window_width=window_width,
                    overlay_opacity=1.0,
                    overlay_mode="fill_contour",
                    contour_thickness=1,
                    draw_labels=True
                )
                if seg_overlay_base64:
                    item["seg_overlay_base64"] = seg_overlay_base64
            except Exception as img_e:
                print(f"[预览] ⚠ 重新生成切片 {item.get('slice_idx')} 图像失败: {img_e}")

        return {
            "exists": True,
            "preview": cache_data,
            "images_available": True
        }
    except Exception as e:
        print(f"[预览] ⚠ 读取缓存失败: {e}")
        return {"exists": False, "preview": None, "error": str(e)}


@app.put("/api/preview/{session_id}")
async def update_cached_preview(session_id: str, request: Request):
    """
    更新缓存的预览数据（用户编辑问答后保存）
    """
    exists = await _session_exists(session_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Session not found")

    preview_cache_path = DATA_DIR / session_id / "preview_cache.json"

    try:
        body = await request.json()
        preview_items = body.get("preview", [])

        # 读取现有缓存
        if preview_cache_path.exists():
            cache_data = json.loads(preview_cache_path.read_text(encoding="utf-8"))
        else:
            cache_data = {
                "session_id": session_id,
                "format": "sharegpt",
                "config": {},
                "preview": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        # 更新预览数据（不保存 base64 图像）
        cache_data["preview"] = [
            {
                "id": item.get("id", ""),
                "slice_idx": item.get("slice_idx", 0),
                "axis": item.get("axis", "axial"),
                "image_path": item.get("image_path", ""),
                "organs": item.get("organs", []),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "source": item.get("source", "template"),
                "selected": item.get("selected", True),
            }
            for item in preview_items
        ]
        cache_data["updated_at"] = datetime.now().isoformat()

        preview_cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"status": "success", "updated_count": len(preview_items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存预览数据失败: {e}")


@app.delete("/api/preview/{session_id}")
async def delete_cached_preview(session_id: str):
    """
    删除缓存的预览数据
    """
    exists = await _session_exists(session_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Session not found")

    preview_cache_path = DATA_DIR / session_id / "preview_cache.json"

    if preview_cache_path.exists():
        preview_cache_path.unlink()
        return {"status": "deleted"}

    return {"status": "not_found"}


# ============ 会话活动状态 API ============

@app.put("/api/session/{session_id}/activity")
async def update_session_activity(session_id: str, request: Request):
    """
    更新会话的活动状态（用于前端通知后端当前正在进行的操作）
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        body = await request.json()
        activity_status = body.get("activity_status", "idle")

        # 有效的活动状态
        valid_statuses = ["idle", "generating_preview", "exporting"]
        if activity_status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid activity status: {activity_status}")

        # 确保内存中有会话引用
        SESSIONS[session_id] = session
        session["activity_status"] = activity_status
        session["activity_updated_at"] = datetime.now().isoformat()

        # 持久化
        await _save_session(session_id, session)

        return {"status": "success", "activity_status": activity_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/activities")
async def get_all_session_activities():
    """
    获取所有会话的活动状态（用于任务列表显示）
    """
    activities = {}

    if _database_mode_enabled():
        # 数据库模式 - 从数据库查询活动状态不为 idle 的会话
        try:
            from api.task_repository import TaskRepository
            active_tasks = await TaskRepository.get_active_sessions()
            for task in active_tasks:
                session_id = task.get("session_id")
                if session_id:
                    activities[session_id] = {
                        "activity_status": task.get("activity_status", "idle"),
                        "updated_at": task.get("activity_updated_at", "")
                    }
        except Exception as e:
            print(f"[活动状态] 从数据库获取活动状态失败: {e}")
            _disable_database_mode("get_active_sessions failed", e)

    if not _database_mode_enabled():
        # 文件模式 - 从内存获取
        for session_id, session in SESSIONS.items():
            activity = session.get("activity_status", "idle")
            if activity != "idle":
                activities[session_id] = {
                    "activity_status": activity,
                    "updated_at": session.get("activity_updated_at", "")
                }

    return {"activities": activities}


@app.get("/api/export/llama-factory-images/{session_id}")
async def export_llama_factory_images(
    session_id: str,
    slice_step: int = Query(10, ge=1, description="切片采样步长"),
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="切片轴向"),
    window_center: int = Query(40, description="窗位"),
    window_width: int = Query(400, gt=0, description="窗宽"),
    output_dir: str = Query("", description="输出目录路径（服务器端）")
):
    """
    导出训练用的切片图像到指定目录
    返回生成的图像列表
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载（从内存或磁盘）
    orig_data, _ = _ensure_session_data_loaded(session)
    if orig_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")

    shape = orig_data.shape

    # 确定输出目录
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = DATA_DIR / "llama_factory_export" / session_id / "images"

    out_path.mkdir(parents=True, exist_ok=True)

    # 根据轴向确定切片范围
    if axis == "axial":
        max_slice = shape[2]
    elif axis == "sagittal":
        max_slice = shape[0]
    else:
        max_slice = shape[1]

    generated_images = []

    for slice_idx in range(0, max_slice, slice_step):
        # 获取切片
        if axis == "axial":
            orig_slice = orig_data[:, :, slice_idx].T
        elif axis == "sagittal":
            orig_slice = orig_data[slice_idx, :, :].T
        else:
            orig_slice = orig_data[:, slice_idx, :].T

        # 应用窗宽窗位
        windowed = apply_window(orig_slice, window_center, window_width)

        # 保存图像
        filename = f"{session_id}_{axis}_{slice_idx}.png"
        filepath = out_path / filename

        img = Image.fromarray(windowed)
        img.save(filepath)

        generated_images.append({
            "filename": filename,
            "path": str(filepath),
            "slice_idx": slice_idx,
            "axis": axis
        })

    return {
        "session_id": session_id,
        "output_dir": str(out_path),
        "total_images": len(generated_images),
        "images": generated_images
    }


# ==================== Label Studio 完整集成 API ====================

class LabelStudioAnnotation(BaseModel):
    """Label Studio 标注结果"""
    task_id: int
    annotations: List[Dict[str, Any]]


# Label Studio 标注配置模板
LABEL_STUDIO_CONFIG_ORGAN_SEGMENTATION = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>

  <!-- 器官分割标注 -->
  <BrushLabels name="organ_mask" toName="image">
    {organ_labels}
  </BrushLabels>

  <!-- 问答标注 -->
  <Header value="医学影像问答标注"/>
  <Text name="question" value="$question"/>
  <TextArea name="answer" toName="question" rows="4"
            placeholder="请输入对该问题的回答..." required="true"/>

  <!-- 质量评估 -->
  <Choices name="quality" toName="image" choice="single">
    <Choice value="高质量" selected="true"/>
    <Choice value="中等质量"/>
    <Choice value="低质量"/>
    <Choice value="需要修正"/>
  </Choices>
</View>
"""

LABEL_STUDIO_CONFIG_VQA = """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>

  <Header value="医学影像问答"/>

  <!-- 预填充的问题 -->
  <View style="margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
    <Header value="问题："/>
    <Text name="question_text" value="$question"/>
  </View>

  <!-- AI 预标注答案 -->
  <View style="margin-top: 10px; padding: 10px; background: #e8f5e9; border-radius: 5px;">
    <Header value="AI 预标注答案："/>
    <Text name="ai_answer" value="$ai_answer"/>
  </View>

  <!-- 人工修正/确认的答案 -->
  <View style="margin-top: 10px;">
    <Header value="人工审核/修正答案："/>
    <TextArea name="human_answer" toName="question_text" rows="4"
              placeholder="请审核并修正AI答案，或确认无误后复制粘贴..."
              required="true"/>
  </View>

  <!-- 标注决策 -->
  <Choices name="decision" toName="image" choice="single" required="true">
    <Choice value="确认正确"/>
    <Choice value="需要修正"/>
    <Choice value="答案错误"/>
    <Choice value="问题不适合"/>
  </Choices>

  <!-- 备注 -->
  <TextArea name="notes" toName="image" rows="2"
            placeholder="可选：添加备注说明..."/>
</View>
"""


@app.get("/api/labelstudio/config/{config_type}")
async def get_label_studio_config(
    config_type: str = "vqa",
    session_id: Optional[str] = None
):
    """
    获取 Label Studio 标注配置模板

    Args:
        config_type: 配置类型 (vqa: 问答标注, segmentation: 分割标注)
        session_id: 可选的 session_id，用于生成包含特定器官的配置
    """
    if config_type == "vqa":
        return {
            "config": LABEL_STUDIO_CONFIG_VQA,
            "description": "医学影像问答标注配置，支持审核和修正 AI 预标注答案"
        }
    elif config_type == "segmentation":
        # 如果提供了 session_id，生成包含实际器官的配置
        organ_labels = ""
        if session_id and session_id in SESSIONS:
            organs = SESSIONS[session_id].get("organs", [])
            for organ in organs[:30]:  # 限制最多30个器官
                organ_cn = get_organ_cn_name(organ)
                color = ORGAN_COLORS.get(organ, [128, 128, 128])
                rgba = f"rgba({color[0]}, {color[1]}, {color[2]}, 0.7)"
                organ_labels += f'    <Label value="{organ_cn}" background="{rgba}"/>\n'
        else:
            # 默认器官列表
            default_organs = ["liver", "spleen", "kidney_left", "kidney_right", "heart", "lung_left", "lung_right"]
            for organ in default_organs:
                organ_cn = get_organ_cn_name(organ)
                color = ORGAN_COLORS.get(organ, [128, 128, 128])
                rgba = f"rgba({color[0]}, {color[1]}, {color[2]}, 0.7)"
                organ_labels += f'    <Label value="{organ_cn}" background="{rgba}"/>\n'

        config = LABEL_STUDIO_CONFIG_ORGAN_SEGMENTATION.format(organ_labels=organ_labels)
        return {
            "config": config,
            "description": "医学影像器官分割标注配置"
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown config type: {config_type}")


@app.get("/api/labelstudio/tasks/{session_id}")
async def export_label_studio_tasks(
    session_id: str,
    slice_step: int = Query(10, ge=1, description="切片采样步长"),
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="切片轴向"),
    num_qa_per_slice: int = Query(3, description="每个切片生成的问答对数量"),
    base_url: str = Query("", description="API 基础 URL，用于生成图像链接"),
    window_center: int = Query(40, description="窗位"),
    window_width: int = Query(400, gt=0, description="窗宽")
):
    """
    导出完整的 Label Studio 任务列表

    生成可直接导入 Label Studio 的 JSON 任务，包含:
    - 图像链接
    - AI 预标注的问答对
    - 器官分割预标注
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载（从内存或磁盘）
    orig_data, seg_data = _ensure_session_data_loaded(session)

    if orig_data is None or seg_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")

    task = session.get("task", "total")
    class_map_task = class_map.get(task, {})
    shape = orig_data.shape
    detected_organs = session.get("organs", [])

    # 确定基础 URL
    if not base_url:
        base_url = "http://localhost:28000"

    tasks = []

    # 根据轴向确定切片范围
    if axis == "axial":
        max_slice = shape[2]
    elif axis == "sagittal":
        max_slice = shape[0]
    else:
        max_slice = shape[1]

    task_id = 0
    for slice_idx in range(0, max_slice, slice_step):
        # 获取该切片的分割数据
        if axis == "axial":
            seg_slice = seg_data[:, :, slice_idx]
            slice_height, slice_width = seg_slice.T.shape
        elif axis == "sagittal":
            seg_slice = seg_data[slice_idx, :, :]
            slice_height, slice_width = seg_slice.T.shape
        else:
            seg_slice = seg_data[:, slice_idx, :]
            slice_height, slice_width = seg_slice.T.shape

        # 找出该切片中存在的器官
        unique_labels = np.unique(seg_slice)
        slice_organs = []
        for label in unique_labels:
            if label > 0 and label in class_map_task:
                organ_name = class_map_task[label]
                if organ_name in detected_organs:
                    slice_organs.append(organ_name)

        if not slice_organs:
            continue

        # 生成图像 URL
        image_url = f"{base_url}/api/slice/{session_id}?axis={axis}&slice_idx={slice_idx}&window_center={window_center}&window_width={window_width}"

        # 生成问答对
        qa_pairs = generate_training_qa_pairs(
            slice_organs,
            {"axis": axis, "slice_idx": slice_idx},
            num_samples=num_qa_per_slice,
            include_augmentation=True
        )

        # 为每个问答对创建一个任务
        for qa in qa_pairs:
            task_id += 1

            # 创建器官分割预标注
            predictions = []
            for organ_name in slice_organs:
                organ_cn = get_organ_cn_name(organ_name)
                color = ORGAN_COLORS.get(organ_name, [128, 128, 128])

                predictions.append({
                    "type": "labels",
                    "value": {"labels": [organ_cn]},
                    "from_name": "organ_label",
                    "to_name": "image"
                })

            # 创建 Label Studio 任务
            ls_task = {
                "id": task_id,
                "data": {
                    "image": image_url,
                    "question": qa["question"],
                    "ai_answer": qa["answer"],
                    "session_id": session_id,
                    "slice_idx": slice_idx,
                    "axis": axis,
                    "organs": slice_organs,
                    "organs_cn": [get_organ_cn_name(o) for o in slice_organs]
                },
                "predictions": [{
                    "model_version": f"TotalSegmentator-{task}",
                    "score": 0.95,
                    "result": predictions
                }]
            }

            tasks.append(ls_task)

    return {
        "session_id": session_id,
        "total_tasks": len(tasks),
        "config_type": "vqa",
        "label_config": LABEL_STUDIO_CONFIG_VQA,
        "tasks": tasks
    }


@app.get("/api/labelstudio/export-offline/{session_id}")
async def export_label_studio_offline(
    session_id: str,
    slice_step: int = Query(10, ge=1, description="切片采样步长"),
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="切片轴向: axial, sagittal, coronal"),
    num_qa_per_slice: int = Query(3, description="每个切片生成的问答对数量"),
    window_center: int = Query(40, description="窗位"),
    window_width: int = Query(400, gt=0, description="窗宽"),
    include_overlay: bool = Query(True, description="是否包含器官叠加图像"),
    overlay_opacity: float = Query(0.5, description="叠加透明度 0-1")
):
    """
    离线导出 Label Studio 任务包

    生成完整的 ZIP 包，包含：
    - images/ 目录：所有切片的 PNG 图像
    - tasks.json：可直接导入 Label Studio 的任务文件
    - label_config.xml：Label Studio 标注配置
    - README.txt：使用说明

    下载后可以完全离线导入 Label Studio 使用
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载
    orig_data, seg_data = _ensure_session_data_loaded(session)
    if orig_data is None or seg_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")

    task_type = session.get("task", "total")
    class_map_task = class_map.get(task_type, {})
    shape = orig_data.shape
    detected_organs = session.get("organs", [])

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images_dir = temp_path / "images"
        images_dir.mkdir()

        # 创建 overlays 目录（如果需要叠加图像）
        overlays_dir = temp_path / "overlays"
        if include_overlay:
            overlays_dir.mkdir()

        # 根据轴向确定切片范围
        if axis == "axial":
            max_slice = shape[2]
        elif axis == "sagittal":
            max_slice = shape[0]
        else:
            max_slice = shape[1]

        tasks = []
        task_id = 0
        name_to_idx = {v: k for k, v in class_map_task.items()}

        for slice_idx in range(0, max_slice, slice_step):
            # 获取切片数据
            if axis == "axial":
                s_idx = min(max(0, slice_idx), shape[2] - 1)
                orig_slice = orig_data[:, :, s_idx].T
                seg_slice = seg_data[:, :, s_idx].T
            elif axis == "sagittal":
                s_idx = min(max(0, slice_idx), shape[0] - 1)
                orig_slice = orig_data[s_idx, :, :].T
                seg_slice = seg_data[s_idx, :, :].T
            else:
                s_idx = min(max(0, slice_idx), shape[1] - 1)
                orig_slice = orig_data[:, slice_idx, :].T
                seg_slice = seg_data[:, slice_idx, :].T

            # 找出该切片中存在的器官
            unique_labels = np.unique(seg_slice)
            slice_organs = []
            for label in unique_labels:
                if label > 0 and label in class_map_task:
                    organ_name = class_map_task[label]
                    if organ_name in detected_organs:
                        slice_organs.append(organ_name)

            if not slice_organs:
                continue

            # 生成原始切片图像
            windowed = apply_window(orig_slice, window_center, window_width)
            rgb_image = np.stack([windowed, windowed, windowed], axis=-1).astype(np.float32)
            rgb_image = np.flipud(rgb_image).astype(np.uint8)

            img_filename = f"{axis}_{slice_idx:04d}.png"
            img = Image.fromarray(rgb_image)
            img.save(images_dir / img_filename)

            # 生成叠加图像（如果需要）
            overlay_filename = None
            if include_overlay:
                overlay_rgb = np.stack([windowed, windowed, windowed], axis=-1).astype(np.float32)
                for organ_name in slice_organs:
                    if organ_name in name_to_idx:
                        organ_idx = name_to_idx[organ_name]
                        mask = seg_slice == organ_idx
                        if np.any(mask):
                            color = np.array(ORGAN_COLORS.get(organ_name, [255, 255, 0]), dtype=np.float32)
                            for c in range(3):
                                overlay_rgb[:, :, c] = np.where(
                                    mask,
                                    (1 - overlay_opacity) * overlay_rgb[:, :, c] + overlay_opacity * color[c],
                                    overlay_rgb[:, :, c]
                                )
                overlay_rgb = np.flipud(overlay_rgb).astype(np.uint8)
                overlay_filename = f"{axis}_{slice_idx:04d}_overlay.png"
                overlay_img = Image.fromarray(overlay_rgb)
                overlay_img.save(overlays_dir / overlay_filename)

            # 生成问答对
            qa_pairs = generate_training_qa_pairs(
                slice_organs,
                {"axis": axis, "slice_idx": slice_idx},
                num_samples=num_qa_per_slice,
                include_augmentation=True
            )

            # 为每个问答对创建任务
            for qa in qa_pairs:
                task_id += 1

                # 创建器官分割预标注
                predictions = []
                for organ_name in slice_organs:
                    organ_cn = get_organ_cn_name(organ_name)
                    predictions.append({
                        "type": "labels",
                        "value": {"labels": [organ_cn]},
                        "from_name": "organ_label",
                        "to_name": "image"
                    })

                # Label Studio 任务 - 使用相对路径
                ls_task = {
                    "id": task_id,
                    "data": {
                        "image": f"/data/local-files/?d=images/{img_filename}",
                        "question": qa["question"],
                        "ai_answer": qa["answer"],
                        "session_id": session_id,
                        "slice_idx": slice_idx,
                        "axis": axis,
                        "organs": slice_organs,
                        "organs_cn": [get_organ_cn_name(o) for o in slice_organs]
                    },
                    "predictions": [{
                        "model_version": f"TotalSegmentator-{task_type}",
                        "score": 0.95,
                        "result": predictions
                    }]
                }

                # 如果有叠加图像，添加引用
                if overlay_filename:
                    ls_task["data"]["overlay_image"] = f"/data/local-files/?d=overlays/{overlay_filename}"

                tasks.append(ls_task)

        # 保存任务 JSON 文件
        tasks_file = temp_path / "tasks.json"
        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        # 保存 Label Studio 配置
        config_file = temp_path / "label_config.xml"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(LABEL_STUDIO_CONFIG_VQA)

        # 创建使用说明
        readme_content = f"""TotalSegmentator Label Studio 导出包
======================================

Session ID: {session_id}
导出时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
切片轴向: {axis}
采样步长: {slice_step}
任务数量: {len(tasks)}

目录结构：
├── images/          - 切片图像 (PNG)
├── overlays/        - 带器官叠加的图像 (PNG) [如果启用]
├── tasks.json       - Label Studio 任务文件
├── label_config.xml - Label Studio 标注配置
└── README.txt       - 本说明文件

导入步骤：
1. 在 Label Studio 中创建新项目
2. 复制 label_config.xml 的内容到项目设置的 Labeling Interface
3. 配置 Local Files Storage:
   - 进入项目设置 -> Cloud Storage -> Add Source Storage
   - 选择 "Local files"
   - 设置 Absolute local path 为解压目录的绝对路径
   - 启用 "Treat every bucket object as a source file"
4. 导入 tasks.json 文件:
   - 进入项目 -> Import
   - 上传 tasks.json 文件

或者使用 URL 导入（如果图片可通过网络访问）：
1. 将 images 目录放到可访问的 Web 服务器
2. 修改 tasks.json 中的图片路径为完整 URL
3. 直接导入 tasks.json

检测到的器官：
{chr(10).join(f"- {get_organ_cn_name(o)} ({o})" for o in detected_organs[:20])}
{"..." if len(detected_organs) > 20 else ""}
"""
        readme_file = temp_path / "README.txt"
        with open(readme_file, "w", encoding="utf-8") as f:
            f.write(readme_content)

        # 创建 ZIP 包
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_path)
                    zf.write(file_path, arcname)

        zip_buffer.seek(0)

        filename = f"labelstudio_export_{session_id}_{axis}.zip"

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )


@app.get("/api/labelstudio/export-simple/{session_id}")
async def export_label_studio_simple(
    session_id: str,
    slice_step: int = Query(20, ge=1, description="切片采样步长"),
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="切片轴向: axial, sagittal, coronal"),
    window_center: int = Query(40, description="窗位"),
    window_width: int = Query(400, gt=0, description="窗宽"),
    max_slices: int = Query(10, description="最大切片数，避免文件过大")
):
    """
    简化版导出 - 直接生成可导入 Label Studio 的 JSON 文件

    图像以 Base64 编码嵌入 JSON 中，可以直接导入 Label Studio，
    无需配置本地文件存储。适合小批量快速测试。

    注意：由于 Base64 编码会增大文件体积，建议限制切片数量。
    """
    import base64

    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载
    orig_data, seg_data = _ensure_session_data_loaded(session)
    if orig_data is None or seg_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")

    task_type = session.get("task", "total")
    class_map_task = class_map.get(task_type, {})
    shape = orig_data.shape
    detected_organs = session.get("organs", [])

    # 根据轴向确定切片范围
    if axis == "axial":
        max_slice = shape[2]
    elif axis == "sagittal":
        max_slice = shape[0]
    else:
        max_slice = shape[1]

    tasks = []
    task_id = 0
    slices_processed = 0

    for slice_idx in range(0, max_slice, slice_step):
        if slices_processed >= max_slices:
            break

        # 获取切片数据
        if axis == "axial":
            s_idx = min(max(0, slice_idx), shape[2] - 1)
            orig_slice = orig_data[:, :, s_idx].T
            seg_slice = seg_data[:, :, s_idx].T
        elif axis == "sagittal":
            s_idx = min(max(0, slice_idx), shape[0] - 1)
            orig_slice = orig_data[s_idx, :, :].T
            seg_slice = seg_data[s_idx, :, :].T
        else:
            s_idx = min(max(0, slice_idx), shape[1] - 1)
            orig_slice = orig_data[:, slice_idx, :].T
            seg_slice = seg_data[:, slice_idx, :].T

        # 找出该切片中存在的器官
        unique_labels = np.unique(seg_slice)
        slice_organs = []
        for label in unique_labels:
            if label > 0 and label in class_map_task:
                organ_name = class_map_task[label]
                if organ_name in detected_organs:
                    slice_organs.append(organ_name)

        if not slice_organs:
            continue

        slices_processed += 1

        # 生成切片图像并转为 Base64
        windowed = apply_window(orig_slice, window_center, window_width)
        rgb_image = np.stack([windowed, windowed, windowed], axis=-1).astype(np.float32)
        rgb_image = np.flipud(rgb_image).astype(np.uint8)

        img = Image.fromarray(rgb_image)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_data_url = f"data:image/png;base64,{img_base64}"

        task_id += 1

        # 创建简化的 Label Studio 任务
        ls_task = {
            "id": task_id,
            "data": {
                "image": image_data_url,
                "session_id": session_id,
                "slice_idx": slice_idx,
                "axis": axis,
                "organs_cn": [get_organ_cn_name(o) for o in slice_organs]
            }
        }

        tasks.append(ls_task)

    # 直接返回 JSON 数组，让前端处理下载
    return tasks


@app.post("/api/labelstudio/import-annotations")
async def import_label_studio_annotations(
    annotations_file: UploadFile = File(...),
    output_format: str = Query("llama-factory", description="输出格式: llama-factory, alpaca, raw")
):
    """
    导入 Label Studio 标注结果，转换为训练数据格式

    接受从 Label Studio 导出的 JSON 文件，解析标注结果并转换为 LLaMA-Factory 训练格式
    """
    content = await annotations_file.read()

    try:
        annotations = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    training_data = []

    for task in annotations:
        task_data = task.get("data", {})
        task_annotations = task.get("annotations", [])

        # 如果没有人工标注，跳过
        if not task_annotations:
            continue

        # 获取最新的标注（通常是最后一个）
        latest_annotation = task_annotations[-1]
        annotation_result = latest_annotation.get("result", [])

        # 解析标注结果
        human_answer = None
        decision = None
        notes = None

        for result in annotation_result:
            result_type = result.get("type")
            from_name = result.get("from_name")
            value = result.get("value", {})

            if from_name == "human_answer":
                human_answer = value.get("text", [""])[0] if isinstance(value.get("text"), list) else value.get("text", "")
            elif from_name == "decision":
                decision = value.get("choices", [""])[0] if isinstance(value.get("choices"), list) else value.get("choices", "")
            elif from_name == "notes":
                notes = value.get("text", [""])[0] if isinstance(value.get("text"), list) else value.get("text", "")

        # 只使用确认正确或修正后的数据
        if decision not in ["确认正确", "需要修正"]:
            continue

        # 使用人工审核的答案，如果为空则使用 AI 答案
        final_answer = human_answer if human_answer else task_data.get("ai_answer", "")
        question = task_data.get("question", "")
        image_url = task_data.get("image", "")

        if not question or not final_answer:
            continue

        # 转换为训练格式
        if output_format == "llama-factory":
            training_data.append({
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": final_answer}
                ],
                "images": [image_url],
                "metadata": {
                    "decision": decision,
                    "notes": notes,
                    "session_id": task_data.get("session_id"),
                    "slice_idx": task_data.get("slice_idx"),
                    "axis": task_data.get("axis")
                }
            })
        elif output_format == "alpaca":
            training_data.append({
                "instruction": question,
                "input": "",
                "output": final_answer,
                "images": [image_url],
                "metadata": {
                    "decision": decision,
                    "notes": notes
                }
            })
        else:  # raw
            training_data.append({
                "question": question,
                "answer": final_answer,
                "ai_answer": task_data.get("ai_answer"),
                "image": image_url,
                "decision": decision,
                "notes": notes,
                "organs": task_data.get("organs", []),
                "session_id": task_data.get("session_id"),
                "slice_idx": task_data.get("slice_idx")
            })

    return {
        "format": output_format,
        "total_samples": len(training_data),
        "accepted": len([d for d in training_data if d.get("metadata", {}).get("decision") == "确认正确"]),
        "corrected": len([d for d in training_data if d.get("metadata", {}).get("decision") == "需要修正"]),
        "data": training_data
    }


@app.get("/api/labelstudio/export-images/{session_id}")
async def export_images_for_label_studio(
    session_id: str,
    slice_step: int = Query(10, ge=1, description="切片采样步长"),
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="切片轴向"),
    window_center: int = Query(40, description="窗位"),
    window_width: int = Query(400, gt=0, description="窗宽"),
    output_format: str = Query("url", description="输出格式: url (返回URL列表), file (保存到文件)")
):
    """
    导出用于 Label Studio 的图像

    支持两种模式：
    - url: 返回可以直接在 Label Studio 中使用的图像 URL
    - file: 将图像保存到服务器目录，返回文件路径
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Segmentation not completed")

    # 确保数据已加载（从内存或磁盘）
    orig_data, _ = _ensure_session_data_loaded(session)
    if orig_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")

    shape = orig_data.shape

    # 根据轴向确定切片范围
    if axis == "axial":
        max_slice = shape[2]
    elif axis == "sagittal":
        max_slice = shape[0]
    else:
        max_slice = shape[1]

    images = []

    if output_format == "file":
        # 创建输出目录
        out_path = DATA_DIR / "labelstudio_export" / session_id
        out_path.mkdir(parents=True, exist_ok=True)

        for slice_idx in range(0, max_slice, slice_step):
            if axis == "axial":
                orig_slice = orig_data[:, :, slice_idx].T
            elif axis == "sagittal":
                orig_slice = orig_data[slice_idx, :, :].T
            else:
                orig_slice = orig_data[:, slice_idx, :].T

            windowed = apply_window(orig_slice, window_center, window_width)

            filename = f"{axis}_{slice_idx:04d}.png"
            filepath = out_path / filename

            img = Image.fromarray(windowed)
            img.save(filepath)

            images.append({
                "slice_idx": slice_idx,
                "axis": axis,
                "filename": filename,
                "path": str(filepath)
            })

        return {
            "session_id": session_id,
            "output_dir": str(out_path),
            "total_images": len(images),
            "images": images
        }
    else:  # url
        base_url = f"/api/slice/{session_id}"

        for slice_idx in range(0, max_slice, slice_step):
            url = f"{base_url}?axis={axis}&slice_idx={slice_idx}&window_center={window_center}&window_width={window_width}"
            images.append({
                "slice_idx": slice_idx,
                "axis": axis,
                "url": url
            })

        return {
            "session_id": session_id,
            "total_images": len(images),
            "images": images
        }


@app.get("/api/labelstudio/workflow-status")
async def get_workflow_status():
    """
    获取标注工作流状态概览

    返回所有 session 的状态统计，用于监控标注进度
    """
    sessions_info = []

    for session_id, session in SESSIONS.items():
        sessions_info.append({
            "session_id": session_id,
            "status": session.get("status", "unknown"),
            "filename": session.get("filename", "unknown"),
            "task": session.get("task", "unknown"),
            "organs_count": len(session.get("organs", [])),
            "created_at": session.get("created_at", None)
        })

    return {
        "total_sessions": len(sessions_info),
        "completed": len([s for s in sessions_info if s["status"] == "completed"]),
        "processing": len([s for s in sessions_info if s["status"] == "processing"]),
        "sessions": sessions_info
    }


# ============ LLM 配置 API ============

class LLMConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None  # LLM 请求超时时间（秒）
    system_prompt: Optional[str] = None
    qa_prompt_template: Optional[str] = None


@app.get("/api/config/llm")
async def get_llm_config():
    """获取 LLM 配置（不返回 API Key）"""
    config = get_config()
    return {
        "enabled": config.llm.enabled,
        "provider": config.llm.provider,
        "base_url": config.llm.base_url,
        "model": config.llm.model,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
        "timeout": config.llm.timeout,
        "system_prompt": config.llm.system_prompt,
        "qa_prompt_template": config.llm.qa_prompt_template,
        "has_api_key": bool(config.llm.api_key)
    }


@app.post("/api/config/llm")
async def set_llm_config(request: LLMConfigRequest):
    """更新 LLM 配置"""
    llm_config = update_llm_config(
        enabled=request.enabled,
        api_key=request.api_key,
        base_url=request.base_url,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        timeout=request.timeout,
        system_prompt=request.system_prompt,
        qa_prompt_template=request.qa_prompt_template
    )
    return {
        "status": "updated",
        "enabled": llm_config.enabled,
        "model": llm_config.model,
        "timeout": llm_config.timeout,
        "has_api_key": bool(llm_config.api_key)
    }


class LLMTestRequest(BaseModel):
    """LLM 测试请求（可选临时配置）"""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/config/llm/test")
async def test_llm_connection(request: Optional[LLMTestRequest] = None):
    """测试 LLM 连接（支持临时配置测试）"""
    config = get_config()

    # 使用请求中的临时配置或后端保存的配置
    base_url = (request.base_url if request and request.base_url else config.llm.base_url)
    api_key = (request.api_key if request and request.api_key else config.llm.api_key)
    model = (request.model if request and request.model else config.llm.model)

    if not api_key:
        return {"status": "error", "message": "未配置 API Key"}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 10
                }
            )
            if response.status_code == 200:
                return {"status": "success", "message": f"连接成功 ({model})"}
            else:
                error_text = response.text[:200] if response.text else str(response.status_code)
                return {"status": "error", "message": f"API 错误: {error_text}"}
    except Exception as e:
        return {"status": "error", "message": f"连接失败: {str(e)}"}


# ==================== QA 模板配置 API ====================

# 注意：QA 模板状态已迁移到 config.py 进行持久化存储
# 使用 get_qa_template_state() 获取状态，不再使用内存变量


class QATemplateCategory(BaseModel):
    """单个类别的模板"""
    templates: List[str]


class QATemplatesRequest(BaseModel):
    """QA 模板更新请求"""
    organ_identification: Optional[List[str]] = None
    organ_location: Optional[List[str]] = None
    organ_description: Optional[List[str]] = None
    diagnostic_analysis: Optional[List[str]] = None
    comprehensive_report: Optional[List[str]] = None


class TemplateSelectionRequest(BaseModel):
    """模板选择状态更新请求"""
    category: str
    enabled: Optional[bool] = None
    priority: Optional[bool] = None


class TemplateBatchSelectionRequest(BaseModel):
    """批量更新模板选择状态"""
    selections: Dict[str, Dict[str, bool]]  # {category: {enabled: bool, priority: bool}}


@app.get("/api/config/qa-templates")
async def get_qa_templates_api():
    """获取当前 QA 模板配置"""
    # 从持久化存储获取状态
    qa_state = get_qa_template_state()

    # 合并默认模板和自定义模板
    result = {}
    for category, templates in QA_TEMPLATES.items():
        # 获取选择状态，默认启用、非重点
        selection = qa_state.selection_state.get(category, {"enabled": True, "priority": False})
        custom_templates = qa_state.custom_templates.get(category, [])

        if custom_templates:
            # 使用自定义模板
            # 自定义模板格式可能是 [{"q": "...", "type": "..."}] 或 ["..."]
            if custom_templates and isinstance(custom_templates[0], dict):
                template_list = [t.get("q", t) if isinstance(t, dict) else t for t in custom_templates]
            else:
                template_list = custom_templates
            result[category] = {
                "templates": template_list,
                "is_custom": True,
                "enabled": selection.get("enabled", True),
                "priority": selection.get("priority", False)
            }
        else:
            # 使用默认模板
            result[category] = {
                "templates": [t["q"] for t in templates],
                "is_custom": False,
                "enabled": selection.get("enabled", True),
                "priority": selection.get("priority", False)
            }

    return {
        "categories": result,
        "available_categories": list(QA_TEMPLATES.keys()),
        "description": {
            "organ_identification": "器官识别类问题 - 询问图像中可见的器官",
            "organ_location": "器官定位类问题 - 询问特定器官的位置（支持 {organ} 占位符）",
            "organ_description": "器官描述类问题 - 询问器官形态特征（支持 {organ} 占位符）",
            "diagnostic_analysis": "诊断分析类问题 - 要求进行初步诊断分析",
            "comprehensive_report": "综合报告类问题 - 要求生成影像学报告",
            "region_analysis": "区域分析类问题 - 分析特定区域",
            "anatomical_relation": "解剖关系类问题 - 描述器官间关系",
            "clinical_relevance": "临床相关类问题 - 临床价值分析"
        }
    }


@app.post("/api/config/qa-templates")
async def set_qa_templates(request: QATemplatesRequest):
    """更新 QA 模板配置"""
    updated_categories = []

    if request.organ_identification is not None:
        CUSTOM_QA_TEMPLATES["organ_identification"] = [
            {"q": q, "type": "list_organs"} for q in request.organ_identification
        ]
        updated_categories.append("organ_identification")

    if request.organ_location is not None:
        CUSTOM_QA_TEMPLATES["organ_location"] = [
            {"q": q, "type": "locate_organ"} for q in request.organ_location
        ]
        updated_categories.append("organ_location")

    if request.organ_description is not None:
        CUSTOM_QA_TEMPLATES["organ_description"] = [
            {"q": q, "type": "describe_organ"} for q in request.organ_description
        ]
        updated_categories.append("organ_description")

    if request.diagnostic_analysis is not None:
        CUSTOM_QA_TEMPLATES["diagnostic_analysis"] = [
            {"q": q, "type": "diagnosis"} for q in request.diagnostic_analysis
        ]
        updated_categories.append("diagnostic_analysis")

    if request.comprehensive_report is not None:
        CUSTOM_QA_TEMPLATES["comprehensive_report"] = [
            {"q": q, "type": "report"} for q in request.comprehensive_report
        ]
        updated_categories.append("comprehensive_report")

    return {
        "status": "updated",
        "updated_categories": updated_categories,
        "message": f"已更新 {len(updated_categories)} 个模板类别"
    }


@app.post("/api/config/qa-templates/selection")
async def set_template_selection(request: TemplateSelectionRequest):
    """设置单个模板类别的选择状态"""
    if request.category not in QA_TEMPLATES:
        return {"status": "error", "message": f"未知的模板类别: {request.category}"}

    # 使用 config.py 中的持久化函数
    new_state = update_template_selection(
        category=request.category,
        enabled=request.enabled,
        priority=request.priority
    )

    return {
        "status": "updated",
        "category": request.category,
        "state": new_state
    }


@app.post("/api/config/qa-templates/selection/batch")
async def set_template_selection_batch(request: TemplateBatchSelectionRequest):
    """批量设置模板选择状态"""
    # 过滤只有有效类别
    valid_selections = {
        category: state for category, state in request.selections.items()
        if category in QA_TEMPLATES
    }

    # 使用 config.py 中的持久化函数
    updated = update_template_selection_batch(valid_selections)

    return {
        "status": "updated",
        "updated_categories": updated,
        "message": f"已更新 {len(updated)} 个模板类别的选择状态"
    }


@app.post("/api/config/qa-templates/reset")
async def reset_qa_templates(category: Optional[str] = None):
    """重置 QA 模板为默认值"""
    # 使用 config.py 中的持久化函数
    reset_qa_templates_config(category)

    # 同时清理内存中的自定义模板（向后兼容）
    if category:
        if category in CUSTOM_QA_TEMPLATES:
            del CUSTOM_QA_TEMPLATES[category]
        return {"status": "reset", "category": category}
    else:
        CUSTOM_QA_TEMPLATES.clear()
        return {"status": "reset_all", "message": "所有模板已重置为默认值"}


def get_effective_templates(category: str) -> List[Dict[str, str]]:
    """获取有效的模板（优先使用自定义模板）"""
    if category in CUSTOM_QA_TEMPLATES and CUSTOM_QA_TEMPLATES[category]:
        return CUSTOM_QA_TEMPLATES[category]
    return QA_TEMPLATES.get(category, [])


def get_template_by_priority() -> Tuple[str, List[Dict[str, str]]]:
    """
    根据优先级获取模板类别和模板列表

    优先级顺序：
    1. 标记为"重点"的模板（随机选一个重点类别）
    2. "已启用"的模板（随机选一个启用类别）
    3. 所有模板（随机选一个）

    Returns:
        (category, templates) 元组
    """
    # 收集各优先级的类别
    priority_categories = []
    enabled_categories = []
    all_categories = list(QA_TEMPLATES.keys())

    # 从 config.py 获取所有状态
    all_states = get_all_template_selection_states()

    for category in all_categories:
        state = all_states.get(category, {"enabled": True, "priority": False})
        if state.get("priority", False) and state.get("enabled", True):
            priority_categories.append(category)
        elif state.get("enabled", True):
            enabled_categories.append(category)

    # 按优先级选择
    if priority_categories:
        selected_category = random.choice(priority_categories)
    elif enabled_categories:
        selected_category = random.choice(enabled_categories)
    else:
        selected_category = random.choice(all_categories)

    templates = get_effective_templates(selected_category)
    return selected_category, templates


def get_all_enabled_templates() -> Dict[str, List[Dict[str, str]]]:
    """
    获取所有启用的模板，按优先级排序

    Returns:
        {category: templates} 字典，重点类别排在前面
    """
    priority_templates = {}
    enabled_templates = {}

    # 从 config.py 获取所有状态
    all_states = get_all_template_selection_states()

    for category in QA_TEMPLATES.keys():
        state = all_states.get(category, {"enabled": True, "priority": False})
        if state.get("enabled", True):
            templates = get_effective_templates(category)
            if state.get("priority", False):
                priority_templates[category] = templates
            else:
                enabled_templates[category] = templates

    # 合并，重点在前
    result = {}
    result.update(priority_templates)
    result.update(enabled_templates)

    # 如果没有任何启用的，返回所有
    if not result:
        for category in QA_TEMPLATES.keys():
            result[category] = get_effective_templates(category)

    return result


def get_template_selection_for_llm() -> Tuple[Dict[str, bool], List[str]]:
    """
    获取模板选择状态，用于传递给 LLM 问答生成器

    Returns:
        (enabled_categories, priority_categories) 元组
        - enabled_categories: {category: enabled} 字典
        - priority_categories: 重点类别列表
    """
    enabled_categories = {}
    priority_categories = []

    # 从 config.py 获取所有状态
    all_states = get_all_template_selection_states()

    for category in QA_TEMPLATES.keys():
        state = all_states.get(category, {"enabled": True, "priority": False})
        enabled = state.get("enabled", True)
        priority = state.get("priority", False)

        enabled_categories[category] = enabled
        if priority and enabled:
            priority_categories.append(category)

    return enabled_categories, priority_categories


# ============ 任务管理 API ============

class BatchDeleteRequest(BaseModel):
    """批量删除请求"""
    session_ids: List[str]
    confirm: bool = False  # 必须设为 True 才能删除


@app.get("/api/tasks")
async def list_tasks(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(10, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="状态筛选")
):
    """
    获取任务列表

    支持分页和状态筛选，返回任务列表和总数
    """
    if USE_DATABASE and not _database_mode_enabled():
        print("ℹ /api/tasks 使用回退模式（数据库当前不可用）")

    task_service = get_task_service(SESSIONS, DATA_DIR)
    if _database_mode_enabled():
        try:
            tasks, total = await task_service.list_tasks(page, page_size, status)
        except Exception as e:
            _disable_database_mode("list_tasks failed", e)
            tasks, total = task_service._list_tasks_memory(page, page_size, status)
    else:
        tasks, total = task_service._list_tasks_memory(page, page_size, status)

    return {
        "tasks": tasks,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 0
    }


@app.get("/api/tasks/active/current")
async def get_active_task():
    """
    获取当前进行中的任务（用于页面刷新后恢复）

    返回最近一个状态为 'uploading' 或 'processing' 的任务
    """
    # 查找进行中的任务
    active_tasks = []
    for session_id, session in SESSIONS.items():
        status = session.get("status", "unknown")
        if status in ("uploading", "processing"):
            active_tasks.append({
                "session_id": session_id,
                "filename": session.get("filename", "unknown"),
                "file_type": session.get("file_type"),
                "status": status,
                "task_type": session.get("task", "total"),
                "created_at": session.get("created_at"),
                "progress": session.get("progress", 0),  # 进度百分比
                "progress_message": session.get("progress_message", ""),
            })

    if not active_tasks:
        return {"active_task": None}

    # 返回最近的一个
    active_tasks.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"active_task": active_tasks[0]}


@app.get("/api/tasks/{session_id}/status")
async def get_task_status(session_id: str):
    """
    获取任务状态（用于轮询进度）

    返回任务的当前状态、进度和结果
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="任务不存在")

    status = session.get("status", "unknown")

    result = {
        "session_id": session_id,
        "status": status,
        "progress": session.get("progress", 0),
        "progress_message": session.get("progress_message", ""),
        "filename": session.get("filename"),
        "task_type": session.get("task"),
    }

    # 如果已完成，返回完整信息
    if status == "completed":
        result["organs"] = session.get("organs", [])
        result["shape"] = session.get("shape")
        result["device_used"] = session.get("device_used")

    # 如果出错，返回错误信息
    if status == "error":
        result["error_message"] = session.get("error", "未知错误")

    return result


@app.get("/api/tasks/{session_id}")
async def get_task_detail(session_id: str):
    """获取单个任务详情"""
    task_service = get_task_service(SESSIONS, DATA_DIR)
    if _database_mode_enabled():
        try:
            task = await task_service.get_task(session_id)
        except Exception as e:
            _disable_database_mode("get_task_detail failed", e)
            task = task_service._get_task_memory(session_id)
    else:
        task = task_service._get_task_memory(session_id)

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    return task


@app.delete("/api/tasks/{session_id}")
async def delete_task(
    session_id: str,
    confirm: bool = Query(False, description="确认删除")
):
    """
    删除单个任务（硬删除）

    需要 confirm=true 参数确认删除
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="请设置 confirm=true 确认删除。此操作将永久删除所有相关数据，无法恢复！"
        )

    task_service = get_task_service(SESSIONS, DATA_DIR)
    if _database_mode_enabled():
        try:
            success = await task_service.delete_task(session_id)
        except Exception as e:
            _disable_database_mode("delete_task endpoint failed", e)
            success = task_service._delete_task_memory(session_id)
    else:
        success = task_service._delete_task_memory(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="任务不存在")

    return {
        "status": "deleted",
        "session_id": session_id,
        "message": "任务已永久删除"
    }


@app.post("/api/tasks/batch-delete")
async def batch_delete_tasks(request: BatchDeleteRequest):
    """
    批量删除任务（硬删除）

    需要 confirm=true 确认删除
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="请设置 confirm=true 确认删除。此操作将永久删除所有相关数据，无法恢复！"
        )

    if not request.session_ids:
        raise HTTPException(status_code=400, detail="请提供要删除的任务 ID 列表")

    task_service = get_task_service(SESSIONS, DATA_DIR)

    def _batch_delete_memory() -> Dict[str, Any]:
        deleted = []
        failed = []
        for sid in request.session_ids:
            try:
                ok = task_service._delete_task_memory(sid)
                if ok:
                    deleted.append(sid)
                else:
                    failed.append({"session_id": sid, "error": "任务不存在"})
            except Exception as mem_err:
                failed.append({"session_id": sid, "error": str(mem_err)})
        return {"deleted": deleted, "failed": failed}

    if _database_mode_enabled():
        try:
            result = await task_service.batch_delete_tasks(request.session_ids)
        except Exception as e:
            _disable_database_mode("batch_delete_tasks failed", e)
            result = _batch_delete_memory()
    else:
        result = _batch_delete_memory()

    return {
        "status": "completed",
        "deleted_count": len(result["deleted"]),
        "failed_count": len(result["failed"]),
        "deleted": result["deleted"],
        "failed": result["failed"]
    }


@app.get("/api/tasks/statistics/summary")
async def get_task_statistics():
    """获取任务统计信息"""
    task_service = get_task_service(SESSIONS, DATA_DIR)
    if _database_mode_enabled():
        try:
            stats = await task_service.get_statistics()
        except Exception as e:
            _disable_database_mode("get_statistics failed", e)
            stats = task_service._get_statistics_memory()
    else:
        stats = task_service._get_statistics_memory()
    return stats


# ==================== Label Studio 直接集成 API ====================

class LabelStudioSyncRequest(BaseModel):
    """Label Studio 同步请求体"""
    organs: Optional[List[str]] = None  # 要同步的器官列表，为空则使用全部


@app.post("/api/labelstudio/sync/{session_id}")
async def sync_to_label_studio(
    session_id: str,
    request: Optional[LabelStudioSyncRequest] = None,
    project_name: Optional[str] = Query(None, description="Label Studio 项目名称（为空则使用配置默认值）"),
    slice_step: Optional[int] = Query(None, ge=1, description="切片采样步长（为空则使用配置默认值）"),
    axis: Optional[Literal["axial", "sagittal", "coronal"]] = Query(None, description="切片轴向: axial, sagittal, coronal（为空则使用配置默认值）")
):
    """
    将会话数据同步到 Label Studio

    完整流程:
    1. 生成切片图像并上传到 MinIO
    2. 创建或获取 Label Studio 项目
    3. 配置 S3 存储连接
    4. 自动导入标注任务

    前置条件:
    - Label Studio 已启动并配置 API Key
    - MinIO 存储可用
    - 会话已完成分割
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="分割任务未完成")

    # 从配置获取默认值
    ls_config = get_labelstudio_config()
    final_project_name = project_name or ls_config.default_project_name or "TotalSegmentator-VQA"
    final_slice_step = slice_step if slice_step is not None else ls_config.default_slice_step or 10
    final_axis = axis or ls_config.default_axis or "axial"

    # 获取要同步的器官列表：优先使用请求体中的，否则使用 session 中的全部器官
    session_organs = session.get("organs", [])
    final_organs = (request.organs if request and request.organs else None) or session_organs

    try:
        from api.labelstudio_service import sync_session_to_labelstudio

        result = await sync_session_to_labelstudio(
            session_id=session_id,
            session_data=session,
            project_name=final_project_name,
            slice_step=final_slice_step,
            axis=final_axis,
            organs=final_organs
        )

        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="Label Studio SDK 未安装，请运行: pip install label-studio-sdk"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"同步失败: {str(e)}")


@app.get("/api/labelstudio/status")
async def get_label_studio_status():
    """
    获取 Label Studio 集成状态

    返回:
    - Label Studio 连接状态
    - MinIO 存储状态
    - 配置信息
    """
    from api.storage import check_minio_connection, MINIO_ENDPOINT, MINIO_BUCKET

    # 从配置获取 Label Studio 设置
    ls_config = get_labelstudio_config()

    status = {
        "labelstudio": {
            "configured": bool(ls_config.api_key),
            "url": ls_config.url,
            "connected": False,
            "sdk_available": False,
            "default_project_name": ls_config.default_project_name
        },
        "minio": {
            "endpoint": MINIO_ENDPOINT,
            "bucket": MINIO_BUCKET,
            "connected": False
        }
    }

    # 检查 SDK
    try:
        from label_studio_sdk import Client
        status["labelstudio"]["sdk_available"] = True
    except ImportError:
        pass

    # 检查 Label Studio 连接
    if status["labelstudio"]["configured"] and status["labelstudio"]["sdk_available"]:
        try:
            from api.labelstudio_service import get_labelstudio_service
            ls_service = get_labelstudio_service()
            status["labelstudio"]["connected"] = ls_service.connect()
        except Exception:
            pass

    # 检查 MinIO 连接
    status["minio"]["connected"] = check_minio_connection()

    return status


class LabelStudioConfigRequest(BaseModel):
    """Label Studio 配置请求"""
    enabled: Optional[bool] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    default_project_name: Optional[str] = None
    default_slice_step: Optional[int] = None
    default_axis: Optional[str] = None
    minio_public_url: Optional[str] = None


@app.get("/api/config/labelstudio")
async def get_labelstudio_config_api():
    """获取 Label Studio 配置（不返回完整 API Key）"""
    ls_config = get_labelstudio_config()
    return {
        "enabled": ls_config.enabled,
        "url": ls_config.url,
        "default_project_name": ls_config.default_project_name,
        "default_slice_step": ls_config.default_slice_step,
        "default_axis": ls_config.default_axis,
        "minio_public_url": ls_config.minio_public_url,
        "has_api_key": bool(ls_config.api_key),
        # 显示 API Key 的部分内容（安全考虑）
        "api_key_preview": f"{ls_config.api_key[:20]}..." if ls_config.api_key and len(ls_config.api_key) > 20 else ""
    }


@app.post("/api/config/labelstudio")
async def set_labelstudio_config_api(request: LabelStudioConfigRequest):
    """更新 Label Studio 配置"""
    ls_config = update_labelstudio_config(
        enabled=request.enabled,
        url=request.url,
        api_key=request.api_key,
        default_project_name=request.default_project_name,
        default_slice_step=request.default_slice_step,
        default_axis=request.default_axis,
        minio_public_url=request.minio_public_url
    )
    return {
        "status": "updated",
        "enabled": ls_config.enabled,
        "url": ls_config.url,
        "default_project_name": ls_config.default_project_name,
        "has_api_key": bool(ls_config.api_key)
    }


@app.get("/api/labelstudio/projects")
async def list_label_studio_projects():
    """
    列出 Label Studio 中的所有项目
    """
    try:
        from api.labelstudio_service import get_labelstudio_service

        ls_service = get_labelstudio_service()
        if not ls_service.connect():
            raise HTTPException(status_code=503, detail="无法连接到 Label Studio")

        projects = ls_service._get_projects()
        return [{
            "id": p.id,
            "title": p.title,
            "task_count": getattr(p, 'task_count', 0) or 0
        } for p in projects]

    except ImportError:
        raise HTTPException(status_code=500, detail="Label Studio SDK 未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/labelstudio/upload-slices/{session_id}")
async def upload_slices_to_minio(
    session_id: str,
    slice_step: int = Query(10, ge=1, description="切片采样步长"),
    axis: Literal["axial", "sagittal", "coronal"] = Query("axial", description="切片轴向")
):
    """
    仅上传切片图像到 MinIO（不导入 Label Studio）

    用于手动配置 Label Studio 的情况
    """
    session = await _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="分割任务未完成")

    try:
        # 确保数据已加载
        orig_data, seg_data = _ensure_session_data_loaded(session)
        if orig_data is None or seg_data is None:
            raise HTTPException(status_code=400, detail="无法加载数据")

        from api.labelstudio_service import generate_slice_images

        organs = session.get("organs", [])

        slices_info = generate_slice_images(
            session_id=session_id,
            ct_data=orig_data,
            seg_data=seg_data,
            organs=organs,
            axis=axis,
            slice_step=slice_step
        )

        return {
            "success": True,
            "uploaded_count": len(slices_info),
            "slices": slices_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=28000)

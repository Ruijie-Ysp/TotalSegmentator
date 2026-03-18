"""
Label Studio ML Backend - TotalSegmentator 预标注服务

实现 Label Studio ML Backend 标准接口：
- GET  /health          健康检查
- POST /setup           初始化设置
- POST /predict         预标注（分割 + VQA 问答）
- POST /webhook         接收训练事件

工作流程：
1. 用户上传 DICOM/NIfTI 图像到 Label Studio
2. Label Studio 调用 /predict 接口
3. TotalSegmentator 执行分割
4. 返回分割蒙版和问答对作为预标注
"""
import os
import io
import json
import base64
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import numpy as np
import nibabel as nib
from PIL import Image
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# TotalSegmentator
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map

# 本地模块
from api.llm_qa_generator import generate_qa_pairs, get_organ_cn

# ============ 配置 ============
ML_BACKEND_VERSION = "1.0.0"
MODEL_NAME = "TotalSegmentator"
SUPPORTED_TASKS = ["total", "lung_vessels", "body", "cerebral_bleed", "hip_implant", 
                   "coronary_arteries", "pleural_pericard_effusion", "liver_vessels"]

# 创建路由
router = APIRouter(prefix="/ml", tags=["Label Studio ML Backend"])


# ============ 请求/响应模型 ============
class SetupRequest(BaseModel):
    """Label Studio 发送的设置请求"""
    project: Optional[str] = None
    schema: Optional[str] = None
    hostname: Optional[str] = None
    access_token: Optional[str] = None


class SetupResponse(BaseModel):
    """设置响应"""
    model_version: str
    model_name: str = MODEL_NAME


class PredictRequest(BaseModel):
    """预测请求"""
    tasks: List[Dict[str, Any]]
    model_version: Optional[str] = None
    project: Optional[str] = None
    label_config: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class WebhookRequest(BaseModel):
    """Webhook 请求（标注完成事件）"""
    action: str
    annotation: Optional[Dict[str, Any]] = None
    task: Optional[Dict[str, Any]] = None
    project: Optional[Dict[str, Any]] = None


# ============ 工具函数 ============
def get_organ_colors() -> Dict[str, List[int]]:
    """器官颜色映射"""
    return {
        "liver": [139, 69, 19], "spleen": [148, 0, 211], "pancreas": [255, 165, 0],
        "kidney_left": [220, 20, 60], "kidney_right": [255, 99, 71],
        "gallbladder": [0, 128, 0], "stomach": [255, 192, 203],
        "heart": [255, 0, 0], "aorta": [255, 0, 0],
        "lung_upper_lobe_left": [135, 206, 235], "lung_lower_lobe_left": [70, 130, 180],
        "lung_upper_lobe_right": [100, 149, 237], "lung_middle_lobe_right": [65, 105, 225],
        "lung_lower_lobe_right": [0, 0, 205], "trachea": [173, 216, 230],
        "esophagus": [255, 218, 185], "vertebrae_L1": [192, 192, 192],
        "hip_left": [169, 169, 169], "hip_right": [169, 169, 169],
        "femur_left": [128, 128, 128], "femur_right": [128, 128, 128],
    }


ORGAN_COLORS = get_organ_colors()


def apply_window(data: np.ndarray, center: int = 40, width: int = 400) -> np.ndarray:
    """应用窗宽窗位"""
    min_val = center - width / 2
    max_val = center + width / 2
    windowed = np.clip(data, min_val, max_val)
    windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return windowed


def rle_encode(mask: np.ndarray) -> List[int]:
    """将二值掩码编码为 RLE 格式（Label Studio BrushLabels 格式）"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


def mask_to_rle_string(mask: np.ndarray) -> str:
    """将掩码转换为 Label Studio RLE 字符串"""
    rle = rle_encode(mask.astype(np.uint8))
    return ' '.join(map(str, rle))


# ============ API 接口 ============
@router.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "UP",
        "model_name": MODEL_NAME,
        "model_version": ML_BACKEND_VERSION,
        "supported_tasks": SUPPORTED_TASKS
    }


@router.post("/setup")
async def setup(request: SetupRequest):
    """初始化设置 - Label Studio 连接时调用"""
    return SetupResponse(model_version=ML_BACKEND_VERSION)


@router.post("/predict")
async def predict(request: PredictRequest):
    """
    预标注接口 - Label Studio 核心调用

    处理流程：
    1. 接收任务数据（图像 URL 或 base64）
    2. 下载/解析图像
    3. 调用 TotalSegmentator 分割
    4. 生成问答对（可选，使用 LLM）
    5. 返回 Label Studio 格式的预标注结果
    """
    predictions = []

    for task in request.tasks:
        try:
            task_prediction = await _process_single_task(
                task=task,
                label_config=request.label_config,
                params=request.params or {}
            )
            predictions.append(task_prediction)
        except Exception as e:
            print(f"[ML Backend] 处理任务失败: {e}")
            # 返回空预测，不阻断整个批次
            predictions.append({
                "model_version": ML_BACKEND_VERSION,
                "score": 0.0,
                "result": [],
                "error": str(e)
            })

    return {"predictions": predictions}


async def _process_single_task(
    task: Dict[str, Any],
    label_config: Optional[str],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """处理单个任务的预标注"""

    task_data = task.get("data", {})

    # 获取图像数据
    image_data = await _get_image_data(task_data)
    if image_data is None:
        return {"model_version": ML_BACKEND_VERSION, "score": 0.0, "result": []}

    # 解析 label_config 确定需要的标注类型
    annotation_types = _parse_label_config(label_config)

    results = []
    score = 0.95  # 置信度分数

    # 如果需要分割预标注
    if "brushlabels" in annotation_types or "segmentation" in annotation_types:
        seg_results = await _generate_segmentation_predictions(
            image_data=image_data,
            task_data=task_data,
            params=params
        )
        results.extend(seg_results)

    # 如果需要 VQA 问答预标注
    if "textarea" in annotation_types or "vqa" in annotation_types:
        vqa_results = await _generate_vqa_predictions(
            image_data=image_data,
            task_data=task_data,
            params=params
        )
        results.extend(vqa_results)

    return {
        "model_version": ML_BACKEND_VERSION,
        "score": score,
        "result": results
    }


async def _get_image_data(task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从任务数据中获取图像"""
    import httpx

    # 尝试多种图像来源
    image_url = task_data.get("image") or task_data.get("img") or task_data.get("url")
    image_base64 = task_data.get("image_base64") or task_data.get("data")

    if image_base64:
        # Base64 编码的图像
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_bytes = base64.b64decode(image_base64)
            return {"type": "bytes", "data": image_bytes}
        except Exception as e:
            print(f"[ML Backend] Base64 解码失败: {e}")

    if image_url:
        # URL 图像 - 下载
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(image_url)
                if response.status_code == 200:
                    return {"type": "bytes", "data": response.content, "url": image_url}
        except Exception as e:
            print(f"[ML Backend] 下载图像失败: {e}")

    # 检查是否有 session_id（已在系统中的分割结果）
    session_id = task_data.get("session_id")
    if session_id:
        return {"type": "session", "session_id": session_id}

    return None


def _parse_label_config(label_config: Optional[str]) -> List[str]:
    """解析 Label Studio 标注配置，提取需要的标注类型"""
    if not label_config:
        return ["brushlabels", "textarea"]  # 默认支持分割和问答

    types = []
    label_config_lower = label_config.lower()

    if "brushlabels" in label_config_lower or "brush" in label_config_lower:
        types.append("brushlabels")
    if "textarea" in label_config_lower or "text" in label_config_lower:
        types.append("textarea")
    if "choices" in label_config_lower:
        types.append("choices")
    if "labels" in label_config_lower:
        types.append("labels")

    return types if types else ["brushlabels", "textarea"]


async def _generate_segmentation_predictions(
    image_data: Dict[str, Any],
    task_data: Dict[str, Any],
    params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    生成分割预标注结果

    返回 Label Studio BrushLabels 格式:
    {
        "type": "brushlabels",
        "value": {
            "format": "rle",
            "rle": [...],
            "brushlabels": ["Liver"]
        },
        "from_name": "brush",
        "to_name": "image",
        "original_width": 512,
        "original_height": 512
    }
    """
    results = []

    # 获取分割任务类型
    task_type = params.get("task", "total")

    try:
        # 根据数据类型处理
        if image_data["type"] == "session":
            # 使用已有的分割结果
            session_id = image_data["session_id"]
            seg_results = await _load_session_segmentation(session_id)
        else:
            # 执行新的分割
            seg_results = await _run_segmentation(image_data["data"], task_type)

        if not seg_results:
            return results

        # 转换为 Label Studio BrushLabels 格式
        for organ_name, mask_data in seg_results.items():
            mask = mask_data["mask"]
            original_height, original_width = mask.shape[:2]

            # RLE 编码
            rle = rle_encode(mask)

            # 获取器官中文名和颜色
            organ_cn = get_organ_cn(organ_name)
            color = ORGAN_COLORS.get(organ_name, [128, 128, 128])

            results.append({
                "type": "brushlabels",
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [organ_cn]
                },
                "from_name": "brush",
                "to_name": "image",
                "original_width": original_width,
                "original_height": original_height,
                "score": 0.95
            })

    except Exception as e:
        print(f"[ML Backend] 分割预标注失败: {e}")

    return results


async def _run_segmentation(
    image_bytes: bytes,
    task_type: str = "total"
) -> Dict[str, Dict[str, Any]]:
    """
    执行 TotalSegmentator 分割

    Returns:
        Dict[organ_name, {"mask": np.ndarray, "slice_idx": int}]
    """
    import shutil

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 保存输入图像
        input_path = tmpdir / "input.nii.gz"
        input_path.write_bytes(image_bytes)

        # 输出目录
        output_path = tmpdir / "segmentation"

        try:
            # 调用 TotalSegmentator
            await run_in_threadpool(
                totalsegmentator,
                input=str(input_path),
                output=str(output_path),
                task=task_type,
                fast=True,  # 使用快速模式
                quiet=True,
            )

            # 读取分割结果
            if output_path.exists():
                # 检查是否是多器官输出目录
                if output_path.is_dir():
                    for seg_file in output_path.glob("*.nii.gz"):
                        organ_name = seg_file.stem.replace(".nii", "")
                        seg_nii = nib.load(str(seg_file))
                        seg_data = seg_nii.get_fdata()

                        # 取中间切片
                        mid_slice = seg_data.shape[2] // 2
                        mask = (seg_data[:, :, mid_slice] > 0).astype(np.uint8)

                        if np.any(mask):
                            results[organ_name] = {
                                "mask": mask,
                                "slice_idx": mid_slice
                            }
                else:
                    # 单文件输出
                    seg_nii = nib.load(str(output_path))
                    seg_data = seg_nii.get_fdata()

                    # 解析多标签分割
                    unique_labels = np.unique(seg_data)
                    class_map_task = class_map.get(task_type, class_map.get("total", {}))

                    for label_idx in unique_labels:
                        if label_idx == 0:
                            continue
                        organ_name = class_map_task.get(int(label_idx), f"organ_{int(label_idx)}")
                        mid_slice = seg_data.shape[2] // 2
                        mask = (seg_data[:, :, mid_slice] == label_idx).astype(np.uint8)

                        if np.any(mask):
                            results[organ_name] = {
                                "mask": mask,
                                "slice_idx": mid_slice
                            }

        except Exception as e:
            print(f"[ML Backend] TotalSegmentator 执行失败: {e}")

    return results


async def _load_session_segmentation(session_id: str) -> Dict[str, Dict[str, Any]]:
    """从已有 session 加载分割结果"""
    results = {}

    try:
        # 直接从数据目录读取
        from api.config import DATA_DIR
        session_dir = DATA_DIR / session_id

        if not session_dir.exists():
            return results

        # 查找分割文件
        seg_path = session_dir / "segmentation.nii.gz"
        if not seg_path.exists():
            return results

        # 读取分割文件
        seg_nii = nib.load(str(seg_path))
        seg_data = seg_nii.get_fdata()

        # 尝试读取 session 元数据获取 task 类型
        metadata_path = session_dir / "metadata.json"
        task_type = "total"
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as fh:
                    metadata = json.load(fh)
                task_type = metadata.get("task", "total")
            except Exception:
                pass

        class_map_task = class_map.get(task_type, class_map.get("total", {}))

        unique_labels = np.unique(seg_data)
        mid_slice = seg_data.shape[2] // 2

        for label_idx in unique_labels:
            if label_idx == 0:
                continue
            organ_name = class_map_task.get(int(label_idx), f"organ_{int(label_idx)}")
            mask = (seg_data[:, :, mid_slice] == label_idx).astype(np.uint8)

            if np.any(mask):
                results[organ_name] = {
                    "mask": mask,
                    "slice_idx": mid_slice
                }

    except Exception as e:
        print(f"[ML Backend] 加载 session 分割失败: {e}")

    return results


async def _generate_vqa_predictions(
    image_data: Dict[str, Any],
    task_data: Dict[str, Any],
    params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    生成 VQA 问答预标注结果

    返回 Label Studio TextArea 格式:
    {
        "type": "textarea",
        "value": {"text": ["这是一张腹部CT图像，可见肝脏、脾脏等器官..."]},
        "from_name": "answer",
        "to_name": "image"
    }
    """
    results = []

    try:
        # 获取器官信息
        organs = task_data.get("organs", [])
        slice_idx = task_data.get("slice_idx", 0)

        # 如果有 session_id，从 session 获取器官信息
        if image_data["type"] == "session":
            session_id = image_data["session_id"]
            organs = await _get_session_organs(session_id, slice_idx)

        if not organs:
            # 没有器官信息，返回通用描述
            results.append({
                "type": "textarea",
                "value": {"text": ["请描述这张医学影像的内容。"]},
                "from_name": "answer",
                "to_name": "image",
                "score": 0.5
            })
            return results

        # 生成问答对
        qa_pairs = generate_qa_pairs(
            organs=organs,
            slice_idx=slice_idx,
            total_slices=task_data.get("total_slices", 100),
            use_llm=params.get("use_llm", False),
            image_base64=params.get("image_base64")
        )

        # 转换为 Label Studio 格式
        for i, qa in enumerate(qa_pairs):
            # 问题作为标签
            results.append({
                "type": "labels",
                "value": {"labels": [qa.get("question_type", "general")]},
                "from_name": "question_type",
                "to_name": "image",
                "score": 0.9
            })

            # 答案作为文本
            results.append({
                "type": "textarea",
                "value": {"text": [qa.get("answer", "")]},
                "from_name": "answer",
                "to_name": "image",
                "score": 0.9
            })

    except Exception as e:
        print(f"[ML Backend] VQA 预标注失败: {e}")

    return results


async def _get_session_organs(session_id: str, slice_idx: int) -> List[str]:
    """从 session 获取指定切片的器官列表"""
    try:
        # 直接从数据目录读取（避免依赖 main.py 全局变量）
        project_root = Path(__file__).resolve().parent.parent
        session_dir = project_root / "api_data" / session_id

        if not session_dir.exists():
            return []

        # 尝试从 metadata 获取器官信息
        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as fh:
                    metadata = json.load(fh)
                organs = metadata.get("organs", [])
                if organs:
                    return organs
            except Exception:
                pass

        # 从分割文件读取
        seg_path = session_dir / "segmentation.nii.gz"
        if seg_path.exists():
            seg_nii = nib.load(str(seg_path))
            seg_data = seg_nii.get_fdata()

            # 获取 task 类型
            task_type = "total"
            if metadata_path.exists():
                try:
                    with open(metadata_path, encoding="utf-8") as fh:
                        metadata = json.load(fh)
                    task_type = metadata.get("task", "total")
                except Exception:
                    pass

            class_map_task = class_map.get(task_type, class_map.get("total", {}))

            slice_data = seg_data[:, :, slice_idx] if slice_idx < seg_data.shape[2] else seg_data[:, :, 0]
            unique_labels = np.unique(slice_data)

            organs = []
            for label_idx in unique_labels:
                if label_idx == 0:
                    continue
                organ_name = class_map_task.get(int(label_idx), f"organ_{int(label_idx)}")
                organs.append(organ_name)

            return organs

    except Exception as e:
        print(f"[ML Backend] 获取 session 器官失败: {e}")

    return []


@router.post("/webhook")
async def webhook(request: WebhookRequest):
    """
    Webhook 接口 - 接收 Label Studio 事件通知

    支持的事件：
    - ANNOTATION_CREATED: 新标注创建
    - ANNOTATION_UPDATED: 标注更新
    - PROJECT_CREATED: 项目创建
    """
    action = request.action

    print(f"[ML Backend] 收到 Webhook: {action}")

    if action == "ANNOTATION_CREATED":
        # 处理新标注 - 可用于触发模型微调
        annotation = request.annotation
        task = request.task

        # TODO: 保存标注结果用于后续训练
        print(f"[ML Backend] 新标注: task_id={task.get('id') if task else 'unknown'}")

    elif action == "ANNOTATION_UPDATED":
        # 标注更新
        annotation = request.annotation
        print(f"[ML Backend] 标注更新: annotation_id={annotation.get('id') if annotation else 'unknown'}")

    elif action == "PROJECT_CREATED":
        # 新项目创建
        project = request.project
        print(f"[ML Backend] 新项目: project_id={project.get('id') if project else 'unknown'}")

    return {"status": "ok"}


# ============ 辅助接口 ============
@router.get("/versions")
async def get_versions():
    """获取支持的模型版本"""
    return {
        "versions": [ML_BACKEND_VERSION],
        "current": ML_BACKEND_VERSION
    }


@router.get("/tasks")
async def get_supported_tasks():
    """获取支持的分割任务类型"""
    return {
        "tasks": SUPPORTED_TASKS,
        "descriptions": {
            "total": "全身 117 个解剖结构",
            "lung_vessels": "肺血管分割",
            "body": "身体组成分析",
            "cerebral_bleed": "脑出血检测",
            "hip_implant": "髋关节植入物",
            "coronary_arteries": "冠状动脉",
            "pleural_pericard_effusion": "胸腔积液/心包积液",
            "liver_vessels": "肝血管分割"
        }
    }

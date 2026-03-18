"""
Label Studio 集成服务
实现与 Label Studio 的自动化集成，包括：
- 切片图像上传到 MinIO
- 自动创建 Label Studio 项目
- 自动导入标注任务
- S3 存储集成配置
"""
import os
import io
import json
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import nibabel as nib

# Label Studio SDK (支持 v1.x 和 v2.x)
LABEL_STUDIO_SDK_AVAILABLE = False
LABEL_STUDIO_SDK_V2 = False
LSClient = None

try:
    # 尝试 v2.x API
    from label_studio_sdk import LabelStudio as LSClient
    LABEL_STUDIO_SDK_AVAILABLE = True
    LABEL_STUDIO_SDK_V2 = True
except ImportError:
    try:
        # 回退到 v1.x API
        from label_studio_sdk import Client as LSClient
        LABEL_STUDIO_SDK_AVAILABLE = True
    except ImportError:
        pass

from api.storage import (
    get_minio_client, upload_bytes, MINIO_BUCKET,
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE
)

# 延迟导入配置模块（避免循环导入）
def _get_ls_config():
    """获取 Label Studio 配置"""
    try:
        from api.config import get_labelstudio_config
        return get_labelstudio_config()
    except ImportError:
        # 回退到环境变量
        class FallbackConfig:
            url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
            api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
            minio_public_url = os.getenv("MINIO_PUBLIC_URL", f"http://{MINIO_ENDPOINT}")
            default_project_name = "TotalSegmentator-VQA"
        return FallbackConfig()


# MinIO 公开访问 URL (用于 Label Studio 访问图片) - 保留作为后备
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", f"http://{MINIO_ENDPOINT}")


def extract_polygon_from_mask(
    mask: np.ndarray,
    simplify_tolerance: float = 3.0,
    min_area: int = 500,
    max_points: int = 50
) -> List[List[float]]:
    """
    从二值 mask 中提取多边形轮廓坐标

    只提取足够大的真实器官分割，过滤噪点

    Args:
        mask: 二值掩码 (H, W)
        simplify_tolerance: 简化容差（越大越简化，减少点数）
        min_area: 最小面积阈值（像素），小于此值的分割会被忽略
        max_points: 多边形最大顶点数

    Returns:
        多边形坐标列表 [[x1,y1], [x2,y2], ...] (像素坐标)
        如果分割太小或无效，返回空列表
    """
    # 计算 mask 面积
    area = np.sum(mask)
    if area < min_area:
        return []  # 面积太小，忽略噪点

    try:
        from skimage import measure

        # 查找轮廓（返回所有轮廓，取最大的一个）
        contours = measure.find_contours(mask.astype(float), 0.5)

        if not contours:
            return []

        # 取最大的轮廓（点数最多的）
        largest_contour = max(contours, key=lambda c: len(c))

        # 如果轮廓点数太少，不是有效多边形
        if len(largest_contour) < 4:
            return []

        # 简化轮廓（减少点数，提高性能）
        from skimage.measure import approximate_polygon
        simplified = approximate_polygon(largest_contour, tolerance=simplify_tolerance)

        # 如果简化后点数仍然太多，进一步简化
        while len(simplified) > max_points and simplify_tolerance < 20:
            simplify_tolerance += 1
            simplified = approximate_polygon(largest_contour, tolerance=simplify_tolerance)

        # 简化后点数太少，不是有效多边形
        if len(simplified) < 4:
            return []

        # 转换为 [x, y] 格式（注意：skimage 返回的是 [row, col] 即 [y, x]）
        polygon = [[float(point[1]), float(point[0])] for point in simplified]

        return polygon
    except ImportError:
        # 如果 skimage 不可用，不生成预标注
        return []
    except Exception:
        return []


def convert_polygon_to_labelstudio(
    polygon: List[List[float]],
    image_width: int,
    image_height: int,
    organ_name: str,
    organ_cn_name: str,
    color: str = "#FF0000"
) -> Dict[str, Any]:
    """
    将多边形坐标转换为 Label Studio 预标注格式

    Label Studio 使用百分比坐标 (0-100)
    """
    import uuid

    if not polygon or len(polygon) < 3:
        return None

    # 转换为百分比坐标
    points = []
    for x, y in polygon:
        # 确保坐标在有效范围内
        x_pct = max(0, min(100, (x / image_width) * 100))
        y_pct = max(0, min(100, (y / image_height) * 100))
        points.append([x_pct, y_pct])

    return {
        "id": str(uuid.uuid4())[:8],
        "type": "polygonlabels",
        "value": {
            "points": points,
            "polygonlabels": [organ_cn_name]
        },
        "from_name": "organs",
        "to_name": "image",  # 绑定到预标注图像
        "original_width": image_width,
        "original_height": image_height
    }


class LabelStudioService:
    """Label Studio 集成服务"""

    def __init__(self):
        self.client: Optional[LSClient] = None
        self._connected = False
        self._url = None
        self._api_key = None

    def connect(self) -> bool:
        """连接到 Label Studio"""
        if not LABEL_STUDIO_SDK_AVAILABLE:
            print("⚠️ Label Studio SDK 未安装，请运行: pip install label-studio-sdk")
            return False

        # 从配置获取连接信息
        ls_config = _get_ls_config()
        self._url = ls_config.url
        self._api_key = ls_config.api_key

        if not self._api_key:
            print("⚠️ LABEL_STUDIO_API_KEY 未配置")
            return False

        try:
            if LABEL_STUDIO_SDK_V2:
                # v2.x API
                self.client = LSClient(base_url=self._url, api_key=self._api_key)
                # 验证连接
                list(self.client.projects.list())
            else:
                # v1.x API
                self.client = LSClient(url=self._url, api_key=self._api_key)
                self.client.get_projects()

            self._connected = True
            print(f"✓ 已连接到 Label Studio: {self._url}")
            return True
        except Exception as e:
            print(f"✗ Label Studio 连接失败: {e}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.client is not None

    def _get_projects(self):
        """获取项目列表（兼容 v1/v2）"""
        if LABEL_STUDIO_SDK_V2:
            return list(self.client.projects.list())
        else:
            return self.client.get_projects()

    def _create_project(self, title: str, label_config: str):
        """创建项目（兼容 v1/v2）"""
        if LABEL_STUDIO_SDK_V2:
            return self.client.projects.create(title=title, label_config=label_config)
        else:
            return self.client.create_project(title=title, label_config=label_config)

    def update_project_label_config(self, project_id: int, label_config: str) -> bool:
        """更新项目的 label_config（企业交付：确保 UI 始终与最新任务字段一致）

        说明：SDK v1/v2 在项目更新 API 上差异较大，这里统一走 REST API。
        """
        if not self.is_connected or not self._url or not self._api_key:
            return False

        base_url = str(self._url).rstrip("/")
        api_url = f"{base_url}/api/projects/{project_id}"
        payload = {"label_config": label_config}
        headers = {
            # Label Studio 采用 Token 认证
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
        }

        # 优先使用 requests（本项目依赖中已存在）；若不可用则降级 urllib
        try:
            import requests  # type: ignore

            resp = requests.patch(api_url, headers=headers, json=payload, timeout=20)
            if 200 <= resp.status_code < 300:
                print(f"✓ 已更新项目 label_config (ID: {project_id})")
                return True
            print(f"⚠️ 更新项目 label_config 失败: status={resp.status_code}, body={resp.text[:200]}")
            return False
        except ImportError:
            try:
                import urllib.request

                req = urllib.request.Request(
                    api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="PATCH",
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    if 200 <= resp.status < 300:
                        print(f"✓ 已更新项目 label_config (ID: {project_id})")
                        return True
                print(f"⚠️ 更新项目 label_config 失败: status={getattr(resp, 'status', None)}")
                return False
            except Exception as e:
                print(f"⚠️ 更新项目 label_config 异常: {e}")
                return False

    def _import_tasks(self, project_id: int, tasks: List[Dict]):
        """导入任务（兼容 v1/v2）"""
        if LABEL_STUDIO_SDK_V2:
            return self.client.projects.import_tasks(id=project_id, request=tasks)
        else:
            project = self.client.get_project(project_id)
            return project.import_tasks(tasks)

    def get_or_create_project(self, project_name: str, label_config: str) -> Optional[int]:
        """获取或创建 Label Studio 项目"""
        if not self.is_connected:
            if not self.connect():
                return None

        try:
            # 查找现有项目
            projects = self._get_projects()
            for project in projects:
                if project.title == project_name:
                    print(f"✓ 找到现有项目: {project_name} (ID: {project.id})")
                    # 企业交付：忽略旧数据，直接把项目配置升级到最新（例如新增 image_ref）
                    self.update_project_label_config(project_id=project.id, label_config=label_config)
                    return project.id

            # 创建新项目
            project = self._create_project(title=project_name, label_config=label_config)
            print(f"✓ 创建新项目: {project_name} (ID: {project.id})")
            return project.id
        except Exception as e:
            print(f"✗ 创建项目失败: {e}")
            return None

    def configure_s3_storage(self, project_id: int) -> bool:
        """为项目配置 S3 (MinIO) 存储

        注意：由于我们使用公开 URL 直接访问图片，S3 存储配置是可选的。
        如果配置失败不影响基本功能。
        """
        if not self.is_connected:
            return False

        try:
            # SDK v2 和 v1 的 API 不同
            if LABEL_STUDIO_SDK_V2:
                # v2 SDK: 使用 REST API 方式配置
                # 由于 v2 SDK 的 storage API 较复杂，暂时跳过
                print("ℹ S3 存储配置跳过 (使用公开 URL 访问图片)")
                return True
            else:
                # v1 SDK
                project = self.client.get_project(project_id)

                # 配置 S3 源存储
                storage_config = {
                    "bucket": MINIO_BUCKET,
                    "prefix": "labelstudio/",
                    "use_blob_urls": True,
                    "aws_access_key_id": MINIO_ACCESS_KEY,
                    "aws_secret_access_key": MINIO_SECRET_KEY,
                    "s3_endpoint": MINIO_PUBLIC_URL,
                    "region_name": "us-east-1",
                }

                # 检查是否已有存储配置
                existing_storages = project.get_source_storages()
                for storage in existing_storages:
                    if storage.get("bucket") == MINIO_BUCKET:
                        print("✓ S3 存储已配置")
                        return True

                # 创建新的 S3 存储
                project.connect_s3_import_storage(**storage_config)
                print("✓ S3 存储配置成功")
                return True
        except Exception as e:
            # S3 配置失败不影响功能，因为我们使用公开 URL
            print(f"ℹ S3 存储配置跳过: {e} (使用公开 URL 访问图片)")
            return True  # 返回 True 以继续流程

    def import_tasks(self, project_id: int, tasks: List[Dict[str, Any]]) -> int:
        """导入任务到 Label Studio 项目"""
        if not self.is_connected:
            return 0

        try:
            self._import_tasks(project_id, tasks)
            print(f"✓ 导入 {len(tasks)} 个任务")
            return len(tasks)
        except Exception as e:
            print(f"✗ 导入任务失败: {e}")
            return 0


# ============== RLE 编码函数 (Label Studio Brush 格式) ==============

def _bits2byte(arr_str: str, n: int = 8) -> List[int]:
    """将位字符串转换为字节列表"""
    rle = []
    numbers = [arr_str[i:i + n] for i in range(0, len(arr_str), n)]
    for i in numbers:
        rle.append(int(i, 2))
    return rle


def _base_rle_encode(inarray: np.ndarray):
    """基础 RLE 编码，返回 (长度, 位置, 值)"""
    ia = np.asarray(inarray)
    n = len(ia)
    if n == 0:
        return None, None, None
    y = ia[1:] != ia[:-1]
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return z, p, ia[i]


def _encode_rle(arr: np.ndarray, wordsize: int = 8, rle_sizes: List[int] = None) -> List[int]:
    """将 1D 数组编码为 Label Studio RLE 格式"""
    if rle_sizes is None:
        rle_sizes = [3, 4, 8, 16]

    num = len(arr)
    numbits = f'{num:032b}'
    wordsizebits = f'{wordsize - 1:05b}'
    rle_bits = ''.join([f'{x - 1:04b}' for x in rle_sizes])
    base_str = numbits + wordsizebits + rle_bits

    out_str = ''
    result = _base_rle_encode(arr)
    if result[0] is None:
        return []

    for length_reeks, p, value in zip(*result):
        if length_reeks == 1:
            out_str += '0'
            out_str += '00'
            out_str += '000'
            out_str += f'{value:08b}'
        elif length_reeks > 1:
            if length_reeks <= 8:
                out_str += '1'
                out_str += '00'
                out_str += f'{length_reeks - 1:03b}'
                out_str += f'{value:08b}'
            elif 8 < length_reeks <= 16:
                out_str += '1'
                out_str += '01'
                out_str += f'{length_reeks - 1:04b}'
                out_str += f'{value:08b}'
            elif 16 < length_reeks <= 256:
                out_str += '1'
                out_str += '10'
                out_str += f'{length_reeks - 1:08b}'
                out_str += f'{value:08b}'
            else:
                length_temp = length_reeks
                while length_temp > 2**16:
                    out_str += '1'
                    out_str += '11'
                    out_str += f'{2 ** 16 - 1:016b}'
                    out_str += f'{value:08b}'
                    length_temp -= 2**16
                out_str += '1'
                out_str += '11'
                out_str += f'{length_temp - 1:016b}'
                out_str += f'{value:08b}'

    nzfill = 8 - len(base_str + out_str) % 8
    if nzfill == 8:
        nzfill = 0
    total_str = base_str + out_str + nzfill * '0'

    return _bits2byte(total_str)


def mask2rle(mask: np.ndarray) -> List[int]:
    """将 2D 掩码转换为 Label Studio RLE 格式

    Args:
        mask: 2D numpy 数组 (uint8)，非零值表示分割区域

    Returns:
        RLE 编码的整数列表
    """
    assert len(mask.shape) == 2, 'mask must be 2D np.array'
    # 转换为二值掩码：非零 -> 255，零 -> 0
    binary_mask = (mask > 0).astype(np.uint8) * 255
    array = binary_mask.ravel()
    array = np.repeat(array, 4)  # Label Studio 需要 4 通道 (RGBA)
    rle = _encode_rle(array)
    return rle


# ============== 参考图生成 (image_ref: 分割叠加 + 中文名称) ==============

def _load_cjk_font(font_size: int = 16) -> ImageFont.ImageFont:
    """尽可能加载可用的中文字体（macOS 开发 + Docker 生产）"""
    candidates = [
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/PingFangSC.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Debian/Ubuntu (fonts-noto-cjk)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_text_with_bg(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont):
    """画带背景的文字，保证在复杂底图上可读"""
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        pad = 2
        bg = (0, 0, 0, 180)
        fg = (255, 255, 255, 255)
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=bg)
        draw.text((x, y), text, font=font, fill=fg)
    except Exception:
        # 兼容旧 Pillow：无 textbbox
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def _create_reference_image(
    ct_flipped_gray: np.ndarray,
    seg_slice: np.ndarray,
    slice_organs: List[str],
    organ_to_idx: Dict[str, int],
    get_organ_cn_name,
    organ_colors: Dict[str, List[int]],
    overlay_opacity: float = 1.0,
    draw_labels: bool = True,
) -> Image.Image:
    """生成参考图：CT + 分割叠加 + 中文名称"""
    # 将 overlay_opacity 映射到更适合“参考图”的可视透明度区间
    alpha = max(0.15, min(0.60, float(overlay_opacity) * 0.35))

    h, w = ct_flipped_gray.shape
    base = np.stack([ct_flipped_gray] * 3, axis=-1).astype(np.float32)
    out = base.copy()

    # 叠加分割颜色
    for organ_name in slice_organs:
        organ_idx = organ_to_idx.get(organ_name)
        if organ_idx is None:
            continue
        mask = (seg_slice == organ_idx)
        if not np.any(mask):
            continue
        mask = np.flipud(mask)  # 与 ct_flipped 对齐
        color = organ_colors.get(organ_name, [255, 128, 0])
        out[mask] = out[mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha

    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")

    # 画中文名称
    if draw_labels and slice_organs:
        draw = ImageDraw.Draw(img)
        font = _load_cjk_font(font_size=max(12, int(min(h, w) * 0.03)))

        used_boxes: List[Tuple[int, int, int, int]] = []
        for organ_name in slice_organs:
            organ_idx = organ_to_idx.get(organ_name)
            if organ_idx is None:
                continue
            mask = (seg_slice == organ_idx)
            if not np.any(mask):
                continue
            mask = np.flipud(mask)

            coords = np.argwhere(mask)
            if coords.shape[0] < 30:
                continue
            cy, cx = coords.mean(axis=0)
            x0, y0 = int(cx), int(cy)
            text = str(get_organ_cn_name(organ_name))

            # 简单避让：最多尝试 6 次纵向偏移
            placed = False
            for dy in [0, -12, 12, -24, 24, 36]:
                x = max(2, min(w - 2, x0))
                y = max(2, min(h - 2, y0 + dy))
                try:
                    bbox = draw.textbbox((x, y), text, font=font)
                    # bbox: (l,t,r,b)
                    l, t, r, b = bbox
                    l = max(0, l - 2)
                    t = max(0, t - 2)
                    r = min(w, r + 2)
                    b = min(h, b + 2)
                    overlap = any(not (r < ul or l > ur or b < ut or t > ub) for (ul, ut, ur, ub) in used_boxes)
                    if overlap:
                        continue
                    used_boxes.append((l, t, r, b))
                    _draw_text_with_bg(draw, (x, y), text, font)
                    placed = True
                    break
                except Exception:
                    # fallback：不做避让
                    _draw_text_with_bg(draw, (x, y), text, font)
                    placed = True
                    break
            if not placed:
                # 最后兜底：仍然画在中心点
                _draw_text_with_bg(draw, (x0, y0), text, font)

    return img


# ============== 切片图像生成 ==============

def generate_slice_images(
    session_id: str,
    ct_data: np.ndarray,
    seg_data: np.ndarray,
    organs: List[str],
    axis: str = "axial",
    slice_step: int = 10,
    window_center: int = 40,
    window_width: int = 400,
    overlay_opacity: float = 1.0,  # 与主界面一致
    draw_labels: bool = True,
    task: str = "total"
) -> List[Dict[str, Any]]:
    """
    生成切片图像并上传到 MinIO（方案B：原始 CT + Brush 预标注）

    - 上传原始 CT 图像（不带分割叠加）
    - 分割结果作为 Brush 预标注（可编辑）

    返回切片信息列表，包含 MinIO 路径和 Brush 预标注数据
    """
    from api.main import get_organ_cn_name, ORGAN_COLORS  # 避免循环导入
    from totalsegmentator.map_to_binary import class_map

    # 从配置获取 MinIO 公开 URL
    ls_config = _get_ls_config()
    minio_public_url = ls_config.minio_public_url or MINIO_PUBLIC_URL

    slices_info = []

    # 确定切片维度
    axis_map = {"axial": 2, "sagittal": 0, "coronal": 1}
    if axis not in axis_map:
        raise ValueError(f"Invalid axis: {axis}. Expected one of {list(axis_map.keys())}.")
    if slice_step <= 0:
        raise ValueError("slice_step must be greater than 0.")
    if window_width <= 0:
        raise ValueError("window_width must be greater than 0.")
    axis_idx = axis_map[axis]
    num_slices = ct_data.shape[axis_idx]

    # 获取当前任务的 class_map
    class_map_task = class_map.get(task, class_map.get("total", {}))
    # 创建反向映射：器官名 -> 标签索引
    organ_to_idx = {v: k for k, v in class_map_task.items()}

    # 窗口化处理
    def apply_window(data: np.ndarray) -> np.ndarray:
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        windowed = np.clip(data, min_val, max_val)
        normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return normalized

    slice_indices = range(0, num_slices, slice_step)

    for slice_idx in slice_indices:
        try:
            # 提取切片
            if axis_idx == 0:  # sagittal
                ct_slice = ct_data[slice_idx, :, :]
                seg_slice = seg_data[slice_idx, :, :]
            elif axis_idx == 1:  # coronal
                ct_slice = ct_data[:, slice_idx, :]
                seg_slice = seg_data[:, slice_idx, :]
            else:  # axial
                ct_slice = ct_data[:, :, slice_idx]
                seg_slice = seg_data[:, :, slice_idx]

            # 转置使图像横向显示（与主界面一致）
            ct_slice = ct_slice.T
            seg_slice = seg_slice.T

            # 应用窗口
            ct_windowed = apply_window(ct_slice)

            # 记录当前切片中实际出现的器官
            slice_organs = []

            # 收集 Brush 预标注数据（RLE 格式，可编辑）
            brush_predictions = []

            # 图像尺寸
            img_height, img_width = seg_slice.shape

            for organ_name in organs:
                # 获取器官的标签索引
                organ_idx = organ_to_idx.get(organ_name)
                if organ_idx is None:
                    continue

                mask = seg_slice == organ_idx
                if not np.any(mask):
                    continue

                # 记录该器官在当前切片中存在
                slice_organs.append(organ_name)

                # 获取器官中文名
                cn_name = get_organ_cn_name(organ_name)

                # ✅ 生成 Brush 预标注（RLE 格式，用于编辑）
                # 翻转 mask 以匹配图像翻转
                flipped_mask = np.flipud(mask).astype(np.uint8)
                rle = mask2rle(flipped_mask)

                if rle:
                    import uuid
                    brush_predictions.append({
                        "id": str(uuid.uuid4())[:8],
                        "type": "brushlabels",
                        "value": {
                            "format": "rle",
                            "rle": rle,
                            "brushlabels": [cn_name]
                        },
                        "origin": "prediction",
                        "to_name": "image",
                        "from_name": "brush",
                        "image_rotation": 0,
                        "original_width": img_width,
                        "original_height": img_height
                    })

            print(f"  切片 {slice_idx}: 生成 {len(brush_predictions)} 个器官的 Brush 预标注")

            # ✅ 方案B：只上传原始 CT 图像（不带分割叠加）
            ct_flipped = np.flipud(ct_windowed)
            ct_img = Image.fromarray(ct_flipped).convert('RGB')

            # 高清图像：使用最低压缩
            img_buffer = io.BytesIO()
            ct_img.save(img_buffer, format='PNG', compress_level=1, optimize=False)
            img_bytes = img_buffer.getvalue()

            # 上传原始 CT 图像到 MinIO
            object_name = f"labelstudio/{session_id}/{axis}_{slice_idx:04d}.png"
            uploaded = upload_bytes(object_name, img_bytes, content_type="image/png")

            if uploaded:
                # 生成并上传参考图（分割叠加 + 中文名称）
                ref_object_name = f"labelstudio/{session_id}/{axis}_{slice_idx:04d}_ref.png"
                public_url_ref = f"{minio_public_url}/{MINIO_BUCKET}/{ref_object_name}"
                url_ref = f"s3://{MINIO_BUCKET}/{ref_object_name}"

                try:
                    ref_img = _create_reference_image(
                        ct_flipped_gray=ct_flipped,
                        seg_slice=seg_slice,
                        slice_organs=slice_organs,
                        organ_to_idx=organ_to_idx,
                        get_organ_cn_name=get_organ_cn_name,
                        organ_colors=ORGAN_COLORS,
                        overlay_opacity=overlay_opacity,
                        draw_labels=draw_labels,
                    )
                    ref_buf = io.BytesIO()
                    # 参考图同样保持高清，压缩较低
                    ref_img.save(ref_buf, format="PNG", compress_level=1, optimize=False)
                    ref_uploaded = upload_bytes(ref_object_name, ref_buf.getvalue(), content_type="image/png")
                    if not ref_uploaded:
                        # 不阻断流程：兜底使用原图
                        public_url_ref = f"{minio_public_url}/{MINIO_BUCKET}/{object_name}"
                        url_ref = f"s3://{MINIO_BUCKET}/{object_name}"
                except Exception as e:
                    print(f"⚠️ 生成/上传参考图失败(slice={slice_idx}): {e}")
                    public_url_ref = f"{minio_public_url}/{MINIO_BUCKET}/{object_name}"
                    url_ref = f"s3://{MINIO_BUCKET}/{object_name}"

                slices_info.append({
                    "slice_idx": slice_idx,
                    "axis": axis,
                    "object_name": object_name,
                    "url": f"s3://{MINIO_BUCKET}/{object_name}",
                    "public_url": f"{minio_public_url}/{MINIO_BUCKET}/{object_name}",
                    "object_name_ref": ref_object_name,
                    "url_ref": url_ref,
                    "public_url_ref": public_url_ref,
                    "organs": slice_organs,
                    # ✅ Brush 预标注数据（RLE 格式，用于编辑）
                    "brush_predictions": brush_predictions,
                    "image_width": ct_img.width,
                    "image_height": ct_img.height
                })
        except Exception as e:
            print(f"生成切片 {slice_idx} 失败: {e}")
            continue

    return slices_info


def create_labelstudio_tasks(
    session_id: str,
    slices_info: List[Dict[str, Any]],
    organs: List[str],
    filename: str,
    use_public_url: bool = True
) -> List[Dict[str, Any]]:
    """
    创建 Label Studio 任务格式的数据（包含可编辑的预标注）
    """
    from api.main import get_organ_cn_name  # 使用统一的中文器官名称

    tasks = []

    # 切面中文名称
    axis_names_cn = {
        "axial": "轴位",
        "sagittal": "矢状位",
        "coronal": "冠状位"
    }

    for slice_info in slices_info:
        # 选择图片 URL
        if use_public_url:
            image_url = slice_info["public_url"]
            image_ref_url = slice_info.get("public_url_ref") or image_url
        else:
            image_url = slice_info["url"]  # S3 URL
            image_ref_url = slice_info.get("url_ref") or image_url

        # 生成问题和 AI 回答（使用中文器官名称）
        slice_organs = slice_info.get("organs", organs[:5])
        organ_list = ", ".join([get_organ_cn_name(o) for o in slice_organs[:5]])

        axis_cn = axis_names_cn.get(slice_info['axis'], slice_info['axis'])
        question = "请描述这张CT切片图像中可见的器官结构。"
        ai_answer = f"在这张{axis_cn}（{slice_info['axis']}）切片中，可以观察到以下器官结构：{organ_list}。"

        task = {
            "data": {
                "image": image_url,
                "image_ref": image_ref_url,
                "question": question,
                "ai_answer": ai_answer,
                "session_id": session_id,
                "filename": filename,
                "axis": slice_info["axis"],
                "slice_idx": slice_info["slice_idx"],
                "organs_in_slice": slice_organs,
            },
            "predictions": [{
                "model_version": "TotalSegmentator-v2",
                "result": []
            }],
        }

        # ✅ 添加 Brush 预标注数据（RLE 格式，可编辑）
        brush_predictions = slice_info.get("brush_predictions", [])
        if brush_predictions:
            task["predictions"][0]["result"] = brush_predictions

        tasks.append(task)

    return tasks


def get_vqa_label_config(organs: List[str] = None) -> str:
    """获取 Brush 标注配置 - 动态生成，只包含用户选择的器官

    功能：
    - 显示原始 CT 图像
    - AI 分割结果作为 Brush 预标注（RLE 格式）
    - 支持使用画笔工具编辑分割区域

    预标注数据可以：
    - 使用画笔工具编辑边界
    - 使用橡皮擦删除错误分割
    - 新增分割区域

    Args:
        organs: 用户选择的器官英文名称列表，用于动态生成标签配置
    """
    from api.main import get_organ_cn_name, ORGAN_COLORS

    # 生成器官标签配置
    label_lines = []
    if organs:
        for organ_name in organs:
            cn_name = get_organ_cn_name(organ_name)
            # 获取器官颜色（RGB -> HEX）
            color = ORGAN_COLORS.get(organ_name, [255, 128, 0])
            color_hex = "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
            label_lines.append(f'    <Label value="{cn_name}" background="{color_hex}"/>')

    # 如果没有器官，添加默认标签
    if not label_lines:
        label_lines = [
            '    <Label value="未知器官" background="#808080"/>'
        ]

    labels_xml = "\n".join(label_lines)

    return f"""
<View>
  <Header value="医学影像器官分割标注 - 可编辑分割（企业生产版）" />

  <View style="display:flex; gap:16px; align-items:flex-start;">
    <View style="flex:1;">
      <Header value="原始 CT（可编辑）" />
      <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true" crosshair="true" />

      <Header value="器官分割标签（使用画笔工具编辑分割区域）" />
      <BrushLabels name="brush" toName="image">
{labels_xml}
      </BrushLabels>
    </View>

    <View style="flex:1;">
      <Header value="参考图（分割叠加 + 中文名称，仅供识别）" />
      <Image name="image_ref" value="$image_ref" zoom="true" zoomControl="true" rotateControl="true" />
    </View>
  </View>

  <Header value="问题" />
  <Text name="question" value="$question" />

  <Header value="AI 回答（参考）" />
  <Text name="ai_answer" value="$ai_answer" />

  <Header value="标注员答案" />
  <TextArea name="answer" toName="image"
    placeholder="请输入正确答案..."
    maxSubmissions="1" editable="true" />

  <Header value="答案质量评估" />
  <Choices name="quality" toName="image" choice="single">
    <Choice value="correct" alias="正确">AI 回答正确</Choice>
    <Choice value="partially_correct" alias="部分正确">AI 回答部分正确</Choice>
    <Choice value="incorrect" alias="错误">AI 回答错误</Choice>
  </Choices>
</View>
"""


async def sync_session_to_labelstudio(
    session_id: str,
    session_data: Dict[str, Any],
    project_name: str = "TotalSegmentator-VQA",
    slice_step: int = 10,
    axis: str = "axial",
    organs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    将会话数据同步到 Label Studio

    完整流程:
    1. 生成切片图像并上传到 MinIO
    2. 创建或获取 Label Studio 项目
    3. 配置 S3 存储
    4. 导入任务

    Args:
        organs: 要同步的器官列表，为空则使用 session 中的全部器官
    """
    result = {
        "success": False,
        "message": "",
        "project_id": None,
        "tasks_imported": 0,
        "slices_uploaded": 0
    }

    try:
        # 1. 加载数据
        session_dir = Path(session_data.get("session_dir", ""))
        output_path = session_dir / "segmentation.nii.gz"
        ct_path = session_data.get("converted_nifti") or session_data.get("input_path")

        if not output_path.exists():
            result["message"] = "分割结果文件不存在"
            return result

        seg_img = nib.load(str(output_path))
        seg_data = seg_img.get_fdata()

        if ct_path and Path(ct_path).exists():
            ct_img = nib.load(str(ct_path))
            ct_data = ct_img.get_fdata()
        else:
            ct_data = np.zeros_like(seg_data)

        # 使用传入的器官列表，如果没有则使用 session 中的全部器官
        session_all_organs = session_data.get("organs", [])
        final_organs = organs if organs else session_all_organs
        filename = session_data.get("filename", "unknown")
        task_type = session_data.get("task", "total")

        if organs:
            print(f"📋 同步用户选中的 {len(final_organs)} 个器官 (共 {len(session_all_organs)} 个)")
        else:
            print(f"📋 同步全部 {len(final_organs)} 个器官")

        # 2. 生成切片并上传到 MinIO
        print(f"正在生成切片图像 (axis={axis}, step={slice_step}, task={task_type})...")
        slices_info = generate_slice_images(
            session_id=session_id,
            ct_data=ct_data,
            seg_data=seg_data,
            organs=final_organs,
            axis=axis,
            slice_step=slice_step,
            task=task_type
        )
        result["slices_uploaded"] = len(slices_info)

        if not slices_info:
            result["message"] = "没有生成切片图像"
            return result

        # 3. 连接 Label Studio
        ls_service = LabelStudioService()
        if not ls_service.connect():
            result["message"] = "无法连接到 Label Studio，请检查配置"
            return result

        # 4. 创建或获取项目（动态生成配置，只包含用户选择的器官）
        label_config = get_vqa_label_config(organs=final_organs)
        project_id = ls_service.get_or_create_project(project_name, label_config)

        if not project_id:
            result["message"] = "创建 Label Studio 项目失败"
            return result

        result["project_id"] = project_id

        # 5. 配置 S3 存储
        ls_service.configure_s3_storage(project_id)

        # 6. 创建并导入任务
        tasks = create_labelstudio_tasks(
            session_id=session_id,
            slices_info=slices_info,
            organs=final_organs,
            filename=filename,
            use_public_url=True
        )

        imported = ls_service.import_tasks(project_id, tasks)
        result["tasks_imported"] = imported

        result["success"] = True
        result["message"] = f"成功同步到 Label Studio: 上传 {len(slices_info)} 张切片，导入 {imported} 个任务"

    except Exception as e:
        result["message"] = f"同步失败: {str(e)}"
        print(f"同步到 Label Studio 失败: {e}")

    return result


# 全局服务实例
_labelstudio_service: Optional[LabelStudioService] = None


def get_labelstudio_service() -> LabelStudioService:
    """获取 Label Studio 服务实例"""
    global _labelstudio_service
    if _labelstudio_service is None:
        _labelstudio_service = LabelStudioService()
    return _labelstudio_service

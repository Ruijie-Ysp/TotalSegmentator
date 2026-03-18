"""
Label Studio PNG 优化版本
仅在同步到 Label Studio 时使用高质量优化
不影响原有的分割界面
"""
import numpy as np
from PIL import Image
import io
from typing import Dict, Any


def apply_window_enhanced(data: np.ndarray, center: int, width: int) -> np.ndarray:
    """
    增强版窗口化（专用于 Label Studio）

    保留 float32 精度 + Gamma 校正
    """
    min_val = center - width / 2
    max_val = center + width / 2
    windowed = np.clip(data, min_val, max_val)

    # 保留 float32 精度（0-1 范围）
    windowed = ((windowed - min_val) / (max_val - min_val)).astype(np.float32)

    # Gamma 校正增强对比度
    gamma = 0.85
    windowed = np.power(windowed, gamma).astype(np.float32)

    return windowed


def extract_contour_precise(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    精确轮廓提取（专用于 Label Studio）

    使用 Marching Squares 算法
    """
    try:
        from skimage import measure
        contour_img = np.zeros_like(mask, dtype=np.uint8)

        contours = measure.find_contours(mask.astype(float), 0.5)

        for contour in contours:
            contour = contour.astype(int)
            contour[:, 0] = np.clip(contour[:, 0], 0, mask.shape[0] - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, mask.shape[1] - 1)

            for y, x in contour:
                y_min = max(0, y - thickness // 2)
                y_max = min(mask.shape[0], y + thickness // 2 + 1)
                x_min = max(0, x - thickness // 2)
                x_max = min(mask.shape[1], x + thickness // 2 + 1)
                contour_img[y_min:y_max, x_min:x_max] = 1

        return contour_img > 0
    except ImportError:
        # 回退到形态学方法
        from scipy import ndimage
        dilated = ndimage.binary_dilation(mask, iterations=thickness)
        eroded = ndimage.binary_erosion(mask, iterations=thickness)
        contour = dilated.astype(np.uint8) - eroded.astype(np.uint8)
        return contour > 0


def generate_optimized_slice_for_labelstudio(
    ct_slice: np.ndarray,
    seg_slice: np.ndarray,
    organs: list,
    organ_to_idx: Dict[str, int],
    organ_colors: Dict[str, list],
    window_center: int = 40,
    window_width: int = 400,
    overlay_opacity: float = 0.6
) -> bytes:
    """
    生成优化的切片图像（专用于 Label Studio）

    优化内容:
    1. 保留 float32 精度
    2. Gamma 校正增强对比度
    3. 增强器官颜色权重（70%）
    4. 精确轮廓算法
    5. PNG 最大压缩

    返回: PNG 图片字节流
    """
    # 1. 增强版窗口化
    windowed = apply_window_enhanced(ct_slice, window_center, window_width)

    # 2. 转换回 0-255 范围（用于显示）
    windowed_255 = (windowed * 255).astype(np.uint8)

    # 3. 创建 RGB 图像
    rgb_image = np.stack([windowed_255] * 3, axis=-1).astype(np.float32)

    # 4. 叠加器官分割（增强颜色权重）
    for organ_name in organs:
        organ_idx = organ_to_idx.get(organ_name)
        if organ_idx is None:
            continue

        mask = seg_slice == organ_idx
        if not np.any(mask):
            continue

        color = np.array(organ_colors.get(organ_name, [255, 255, 0]), dtype=np.float32)

        # 增强器官颜色权重（70%）
        enhanced_opacity = overlay_opacity
        for c in range(3):
            rgb_image[:, :, c] = np.where(
                mask,
                (1 - enhanced_opacity) * rgb_image[:, :, c] + enhanced_opacity * color[c],
                rgb_image[:, :, c]
            )

    # 5. 转换为 PNG
    rgb_image = np.flipud(rgb_image).astype(np.uint8)
    img = Image.fromarray(rgb_image)

    # 6. PNG 最大压缩
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", compress_level=9, optimize=True)

    return buffer.getvalue()

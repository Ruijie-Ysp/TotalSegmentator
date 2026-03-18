import gradio as gr
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
import torch

from PIL import Image
import shutil

import zipfile
import json
from datetime import datetime

def _safe_filename(name: str) -> str:
    # Keep it filesystem-friendly
    keep = []
    for ch in (name or ""):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "input"


def _collect_dicom_candidates(root: Path, max_files: int = 20000) -> tuple[list[Path], bool]:
    """
    Collect files under root that are likely DICOMs (best-effort).
    Returns (files, truncated).
    """
    files: list[Path] = []
    truncated = False
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Fast path: extension suggests DICOM
            if p.suffix.lower() in (".dcm", ".dicom"):
                files.append(p)
            else:
                # Heuristic: keep a small number of "unknown" files too, we'll validate later with pydicom
                # (some DICOMs have no extension)
                if len(files) < max_files:
                    files.append(p)
            if len(files) >= max_files:
                truncated = True
                break
    except Exception:
        pass
    return files, truncated


def _dicom_series_summary(dicom_root: Path, out_json_path: Path, max_files: int = 20000) -> dict:
    """
    Create a lightweight DICOM header summary for debugging conversion issues.
    Only reads metadata (stop_before_pixels=True).
    """
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
    if truncated:
        summary["warnings"].append("file_list_truncated")

    def get_tag(ds, name, default=None):
        try:
            v = getattr(ds, name, default)
            if v is None:
                return default
            # Convert pydicom types to JSON-friendly
            return str(v)
        except Exception:
            return default

    # Parse headers
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            # Minimal validation: has SOPClassUID or Modality
            sop = get_tag(ds, "SOPClassUID", None)
            modality = get_tag(ds, "Modality", None)
            if sop is None and modality is None:
                continue

            summary["file_count_parsed"] += 1

            study_uid = get_tag(ds, "StudyInstanceUID", "UNKNOWN_STUDY")
            series_uid = get_tag(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")
            series_desc = get_tag(ds, "SeriesDescription", None)
            manufacturer = get_tag(ds, "Manufacturer", None)

            # Timepoint-ish tags (best-effort)
            temporal_pos = get_tag(ds, "TemporalPositionIdentifier", None)
            acquisition_num = get_tag(ds, "AcquisitionNumber", None)
            trigger_time = get_tag(ds, "TriggerTime", None)

            instance_number_raw = getattr(ds, "InstanceNumber", None)
            instance_number = None
            try:
                if instance_number_raw is not None:
                    instance_number = int(instance_number_raw)
            except Exception:
                instance_number = None

            studies = summary["studies"]
            study = studies.setdefault(study_uid, {"series": {}, "notes": []})
            series = study["series"].setdefault(series_uid, {
                "modality": modality,
                "series_description": series_desc,
                "manufacturer": manufacturer,
                "files": 0,
                "timepoints": {},
            })

            series["files"] += 1

            tp_key = temporal_pos or acquisition_num or trigger_time or "NA"
            tp = series["timepoints"].setdefault(tp_key, {
                "files": 0,
                "instance_numbers": [],
                "has_instance_number": False,
                "example_file": str(f),
            })
            tp["files"] += 1
            if instance_number is not None:
                tp["has_instance_number"] = True
                tp["instance_numbers"].append(instance_number)
        except Exception:
            continue

    # Post-process per timepoint missing instance numbers
    candidates = []
    for study_uid, study in summary["studies"].items():
        for series_uid, series in study["series"].items():
            total_files = int(series.get("files", 0) or 0)
            tp_info = series.get("timepoints", {})
            timepoint_count = len(tp_info)

            # Compute missing ranges per timepoint when possible
            missing_total = 0
            known_total = 0
            for tp_key, tp in tp_info.items():
                inst = tp.get("instance_numbers", [])
                if inst:
                    inst_sorted = sorted(set(inst))
                    tp["instance_number_min"] = inst_sorted[0]
                    tp["instance_number_max"] = inst_sorted[-1]
                    tp["unique_instance_numbers"] = len(inst_sorted)
                    # Estimate missing inside min..max
                    expected = inst_sorted[-1] - inst_sorted[0] + 1
                    missing = max(0, expected - len(inst_sorted))
                    tp["missing_instance_numbers_estimate"] = missing
                    missing_total += missing
                    known_total += expected
                else:
                    tp["instance_number_min"] = None
                    tp["instance_number_max"] = None
                    tp["unique_instance_numbers"] = 0
                    tp["missing_instance_numbers_estimate"] = None

            series["timepoint_count"] = timepoint_count
            series["missing_instance_numbers_estimate_total"] = missing_total if known_total > 0 else None

            # Candidate scoring: prioritize large, single-timepoint, low-missing series
            missing_ratio = (missing_total / known_total) if known_total > 0 else 0.5
            score = total_files * 1.0
            if timepoint_count == 1:
                score *= 1.2
            score *= (1.0 - min(0.9, missing_ratio))

            candidates.append({
                "study_uid": study_uid,
                "series_uid": series_uid,
                "modality": series.get("modality"),
                "series_description": series.get("series_description"),
                "files": total_files,
                "timepoint_count": timepoint_count,
                "missing_instance_numbers_estimate_total": series.get("missing_instance_numbers_estimate_total"),
                "score": round(float(score), 4),
            })

    candidates_sorted = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    summary["best_candidates"] = candidates_sorted[:10]

    try:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        summary["warnings"].append(f"write_summary_failed: {e}")

    return summary


def _choose_best_dicom_subset(dicom_root: Path, max_files: int = 20000) -> dict | None:
    """
    Scan dicom_root recursively, group by Study/Series/Timepoint, and pick a best subset of files.
    Returns a dict with keys:
      - study_uid, series_uid, series_description, modality, manufacturer
      - timepoint_key (or 'NA')
      - files: list[Path]
      - reason: str
    Returns None if no DICOM files found / pydicom not available.
    """
    try:
        import pydicom
    except Exception:
        return None

    files, _truncated = _collect_dicom_candidates(dicom_root, max_files=max_files)

    def get_tag(ds, name, default=None):
        try:
            v = getattr(ds, name, default)
            if v is None:
                return default
            return str(v)
        except Exception:
            return default

    # studies[study_uid][series_uid] = {...}
    studies: dict = {}
    parsed_any = False

    for f in files:
        # Skip obvious macOS metadata
        try:
            if "__MACOSX" in f.parts:
                continue
        except Exception:
            pass
        try:
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

            # Timepoint-ish tags (best-effort)
            temporal_pos = get_tag(ds, "TemporalPositionIdentifier", None)
            acquisition_num = get_tag(ds, "AcquisitionNumber", None)
            trigger_time = get_tag(ds, "TriggerTime", None)
            tp_key = temporal_pos or acquisition_num or trigger_time or "NA"

            instance_number_raw = getattr(ds, "InstanceNumber", None)
            instance_number = None
            try:
                if instance_number_raw is not None:
                    instance_number = int(instance_number_raw)
            except Exception:
                instance_number = None

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
        # Prefer "std" over "lung" when all else equal (more general reconstruction)
        if "std" in d:
            bonus *= 1.10
        if "lung" in d:
            bonus *= 0.95
        if "localizer" in d or "scout" in d:
            bonus *= 0.10
        if "summary" in d:
            bonus *= 0.30
        return bonus

    candidates: list[dict] = []
    for study_uid, series_map in studies.items():
        for series_uid, series in series_map.items():
            tps = series.get("timepoints", {})
            if not tps:
                continue
            # Choose best timepoint within series
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
                # Score: prefer lots of files and low missing; penalize unknown instance numbers
                score = float(n_files)
                if missing_est is not None and (inst_max - inst_min + 1) > 0:
                    expected = (inst_max - inst_min + 1)
                    miss_ratio = missing_est / expected if expected else 0.0
                    score *= (1.0 - min(0.9, miss_ratio))
                else:
                    # No usable instance numbers; de-prioritize slightly
                    score *= 0.85
                tp_candidates.append((score, tp_key, missing_est, n_files))

            tp_candidates.sort(reverse=True, key=lambda x: x[0])
            best_tp_score, best_tp_key, best_tp_missing, best_tp_files = tp_candidates[0]

            desc = series.get("series_description", "")
            tp_count = len(tps)
            score = best_tp_score
            # Prefer single-timepoint series, but still allow multi-timepoint by choosing best timepoint
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
                "missing_instance_numbers_estimate": best_tp_missing,
                "files_count": best_tp_files,
            })

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x.get("score", 0))
    chosen = candidates[0]
    reason = (
        f"auto_selected: files={chosen.get('files_count')} "
        f"timepoint_count={chosen.get('timepoint_count')} "
        f"tp={chosen.get('timepoint_key')} "
        f"desc={chosen.get('series_description')!r}"
    )
    chosen["reason"] = reason
    return chosen


def _stage_dicom_files(files: list[Path], dst_dir: Path) -> tuple[Path, int]:
    """
    Put selected DICOM files into dst_dir using hardlinks if possible, else copy.
    Returns (dst_dir, count).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in files:
        try:
            # Keep original filename to minimize surprises
            dst = dst_dir / src.name
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except Exception:
                # Fallback to copy (still metadata-only; pixel data stays in file)
                shutil.copy2(src, dst)
            count += 1
        except Exception:
            continue
    return dst_dir, count


def process_nifti(input_file, task, fast, device, debug_mode):
    if input_file is None:
        return None, None, "请上传一个文件 (NIfTI, DICOM, ZIP 或 JPG/PNG)"
    
    # 手动管理临时目录，便于调试时保留
    tmp_dir = Path(tempfile.mkdtemp(prefix="totalseg_web_tmp_"))
    input_path = Path(input_file.name)
    nifti_input_path = tmp_dir / "input.nii.gz"
    output_path = tmp_dir / "segmentation.nii.gz"

    # Debug output dir (created only if needed)
    debug_out_dir = None
    debug_prefix = f"debug_{_safe_filename(input_path.stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        try:
            # 检查文件类型
            suffix = input_path.suffix.lower()
            is_2d = suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            is_dicom = suffix in [".dcm", ".dicom"]
            is_zip = suffix == ".zip"
            
            current_input = str(input_path)
            
            if is_2d:
                print(f"Converting 2D image {input_path.name} to NIfTI...")
                img_2d = Image.open(input_path).convert("L")
                data_2d = np.array(img_2d)
                data_3d = data_2d.T[:, :, np.newaxis]
                affine = np.eye(4)
                new_img = nib.Nifti1Image(data_3d.astype(np.float32), affine)
                nib.save(new_img, str(nifti_input_path))
                current_input = str(nifti_input_path)
            
            elif is_zip:
                print(f"Extracting ZIP file {input_path.name}...")
                zip_extract_dir = Path(tmp_dir) / "extracted_zip"
                zip_extract_dir.mkdir()
                with zipfile.ZipFile(input_path, 'r') as zip_ref:
                    zip_ref.extractall(zip_extract_dir)

                # 产品级：ZIP 里常混有多个 Series。这里自动选择最佳 Series/Timepoint，并只把该子集交给后续转换。
                chosen = _choose_best_dicom_subset(zip_extract_dir)
                if chosen and chosen.get("files"):
                    selected_dir = Path(tmp_dir) / "selected_dicom"
                    _, staged = _stage_dicom_files(chosen["files"], selected_dir)
                    current_input = str(selected_dir)
                    print(
                        "Selected DICOM subset -> "
                        f"desc={chosen.get('series_description')!r}, "
                        f"series_uid={chosen.get('series_uid')}, "
                        f"timepoint={chosen.get('timepoint_key')}, "
                        f"files={staged} | {chosen.get('reason')}"
                    )
                else:
                    # Fallback to previous behavior: pick directory with most .dcm files
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
                        print(f"No DICOM files found in ZIP, using root: {current_input}")
            
            elif is_dicom:
                # 如果是单个 dicom，创建一个目录放进去，因为 TotalSegmentator 喜欢目录
                dicom_dir = Path(tmp_dir) / "single_dicom"
                dicom_dir.mkdir()
                shutil.copy(input_path, dicom_dir / input_path.name)
                current_input = str(dicom_dir)

            # 运行 TotalSegmentator
            dev_map = {"GPU": "gpu", "CPU": "cpu", "MPS": "mps"}
            selected_device = dev_map.get(device, "cpu")
            
            print(f"Running task {task} on {selected_device}...")
            
            # 自动处理 dicom2nifti 可能的缺失切片错误
            # 设置环境变量跳过缺失切片检查（如果用户确定数据可用）
            os.environ["DICOM2NIFTI_ALLOW_MISSING_SLICES"] = "True"
            
            result = totalsegmentator(
                input=current_input,
                output=str(output_path),
                task=task,
                fast=(fast == "Fast (3mm)"),
                device=selected_device,
                ml=True
            )
            
            # 处理可能的返回元组 (seg_img, stats)
            if isinstance(result, tuple):
                result_img = result[0]
            else:
                result_img = result

            # 加载分割结果进行预览
            seg_img = result_img.get_fdata()
            
            # 尝试加载原始图像进行预览（如果是 DICOM/ZIP，我们需要转换后的 NIfTI）
            # TotalSegmentator 内部会转换，但我们这里为了预览，简单处理
            # 如果是 NIfTI 直接读，否则尝试从输出目录找转换后的原图（TotalSegmentator 可能会生成）
            # 或者直接只显示分割结果。为了体验，我们尝试读取。
            
            if is_2d:
                orig_img = nib.load(current_input).get_fdata()
            elif suffix in [".gz", ".nii"]:
                orig_img = nib.load(current_input).get_fdata()
            else:
                # 对于 DICOM，我们不方便直接读，预览图只显示分割结果或使用中间结果
                # 简单起见，如果读不到原图，就只显示分割图
                orig_img = np.zeros_like(seg_img)
            
            # 选择中间切面进行预览
            slice_idx = seg_img.shape[2] // 2
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # 原始图像预览
            if np.any(orig_img):
                ax1.imshow(orig_img[:, :, slice_idx].T, cmap='gray', origin='lower')
                ax1.set_title("原始图像预览")
            else:
                ax1.text(0.5, 0.5, "预览不可用 (DICOM)", ha='center')
            ax1.axis('off')
            
            # 分割结果叠加
            if np.any(orig_img):
                ax2.imshow(orig_img[:, :, slice_idx].T, cmap='gray', origin='lower')
            masked_seg = np.ma.masked_where(seg_img[:, :, slice_idx] == 0, seg_img[:, :, slice_idx])
            ax2.imshow(masked_seg.T, cmap='jet', alpha=0.5, origin='lower', interpolation='nearest')
            ax2.set_title("分割结果")
            ax2.axis('off')
            
            plt.tight_layout()
            
            # 保存预览图
            preview_path = Path(tmp_dir) / "preview.png"
            plt.savefig(preview_path)
            plt.close()
            
            # 结果持久化
            final_output_dir = Path("outputs")
            final_output_dir.mkdir(exist_ok=True)
            
            output_filename = f"seg_{input_path.stem}.nii.gz"
            final_nifti_path = final_output_dir / output_filename
            nib.save(result_img, str(final_nifti_path))
            
            final_preview_path = final_output_dir / f"preview_{input_path.stem}.png"
            shutil.copy(preview_path, final_preview_path)
            
            status_msg = "处理完成！"
            if is_2d:
                status_msg += " (2D 模式)"
            if is_zip or is_dicom:
                status_msg += " (DICOM 模式)"
            if bool(debug_mode):
                # Preserve tmp for post-mortem
                debug_out_dir = final_output_dir / debug_prefix
                debug_out_dir.mkdir(parents=True, exist_ok=True)
                preserved_tmp = debug_out_dir / "tmp"
                try:
                    if preserved_tmp.exists():
                        shutil.rmtree(preserved_tmp)
                    shutil.move(str(tmp_dir), str(preserved_tmp))
                    status_msg += f" | 调试数据已保留: {preserved_tmp}"
                    # Prevent later cleanup
                    tmp_dir = preserved_tmp
                except Exception as e:
                    status_msg += f" | 调试数据保留失败: {e}"
                
            return str(final_preview_path), str(final_nifti_path), status_msg
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Debug: preserve tmp + write DICOM summary (best-effort)
            err_msg = f"发生错误: {str(e)}"
            if bool(debug_mode):
                final_output_dir = Path("outputs")
                final_output_dir.mkdir(exist_ok=True)
                debug_out_dir = final_output_dir / debug_prefix
                debug_out_dir.mkdir(parents=True, exist_ok=True)
                preserved_tmp = debug_out_dir / "tmp"
                try:
                    if preserved_tmp.exists():
                        shutil.rmtree(preserved_tmp)
                    shutil.move(str(tmp_dir), str(preserved_tmp))
                    tmp_dir = preserved_tmp  # prevent cleanup
                except Exception as move_e:
                    err_msg += f" | 调试数据保留失败: {move_e}"

                # Attempt to locate a DICOM root inside tmp (zip extraction) or use current_input if it is a directory
                try:
                    dicom_root = None
                    if is_zip:
                        # app.py extracts zip into tmp_dir/extracted_zip
                        candidate = tmp_dir / "extracted_zip"
                        if candidate.exists():
                            dicom_root = candidate
                    elif is_dicom:
                        candidate = tmp_dir / "single_dicom"
                        if candidate.exists():
                            dicom_root = candidate
                    else:
                        # If current_input points to a folder, try summarize it
                        try:
                            ci = Path(current_input) if "current_input" in locals() else None
                            if ci is not None and ci.exists() and ci.is_dir():
                                dicom_root = ci
                        except Exception:
                            dicom_root = None

                    if dicom_root is not None and dicom_root.exists():
                        summary_path = debug_out_dir / "dicom_series_summary.json"
                        _dicom_series_summary(dicom_root, summary_path)
                        err_msg += f" | 调试目录: {debug_out_dir} | 统计报告: {summary_path.name}"
                    else:
                        err_msg += f" | 调试目录: {debug_out_dir}"
                except Exception as sum_e:
                    err_msg += f" | 生成DICOM统计失败: {sum_e}"

            return None, None, err_msg
    finally:
        # Cleanup tmp dir unless preserved into outputs/debug_*
        try:
            # If tmp_dir was moved into outputs, it will still exist but we should keep it.
            if debug_out_dir is None and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

# 获取所有可用任务
# 从 README 或 python_api 中分析出的常用任务
tasks = [
    "total", "body", "lung_vessels", "vertebrae_mr", "total_mr", 
    "appendicular_bones", "tissue_types", "heartchambers_highres",
    "face", "brain_structures", "liver_segments"
]

# 检查 MPS 可用性
default_device = "CPU"
if torch.backends.mps.is_available():
    default_device = "MPS"
elif torch.cuda.is_available():
    default_device = "GPU"

with gr.Blocks(title="TotalSegmentator Web UI") as demo:
    gr.Markdown("# 🩻 TotalSegmentator Web 界面")
    gr.Markdown("基于 TotalSegmentator 的解剖结构全自动分割工具。支持 CT 和 MR 图像。")
    
    with gr.Row():
        with gr.Column():
            input_file = gr.File(
                label="上传医学影像 (NIfTI, DICOM, ZIP, JPG, PNG)", 
                file_types=[".gz", ".nii", ".jpg", ".jpeg", ".png", ".dcm", ".dicom", ".zip"]
            )
            
            task_dropdown = gr.Dropdown(
                choices=tasks, 
                value="total", 
                label="选择分割任务",
                info="total 是默认的 117 类 CT 分割任务"
            )
            
            fast_radio = gr.Radio(
                choices=["Standard (1.5mm)", "Fast (3mm)"], 
                value="Standard (1.5mm)", 
                label="运行模式",
                info="Fast 模式更快，显存占用更低"
            )
            
            device_radio = gr.Radio(
                choices=["GPU", "CPU", "MPS"], 
                value=default_device, 
                label="运行设备",
                info=f"当前系统推荐: {default_device}"
            )

            debug_checkbox = gr.Checkbox(
                value=False,
                label="调试模式（保留临时文件 + 生成DICOM统计报告）",
                info="仅在需要排查 DICOM 转换失败时开启；会在 outputs/debug_* 目录保留中间文件与统计报告"
            )
            
            run_btn = gr.Button("开始分割", variant="primary")
            
        with gr.Column():
            status_text = gr.Textbox(label="状态", interactive=False)
            preview_img = gr.Image(label="分割预览 (中间切面)")
            output_file = gr.File(label="下载分割结果 (.nii.gz)")

    run_btn.click(
        fn=process_nifti,
        inputs=[input_file, task_dropdown, fast_radio, device_radio, debug_checkbox],
        outputs=[preview_img, output_file, status_text]
    )
    
    gr.Markdown("### 注意事项")
    gr.Markdown("- 首次运行新任务时会从 Zenodo 下载预训练权重，请耐心等待。")
    gr.Markdown("- 推荐在 Mac 上使用 MPS 以获得硬件加速。")
    gr.Markdown("- 原始图像较大时，处理可能需要几分钟时间。")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

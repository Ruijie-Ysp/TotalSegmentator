# LLaMA-Factory 医学影像 VLM 训练配置

本目录包含用于微调 Qwen2-VL 模型进行医学影像分析的 LLaMA-Factory 配置文件。

## 完整工作流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  上传医学影像    │ --> │  TotalSegmentator│ --> │  Label Studio   │ --> │  LLaMA-Factory  │
│  (CT/DICOM)     │     │  自动分割        │     │  人工审核标注    │     │  模型微调       │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `dataset_info.json` | 数据集配置，定义了训练数据的格式和字段映射 |
| `qwen2_vl_sft.yaml` | LoRA 微调配置（推荐，显存要求低） |
| `qwen2_vl_full.yaml` | 完整微调配置（需要大显存） |
| `qwen2_vl_inference.yaml` | 推理配置 |
| `ds_z3_config.json` | DeepSpeed ZeRO-3 配置 |
| `qa_templates.py` | 问答模板和数据增强策略 |

---

## 方式一：直接导出训练数据（无人工审核）

### 1. 导出训练数据

```bash
# 导出训练数据（ShareGPT 格式）
curl "http://localhost:8000/api/export/llama-factory/{session_id}?format=sharegpt&num_samples=5" > train_data.json

# 导出训练图像
curl "http://localhost:8000/api/export/llama-factory-images/{session_id}?slice_step=10"
```

---

## 方式二：通过 Label Studio 人工审核（推荐）

### 1. 获取 Label Studio 标注配置

```bash
# 获取 VQA 问答标注配置
curl "http://localhost:8000/api/labelstudio/config/vqa" | jq -r '.config' > labeling_config.xml

# 或获取器官分割标注配置（带特定 session 的器官）
curl "http://localhost:8000/api/labelstudio/config/segmentation?session_id={session_id}"
```

### 2. 导出 Label Studio 任务

```bash
# 导出完整的 Label Studio 任务（包含 AI 预标注）
curl "http://localhost:8000/api/labelstudio/tasks/{session_id}?base_url=http://your-server:8000" > tasks.json
```

### 3. 导入到 Label Studio

1. 创建新项目，粘贴标注配置
2. 导入 `tasks.json`
3. 人工审核/修正 AI 预标注的答案

### 4. 导出标注结果并转换为训练数据

```bash
# 从 Label Studio 导出 JSON 标注结果后
curl -X POST "http://localhost:8000/api/labelstudio/import-annotations?output_format=llama-factory" \
     -F "annotations_file=@exported_annotations.json" > train_data.json
```

### 2. 准备数据目录

```bash
# 创建数据目录
mkdir -p data/images

# 将 dataset_info.json 复制到 LLaMA-Factory 的 data 目录
cp dataset_info.json /path/to/LLaMA-Factory/data/

# 将训练数据放入 data 目录
mv train_data.json /path/to/LLaMA-Factory/data/ct_segmentation_train.json
```

### 3. 开始训练

```bash
cd /path/to/LLaMA-Factory

# LoRA 微调（推荐）
llamafactory-cli train qwen2_vl_sft.yaml

# 或使用 Web UI
llamafactory-cli webui
```

## 数据格式

### ShareGPT 格式（推荐）

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\n这张CT图像中能看到哪些器官？"},
    {"from": "gpt", "value": "在这张CT图像中，可以识别到以下器官：肝脏、脾脏、左肾、右肾..."}
  ],
  "images": ["images/session_axial_50.png"]
}
```

### Alpaca 格式

```json
{
  "instruction": "这张CT图像中能看到哪些器官？",
  "input": "",
  "output": "在这张CT图像中，可以识别到以下器官：肝脏、脾脏...",
  "images": ["images/session_axial_50.png"]
}
```

## 问答模板类型

| 类型 | 说明 | 示例 |
|------|------|------|
| 器官识别 | 列出图像中的所有器官 | "这张CT图像中能看到哪些器官？" |
| 器官定位 | 描述特定器官的位置 | "请描述肝脏在图像中的位置。" |
| 形态描述 | 描述器官的形态特征 | "分析脾脏的影像学特征。" |
| 区域分析 | 分析特定解剖区域 | "分析图像中腹部区域的器官分布。" |
| 解剖关系 | 描述器官间的位置关系 | "肝脏和胆囊的位置关系是什么？" |
| 临床相关 | 临床诊断角度分析 | "这张CT切片对诊断有什么参考价值？" |

## 数据增强策略

1. **问题改写**: 生成语义相同但表达不同的问题变体
2. **答案丰富**: 添加专业前缀和确定性描述
3. **多轮对话**: 模拟真实临床问诊场景
4. **对比学习**: 生成不同切片间的对比数据

## 硬件要求

| 配置 | 显存要求 | 推荐 GPU |
|------|----------|----------|
| LoRA 微调 | 24GB+ | RTX 3090/4090, A100 |
| 完整微调 | 80GB+ | A100 80G, H100 |

---

## Label Studio 集成 API 参考

### 核心 API 端点

| API | 方法 | 说明 |
|-----|------|------|
| `/api/labelstudio/config/{type}` | GET | 获取 Label Studio 标注配置模板 |
| `/api/labelstudio/tasks/{session_id}` | GET | 导出 Label Studio 任务（含 AI 预标注） |
| `/api/labelstudio/import-annotations` | POST | 导入标注结果，转换为训练数据 |
| `/api/labelstudio/export-images/{session_id}` | GET | 导出切片图像 |
| `/api/labelstudio/workflow-status` | GET | 获取工作流状态 |
| `/api/export/llama-factory/{session_id}` | GET | 直接导出 LLaMA-Factory 训练数据 |

### 参数说明

**`/api/labelstudio/tasks/{session_id}`**
- `slice_step`: 切片采样步长 (默认: 10)
- `axis`: 切片轴向 axial/sagittal/coronal (默认: axial)
- `num_qa_per_slice`: 每切片问答数 (默认: 3)
- `base_url`: API 基础 URL，用于图像链接

**`/api/labelstudio/import-annotations`**
- `output_format`: 输出格式 llama-factory/alpaca/raw (默认: llama-factory)

---

## 注意事项

1. 确保 `<image>` 标签数量与 `images` 列表中图像数量一致
2. 图像路径应为相对于训练数据目录的相对路径
3. 建议使用 PNG 格式保存医学影像以保持质量
4. 窗宽窗位建议使用默认值 (40/400) 或针对特定任务调整


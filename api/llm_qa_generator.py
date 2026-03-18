"""
LLM 问答生成器
支持模板模式和 LLM 多模态模式生成高质量医学影像问答对
支持传递图像给多模态大模型进行专业的影像分析和疾病诊断
"""
import json
import re
import random
import base64
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from api.config import get_config, LLMConfig


# 器官中英文映射
ORGAN_CN_MAP = {
    "spleen": "脾脏", "kidney_right": "右肾", "kidney_left": "左肾",
    "gallbladder": "胆囊", "liver": "肝脏", "stomach": "胃",
    "pancreas": "胰腺", "adrenal_gland_right": "右肾上腺", "adrenal_gland_left": "左肾上腺",
    "lung_upper_lobe_left": "左肺上叶", "lung_lower_lobe_left": "左肺下叶",
    "lung_upper_lobe_right": "右肺上叶", "lung_middle_lobe_right": "右肺中叶",
    "lung_lower_lobe_right": "右肺下叶", "esophagus": "食管", "trachea": "气管",
    "thyroid_gland": "甲状腺", "small_bowel": "小肠", "duodenum": "十二指肠",
    "colon": "结肠", "urinary_bladder": "膀胱", "prostate": "前列腺",
    "kidney_cyst_left": "左肾囊肿", "kidney_cyst_right": "右肾囊肿",
    "sacrum": "骶骨", "vertebrae": "脊椎", "intervertebral_discs": "椎间盘",
    "spinal_cord": "脊髓", "heart": "心脏", "aorta": "主动脉",
    "pulmonary_artery": "肺动脉", "brachiocephalic_trunk": "头臂干",
    "subclavian_artery_right": "右锁骨下动脉", "subclavian_artery_left": "左锁骨下动脉",
    "common_carotid_artery_right": "右颈总动脉", "common_carotid_artery_left": "左颈总动脉",
    "brachiocephalic_vein_left": "左头臂静脉", "brachiocephalic_vein_right": "右头臂静脉",
    "atrial_appendage_left": "左心耳", "superior_vena_cava": "上腔静脉",
    "inferior_vena_cava": "下腔静脉", "portal_vein_and_splenic_vein": "门静脉和脾静脉",
    "iliac_artery_left": "左髂动脉", "iliac_artery_right": "右髂动脉",
    "iliac_vena_left": "左髂静脉", "iliac_vena_right": "右髂静脉",
    "humerus_left": "左肱骨", "humerus_right": "右肱骨",
    "scapula_left": "左肩胛骨", "scapula_right": "右肩胛骨",
    "clavicula_left": "左锁骨", "clavicula_right": "右锁骨",
    "femur_left": "左股骨", "femur_right": "右股骨",
    "hip_left": "左髋骨", "hip_right": "右髋骨",
    "gluteus_maximus_left": "左臀大肌", "gluteus_maximus_right": "右臀大肌",
    "gluteus_medius_left": "左臀中肌", "gluteus_medius_right": "右臀中肌",
    "gluteus_minimus_left": "左臀小肌", "gluteus_minimus_right": "右臀小肌",
    "autochthon_left": "左竖脊肌", "autochthon_right": "右竖脊肌",
    "iliopsoas_left": "左髂腰肌", "iliopsoas_right": "右髂腰肌",
    "brain": "脑", "skull": "颅骨", "rib_left": "左侧肋骨", "rib_right": "右侧肋骨",
    "sternum": "胸骨", "costal_cartilages": "肋软骨"
}


def get_organ_cn(organ: str) -> str:
    """获取器官中文名"""
    return ORGAN_CN_MAP.get(organ, organ)


# 问题模板库
QUESTION_TEMPLATES = {
    "organ_list": [
        "这张CT图像中能看到哪些器官？",
        "请识别这张医学影像中的解剖结构。",
        "图像中显示了哪些人体器官？",
        "请列出这张CT切片中可见的所有器官。",
    ],
    "organ_location": [
        "请描述{organ}在图像中的位置。",
        "{organ}位于图像的什么位置？",
        "能否定位{organ}在这张CT中的具体位置？",
    ],
    "organ_feature": [
        "请描述{organ}的影像学特征。",
        "{organ}在这张图像中呈现什么形态？",
        "分析{organ}的CT表现。",
    ],
    "clinical": [
        "这张CT切片对临床诊断有什么参考价值？",
        "从这张图像可以获取哪些临床信息？",
        "这个层面的CT图像主要用于评估什么？",
    ],
    "anatomy_relation": [
        "{organ1}和{organ2}的位置关系如何？",
        "请描述{organ1}与{organ2}之间的解剖关系。",
    ],
    # 新增：疾病诊断分析类问题
    "diagnosis": [
        "请对这张CT图像进行诊断分析，描述可能存在的异常表现。",
        "作为放射科医生，请分析这张CT图像是否存在病变征象。",
        "请对图像中的器官进行系统性评估，指出正常和可能异常的表现。",
        "基于这张CT图像，请给出影像学诊断意见。",
        "请分析这张CT切片，描述各器官的形态、密度是否正常。",
    ],
    # 新增：影像报告类问题
    "report": [
        "请为这张CT图像撰写一份规范的影像学报告。",
        "请按照放射科报告格式，描述这张CT图像的所见和诊断意见。",
        "请生成这张CT切片的影像学描述报告。",
    ],
    # 新增：病变特征描述
    "lesion_analysis": [
        "请仔细观察这张CT图像，描述是否存在占位性病变、积液或其他异常。",
        "分析这张CT图像中{organ}是否存在异常表现，如有请详细描述。",
        "请评估图像中各器官的密度均匀性和边界清晰度。",
    ]
}


def generate_template_answer(question_type: str, organs: List[str], **kwargs) -> str:
    """基于模板生成回答"""
    organs_cn = [get_organ_cn(o) for o in organs]
    
    if question_type == "organ_list":
        if not organs_cn:
            return "在这张图像中未能识别到明显的器官结构。"
        organ_str = "、".join(organs_cn)
        return f"在这张CT图像中，可以识别到以下器官：{organ_str}。"
    
    elif question_type == "organ_location":
        organ = kwargs.get("organ", organs[0] if organs else "")
        organ_cn = get_organ_cn(organ)
        # 简化位置描述
        return f"{organ_cn}在图像中清晰可见，呈现其典型的解剖位置。"
    
    elif question_type == "organ_feature":
        organ = kwargs.get("organ", organs[0] if organs else "")
        organ_cn = get_organ_cn(organ)
        return f"{organ_cn}在CT图像中显示正常密度，边界清晰，形态规则。"
    
    elif question_type == "clinical":
        if organs_cn:
            return f"这张CT切片显示了{organs_cn[0]}等结构，可用于评估相关区域的病变情况。"
        return "这张CT切片可用于评估相应解剖区域的结构和病变。"
    
    elif question_type == "anatomy_relation":
        organ1 = kwargs.get("organ1", "")
        organ2 = kwargs.get("organ2", "")
        return f"{get_organ_cn(organ1)}与{get_organ_cn(organ2)}相邻，两者之间存在正常的解剖关系。"
    
    return "这是一张CT影像切片。"


async def generate_llm_answer(
    question: str,
    organs: List[str],
    slice_info: Dict,
    config: LLMConfig,
    image_base64: Optional[str] = None
) -> Optional[str]:
    """
    使用多模态 LLM 生成回答

    Args:
        question: 问题文本
        organs: 检测到的器官列表
        slice_info: 切片信息
        config: LLM 配置
        image_base64: 图像的 Base64 编码（并排对比图：左原始CT + 右分割叠加）

    Returns:
        LLM 生成的回答，失败返回 None
    """
    import httpx
    import time

    slice_idx = slice_info.get('slice_idx', 0)
    total_slices = slice_info.get('total_slices', 0)
    axis = slice_info.get('axis', 'axial')

    # 日志：开始调用
    print(f"\n{'='*60}")
    print("[LLM] 🚀 开始调用大模型")
    print(f"[LLM] 📍 切片: {axis} 第 {slice_idx}/{total_slices} 层")
    print(f"[LLM] 🔧 模型: {config.model}")
    print(f"[LLM] 🌐 API: {config.base_url}")
    print(f"[LLM] 📷 图像: {'有 (' + str(len(image_base64)//1024) + 'KB)' if image_base64 else '无'}")
    print(f"[LLM] ❓ 问题: {question[:50]}{'...' if len(question) > 50 else ''}")

    organs_cn = [get_organ_cn(o) for o in organs]
    print(f"[LLM] 🫀 器官: {', '.join(organs_cn[:5])}{'...' if len(organs_cn) > 5 else ''}")

    # 构建上下文信息
    axis_map = {"axial": "横断面", "sagittal": "矢状面", "coronal": "冠状面"}
    axis_cn = axis_map.get(axis, '横断面')

    context = f"""当前CT切片信息：
- 视图方向：{axis_cn}
- 切片位置：第 {slice_idx} 层，共 {total_slices} 层
- AI分割检测到的器官：{', '.join(organs_cn)}

图像说明：
- 左侧图像：原始CT图像（经窗宽窗位调整）
- 右侧图像：分割叠加图（不同颜色标注不同器官区域）

请基于图像内容回答以下问题，注意：
1. 仔细观察原始CT图像的密度、形态、边界等特征
2. 参考右侧分割图确认器官位置
3. 如涉及诊断，请给出专业、谨慎的分析"""

    # 构建消息内容
    if image_base64:
        # 多模态消息格式（支持 OpenAI、Qwen-VL 等）
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high"  # 高清模式，确保细节可见
                }
            },
            {
                "type": "text",
                "text": f"{context}\n\n问题：{question}"
            }
        ]
    else:
        # 纯文本模式（无图像时回退）
        user_content = f"{context}\n\n问题：{question}"

    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_content}
    ]

    # 使用配置的超时时间，默认 120 秒
    timeout_seconds = getattr(config, 'timeout', 120)

    start_time = time.time()
    print(f"[LLM] ⏳ 发送请求中... (超时设置: {timeout_seconds}s)")

    try:
        async with httpx.AsyncClient(timeout=float(timeout_seconds)) as client:
            response = await client.post(
                f"{config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": config.model,
                    "messages": messages,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]

                # 获取 token 使用信息（如果有）
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "N/A")
                completion_tokens = usage.get("completion_tokens", "N/A")
                total_tokens = usage.get("total_tokens", "N/A")

                # 详细日志输出
                print("[LLM] ✅ 响应成功!")
                print(f"[LLM] ⏱️ 耗时: {elapsed:.2f}s")
                print(f"[LLM] 📊 Token 使用: 输入={prompt_tokens}, 输出={completion_tokens}, 总计={total_tokens}")
                print(f"[LLM] 📝 回答长度: {len(answer)} 字符")
                print(f"[LLM] 💬 回答预览: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                print(f"[LLM] 🔄 超时计时已重置 (下次超时: {timeout_seconds}s)")
                print(f"{'='*60}\n")

                return answer
            else:
                print(f"[LLM] ❌ API错误! 状态码: {response.status_code}")
                print(f"[LLM] 📄 响应内容: {response.text[:500]}")
                print(f"[LLM] ⏱️ 耗时: {elapsed:.2f}s")
                print(f"{'='*60}\n")
                return None
    except httpx.TimeoutException:
        elapsed = time.time() - start_time
        print(f"[LLM] ⏰ 请求超时! 耗时: {elapsed:.2f}s (超过 {timeout_seconds}s)")
        print(f"{'='*60}\n")
        return None
    except httpx.ConnectError as e:
        elapsed = time.time() - start_time
        print(f"[LLM] 🔌 连接失败! 无法连接到 {config.base_url}")
        print(f"[LLM] 📄 错误: {e}")
        print(f"{'='*60}\n")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[LLM] ❌ 调用失败! 耗时: {elapsed:.2f}s")
        print(f"[LLM] 📄 错误类型: {type(e).__name__}")
        print(f"[LLM] 📄 错误信息: {e}")
        print(f"{'='*60}\n")
        return None


async def generate_qa_pairs(
    organs: List[str],
    slice_info: Dict,
    num_pairs: int = 3,
    use_llm: bool = False,
    include_types: Optional[List[str]] = None,
    image_base64: Optional[str] = None,
    enabled_categories: Optional[Dict[str, bool]] = None,
    priority_categories: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    生成问答对（支持多模态）

    Args:
        organs: 检测到的器官列表
        slice_info: 切片信息 {axis, slice_idx, total_slices}
        num_pairs: 生成的问答对数量
        use_llm: 是否使用 LLM 生成
        include_types: 包含的问题类型
        image_base64: 图像 Base64（并排对比图）
        enabled_categories: 启用的类别 {category: enabled}
        priority_categories: 重点类别列表（优先使用）

    Returns:
        问答对列表 [{"question": ..., "answer": ..., "type": ..., "source": ...}]
    """
    config = get_config()
    qa_pairs = []

    # 类型映射：将 main.py 中的类别名映射到 llm_qa_generator 的类型名
    category_to_type = {
        "organ_identification": "organ_list",
        "organ_location": "organ_location",
        "organ_description": "organ_feature",
        "diagnostic_analysis": "diagnosis",
        "comprehensive_report": "report",
        "region_analysis": "clinical",
        "anatomical_relation": "anatomy_relation",
        "clinical_relevance": "clinical"
    }

    # 确定要使用的问题类型，考虑模板选择状态
    if include_types is None:
        # 根据启用状态和优先级构建类型列表
        if enabled_categories is not None:
            # 优先添加重点类别
            priority_types = []
            enabled_types = []

            for category, enabled in enabled_categories.items():
                if enabled:
                    q_type = category_to_type.get(category, category)
                    if priority_categories and category in priority_categories:
                        priority_types.append(q_type)
                    else:
                        enabled_types.append(q_type)

            # 重点类别优先
            include_types = priority_types + enabled_types

            if not include_types:
                # 如果没有任何启用的类别，使用默认
                if use_llm and image_base64:
                    include_types = ["organ_list", "organ_feature", "diagnosis", "clinical", "report"]
                else:
                    include_types = ["organ_list", "organ_location", "organ_feature", "clinical"]
        else:
            # 原有逻辑：如果使用 LLM 且有图像，增加诊断分析类问题
            if use_llm and image_base64:
                include_types = [
                    "organ_list", "organ_feature", "diagnosis",
                    "clinical", "lesion_analysis", "report"
                ]
            else:
                include_types = ["organ_list", "organ_location", "organ_feature", "clinical"]

    # 如果启用 LLM 且有 API Key
    should_use_llm = use_llm and config.llm.enabled and config.llm.api_key

    organs_cn = [get_organ_cn(o) for o in organs]

    # 日志：问答生成开始
    slice_idx = slice_info.get('slice_idx', 0)
    axis = slice_info.get('axis', 'axial')
    print(f"\n[QA生成] 📋 切片 {axis}:{slice_idx} 开始生成 {num_pairs} 个问答对")
    print(f"[QA生成] 🔧 模式: {'LLM多模态' if should_use_llm else '模板'}")
    print(f"[QA生成] 📂 问题类型: {include_types}")

    for i in range(num_pairs):
        # 选择问题类型
        q_type = include_types[i % len(include_types)]
        templates = QUESTION_TEMPLATES.get(q_type, QUESTION_TEMPLATES["organ_list"])

        # 选择问题模板
        template = random.choice(templates)

        # 填充模板变量
        if "{organ}" in template and organs:
            organ = random.choice(organs)
            question = template.format(organ=get_organ_cn(organ))
            kwargs = {"organ": organ}
        elif "{organ1}" in template and len(organs) >= 2:
            organ1, organ2 = random.sample(organs, 2)
            question = template.format(
                organ1=get_organ_cn(organ1),
                organ2=get_organ_cn(organ2)
            )
            kwargs = {"organ1": organ1, "organ2": organ2}
        else:
            question = template
            kwargs = {}

        print(f"[QA生成] 🔄 ({i+1}/{num_pairs}) 类型={q_type}")

        # 生成回答
        if should_use_llm:
            # 传递图像给多模态 LLM
            answer = await generate_llm_answer(
                question, organs, slice_info, config.llm, image_base64
            )
            if answer is None:
                # LLM 失败，回退到模板
                answer = generate_template_answer(q_type, organs, **kwargs)
                source = "template_fallback"
                print("[QA生成] ⚠️ LLM失败，使用模板回退")
            else:
                source = "llm"
        else:
            answer = generate_template_answer(q_type, organs, **kwargs)
            source = "template"

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "type": q_type,
            "source": source
        })

    print(f"[QA生成] ✅ 切片 {axis}:{slice_idx} 完成 {len(qa_pairs)} 个问答对\n")
    return qa_pairs


async def generate_batch_qa(
    session_data: Dict,
    slice_indices: List[int],
    axis: str = "axial",
    num_qa_per_slice: int = 3,
    use_llm: bool = False
) -> List[Dict]:
    """
    批量生成多个切片的问答对

    Returns:
        [{slice_idx, axis, organs, qa_pairs}, ...]
    """
    results = []
    total_slices = session_data.get("shape", [0, 0, 0])[2] if axis == "axial" else session_data.get("shape", [0, 0, 0])[0]

    for slice_idx in slice_indices:
        slice_info = {
            "axis": axis,
            "slice_idx": slice_idx,
            "total_slices": total_slices
        }

        # 获取该切片的器官（从分割数据中提取）
        organs = session_data.get("organs", [])

        qa_pairs = await generate_qa_pairs(
            organs=organs,
            slice_info=slice_info,
            num_pairs=num_qa_per_slice,
            use_llm=use_llm
        )

        results.append({
            "slice_idx": slice_idx,
            "axis": axis,
            "organs": organs,
            "organs_cn": [get_organ_cn(o) for o in organs],
            "qa_pairs": qa_pairs
        })

    return results

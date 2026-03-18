"""
医学影像问答模板库 - 用于 LLaMA-Factory 训练数据生成

包含多样化的问答模板，覆盖以下场景:
1. 器官识别与列举
2. 解剖位置描述
3. 形态学特征描述
4. 异常检测与报告
5. 临床相关性分析
6. 多器官关系描述
"""

import random
from typing import List, Dict, Tuple, Optional

# ==================== 问题模板库 ====================

# 器官识别类 - 询问图像中的器官
ORGAN_IDENTIFICATION_TEMPLATES = [
    "这张CT图像中能看到哪些器官？",
    "请识别图像中所有可见的解剖结构。",
    "图中显示了哪些人体器官和组织？",
    "分析这张医学影像，列出所有识别到的器官。",
    "这张切片包含哪些解剖结构？",
    "请告诉我图像中可以识别的器官。",
    "从这张CT扫描中你能识别出什么器官？",
    "这是一张CT图像，请分析其中的解剖结构。",
]

# 特定器官定位类 - 询问某个器官的位置
ORGAN_LOCATION_TEMPLATES = [
    "请描述{organ}在图像中的位置。",
    "{organ}在这张CT切片中位于哪里？",
    "能看到{organ}吗？它在图像的什么位置？",
    "找出并定位{organ}在图像中的区域。",
    "这张图像中{organ}的解剖位置是什么？",
    "{organ}在切片中占据什么区域？",
]

# 器官形态描述类
ORGAN_DESCRIPTION_TEMPLATES = [
    "请描述{organ}的形态特征。",
    "{organ}在这张图像中表现如何？",
    "分析{organ}的影像学特征。",
    "{organ}的大小、形态和密度如何？",
    "这张图像中{organ}有什么特点？",
    "描述{organ}的边界和内部结构。",
]

# 区域分析类
REGION_ANALYSIS_TEMPLATES = [
    "分析图像中{region}区域的器官分布。",
    "{region}区域可见哪些结构？",
    "这张切片主要展示{region}的哪些组织？",
    "{region}区域的主要解剖标志是什么？",
    "描述{region}区域的影像学表现。",
]

# 解剖关系类
ANATOMICAL_RELATION_TEMPLATES = [
    "{organ1}和{organ2}的位置关系是什么？",
    "描述{organ1}与{organ2}之间的解剖关系。",
    "{organ1}相对于{organ2}位于什么方向？",
    "这两个结构({organ1}和{organ2})在解剖上有什么关联？",
]

# 临床相关类
CLINICAL_RELEVANCE_TEMPLATES = [
    "从临床角度分析这张图像显示的主要结构。",
    "这张CT切片对诊断有什么参考价值？",
    "这个层面的CT图像通常用于评估什么？",
    "该切片层面的关键解剖标志有哪些？",
    "这张图像可以用来观察哪些临床相关结构？",
]

# 异常检测类（正常情况）
ABNORMALITY_DETECTION_TEMPLATES = [
    "这张图像中有没有明显的异常？",
    "各器官是否表现正常？",
    "能发现任何病变或异常吗？",
    "从影像学角度看，有什么需要注意的地方？",
]

# ==================== 回答模板库 ====================

# 器官列表回答模板
ORGAN_LIST_ANSWER_TEMPLATES = [
    "在这张CT图像中，可以识别到以下器官：{organs}。",
    "这张切片显示了{organs}等解剖结构。",
    "图像中可见的主要器官包括：{organs}。",
    "分析该CT切片，能够辨认出{organs}。",
    "从这张图像中我识别到了{organs}。",
]

# 详细器官列表（按区域分组）
DETAILED_ORGAN_LIST_TEMPLATE = (
    "在这张CT图像中，共识别到{count}个解剖结构。{region_details}"
)

# 器官定位回答模板
ORGAN_LOCATION_ANSWER_TEMPLATES = [
    "{organ}在图像中位于{position}区域，{anatomical_desc}。",
    "是的，可以看到{organ}，位于图像的{position}部分。{anatomical_desc}。",
    "{organ}清晰可见，位于{position}，{anatomical_desc}。",
]

# 正常器官描述模板
NORMAL_ORGAN_DESCRIPTION_TEMPLATES = [
    "{organ}在图像中显示正常形态，边界清晰，密度均匀，未见明显异常信号。",
    "该{organ}形态规则，大小正常，内部结构均匀，未见明显占位或异常密度灶。",
    "{organ}表现正常，轮廓清楚，密度均匀，无明显病变征象。",
    "图像中{organ}形态、大小、密度均在正常范围内。",
]

# 无异常回答模板
NO_ABNORMALITY_TEMPLATES = [
    "在这张图像中，各可见器官形态、大小、密度均正常，未发现明显异常。",
    "各解剖结构显示正常，未见明显病变或异常信号。",
    "影像学检查未见明显异常，各器官表现正常。",
    "该切片层面所示结构均未见明显异常改变。",
]


# ==================== 解剖区域定义 ====================

ANATOMICAL_REGIONS = {
    "腹部": {
        "organs": ["liver", "spleen", "pancreas", "gallbladder", "stomach",
                   "small_bowel", "duodenum", "colon", "kidney_left", "kidney_right",
                   "adrenal_gland_left", "adrenal_gland_right"],
        "description": "位于膈肌以下、盆腔以上的区域"
    },
    "胸部": {
        "organs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                   "lung_middle_lobe_right", "lung_lower_lobe_right", "heart", "aorta",
                   "trachea", "esophagus", "pulmonary_artery"],
        "description": "位于颈部以下、膈肌以上的区域"
    },
    "盆腔": {
        "organs": ["urinary_bladder", "prostate", "hip_left", "hip_right", "sacrum"],
        "description": "位于腹部以下的盆腔区域"
    },
    "脊柱": {
        "organs": ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
                   "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
                   "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10",
                   "vertebrae_T11", "vertebrae_T12", "vertebrae_C1", "vertebrae_C2", "vertebrae_C3",
                   "vertebrae_C4", "vertebrae_C5", "vertebrae_C6", "vertebrae_C7", "spinal_cord"],
        "description": "人体中轴骨骼系统"
    },
    "心血管": {
        "organs": ["heart", "aorta", "inferior_vena_cava", "superior_vena_cava",
                   "portal_vein_and_splenic_vein", "pulmonary_artery", "pulmonary_vein"],
        "description": "心脏及大血管系统"
    },
}


# ==================== 数据增强策略 ====================

class DataAugmentationStrategy:
    """训练数据增强策略"""

    @staticmethod
    def paraphrase_question(question: str) -> List[str]:
        """问题改写 - 生成语义相同但表达不同的问题变体"""
        variations = []

        # 添加礼貌用语
        polite_prefixes = ["请问", "麻烦", "能否告诉我"]
        for prefix in polite_prefixes:
            if not question.startswith(prefix):
                variations.append(f"{prefix}，{question}")

        # 添加专业前缀
        professional_prefixes = ["从影像学角度", "根据CT表现", "基于这张图像"]
        for prefix in professional_prefixes:
            if prefix not in question:
                variations.append(f"{prefix}，{question}")

        return variations

    @staticmethod
    def augment_answer(answer: str, organ_count: int) -> List[str]:
        """答案增强 - 生成语义相同但表达略有不同的答案"""
        variations = [answer]

        # 添加确定性前缀
        certainty_prefixes = ["根据分析，", "经过仔细观察，", "从图像中可以确认，"]
        for prefix in certainty_prefixes:
            if not answer.startswith(prefix):
                variations.append(f"{prefix}{answer}")

        return variations

    @staticmethod
    def generate_multi_turn_conversation(
        organs: List[str],
        organ_cn_names: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """生成多轮对话数据"""
        conversations = []

        if not organs:
            return conversations

        # 第一轮：询问所有器官
        q1 = random.choice(ORGAN_IDENTIFICATION_TEMPLATES)
        organ_names = [organ_cn_names.get(o, o) for o in organs]
        a1 = random.choice(ORGAN_LIST_ANSWER_TEMPLATES).format(
            organs="、".join(organ_names)
        )
        conversations.append({"from": "human", "value": f"<image>\n{q1}"})
        conversations.append({"from": "gpt", "value": a1})

        # 第二轮：询问特定器官
        if len(organs) >= 1:
            organ = random.choice(organs)
            organ_cn = organ_cn_names.get(organ, organ)
            q2 = random.choice(ORGAN_LOCATION_TEMPLATES).format(organ=organ_cn)
            a2 = random.choice(NORMAL_ORGAN_DESCRIPTION_TEMPLATES).format(organ=organ_cn)
            conversations.append({"from": "human", "value": q2})
            conversations.append({"from": "gpt", "value": a2})

        # 第三轮：询问是否有异常
        q3 = random.choice(ABNORMALITY_DETECTION_TEMPLATES)
        a3 = random.choice(NO_ABNORMALITY_TEMPLATES)
        conversations.append({"from": "human", "value": q3})
        conversations.append({"from": "gpt", "value": a3})

        return conversations

    @staticmethod
    def generate_contrastive_pairs(
        organs_set1: List[str],
        organs_set2: List[str],
        organ_cn_names: Dict[str, str]
    ) -> List[Tuple[Dict, Dict]]:
        """生成对比学习数据对 - 用于区分不同切片/不同器官"""
        pairs = []

        # 找出差异器官
        unique_to_set1 = set(organs_set1) - set(organs_set2)
        unique_to_set2 = set(organs_set2) - set(organs_set1)

        if unique_to_set1 or unique_to_set2:
            q = "这张图像与之前的图像相比，器官组成有什么不同？"

            diff_parts = []
            if unique_to_set1:
                names = [organ_cn_names.get(o, o) for o in unique_to_set1]
                diff_parts.append(f"此图像独有的器官：{', '.join(names)}")
            if unique_to_set2:
                names = [organ_cn_names.get(o, o) for o in unique_to_set2]
                diff_parts.append(f"缺少的器官：{', '.join(names)}")

            a = "；".join(diff_parts) + "。"

            pairs.append((
                {"question": q, "answer": a, "type": "contrastive"},
                {"set1": organs_set1, "set2": organs_set2}
            ))

        return pairs


# ==================== 使用示例 ====================

def generate_training_sample(
    organs: List[str],
    organ_cn_names: Dict[str, str],
    image_path: str,
    template_type: str = "random"
) -> Dict:
    """
    生成单个训练样本

    Args:
        organs: 器官列表（英文名）
        organ_cn_names: 英文到中文的映射
        image_path: 图像路径
        template_type: 模板类型 (random, identification, location, description)

    Returns:
        ShareGPT 格式的训练样本
    """
    if template_type == "random":
        template_type = random.choice(["identification", "location", "description"])

    if template_type == "identification":
        q = random.choice(ORGAN_IDENTIFICATION_TEMPLATES)
        organ_names = [organ_cn_names.get(o, o) for o in organs]
        a = random.choice(ORGAN_LIST_ANSWER_TEMPLATES).format(
            organs="、".join(organ_names)
        )
    elif template_type == "location" and organs:
        organ = random.choice(organs)
        organ_cn = organ_cn_names.get(organ, organ)
        q = random.choice(ORGAN_LOCATION_TEMPLATES).format(organ=organ_cn)
        a = f"{organ_cn}在图像中清晰可见，形态正常，边界清楚。"
    else:  # description
        if organs:
            organ = random.choice(organs)
            organ_cn = organ_cn_names.get(organ, organ)
            q = random.choice(ORGAN_DESCRIPTION_TEMPLATES).format(organ=organ_cn)
            a = random.choice(NORMAL_ORGAN_DESCRIPTION_TEMPLATES).format(organ=organ_cn)
        else:
            q = random.choice(ORGAN_IDENTIFICATION_TEMPLATES)
            a = "在这张图像中未能识别到明显的器官结构。"

    return {
        "conversations": [
            {"from": "human", "value": f"<image>\n{q}"},
            {"from": "gpt", "value": a}
        ],
        "images": [image_path]
    }


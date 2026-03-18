"""
配置管理模块
支持 LLM 配置、Session 持久化等
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from datetime import datetime

# 获取项目根目录（api 目录的父目录）
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent

# 配置文件路径（使用绝对路径确保一致性）
CONFIG_DIR = _PROJECT_ROOT / "api_data" / ".config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "settings.json"
SESSIONS_FILE = CONFIG_DIR / "sessions.json"


@dataclass
class LLMConfig:
    """LLM 配置"""
    enabled: bool = False
    provider: str = "openai"  # openai 兼容
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 120  # LLM 请求超时时间（秒）
    # 提示词模板
    system_prompt: str = """你是一位专业的医学影像分析专家。根据提供的 CT 影像信息，生成专业、准确的医学描述。
请使用中文回答，语言要专业但易于理解。"""

    # 问答生成提示词
    qa_prompt_template: str = """基于以下 CT 切片信息生成专业的医学问答：

切片信息：
- 轴向：{axis}
- 切片位置：{slice_idx}/{total_slices}
- 检测到的器官：{organs}
- 器官中文名：{organs_cn}

请生成 {num_qa} 个高质量的问答对，格式如下：
Q: [问题]
A: [详细专业的回答]

问答类型包括：器官识别、位置描述、形态分析、临床意义等。"""


@dataclass
class LabelStudioConfig:
    """Label Studio 配置"""
    enabled: bool = False
    url: str = "http://localhost:8080"
    api_key: str = ""
    default_project_name: str = "TotalSegmentator-VQA"
    # 同步参数默认值
    default_slice_step: int = 10
    default_axis: str = "axial"
    # MinIO 公开访问 URL
    minio_public_url: str = ""


@dataclass
class AppConfig:
    """应用配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    labelstudio: LabelStudioConfig = field(default_factory=LabelStudioConfig)
    session_persist: bool = True  # 是否持久化 session
    auto_save_interval: int = 60  # 自动保存间隔（秒）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm": asdict(self.llm),
            "labelstudio": asdict(self.labelstudio),
            "session_persist": self.session_persist,
            "auto_save_interval": self.auto_save_interval
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        llm_data = data.get("llm", {})
        llm = LLMConfig(**llm_data) if llm_data else LLMConfig()
        ls_data = data.get("labelstudio", {})
        labelstudio = LabelStudioConfig(**ls_data) if ls_data else LabelStudioConfig()
        return cls(
            llm=llm,
            labelstudio=labelstudio,
            session_persist=data.get("session_persist", True),
            auto_save_interval=data.get("auto_save_interval", 60)
        )


# 全局配置实例
_config: Optional[AppConfig] = None


def load_config() -> AppConfig:
    """加载配置"""
    global _config
    if _config is not None:
        return _config
    
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            _config = AppConfig.from_dict(data)
        except Exception as e:
            print(f"加载配置失败: {e}，使用默认配置")
            _config = AppConfig()
    else:
        _config = AppConfig()
    
    # 从环境变量覆盖 LLM 配置
    if os.getenv("LLM_API_KEY"):
        _config.llm.api_key = os.getenv("LLM_API_KEY", "")
        _config.llm.enabled = True
    if os.getenv("LLM_BASE_URL"):
        _config.llm.base_url = os.getenv("LLM_BASE_URL", "")
    if os.getenv("LLM_MODEL"):
        _config.llm.model = os.getenv("LLM_MODEL", "")
    
    return _config


def save_config(config: AppConfig) -> None:
    """保存配置"""
    global _config
    _config = config
    CONFIG_FILE.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def get_config() -> AppConfig:
    """获取当前配置"""
    return load_config()


def update_llm_config(
    enabled: Optional[bool] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    system_prompt: Optional[str] = None,
    qa_prompt_template: Optional[str] = None
) -> LLMConfig:
    """更新 LLM 配置"""
    config = get_config()

    if enabled is not None:
        config.llm.enabled = enabled
    if api_key is not None:
        config.llm.api_key = api_key
    if base_url is not None:
        config.llm.base_url = base_url
    if model is not None:
        config.llm.model = model
    if temperature is not None:
        config.llm.temperature = temperature
    if max_tokens is not None:
        config.llm.max_tokens = max_tokens
    if timeout is not None:
        config.llm.timeout = timeout
    if system_prompt is not None:
        config.llm.system_prompt = system_prompt
    if qa_prompt_template is not None:
        config.llm.qa_prompt_template = qa_prompt_template

    save_config(config)
    return config.llm


def update_labelstudio_config(
    enabled: Optional[bool] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    default_project_name: Optional[str] = None,
    default_slice_step: Optional[int] = None,
    default_axis: Optional[str] = None,
    minio_public_url: Optional[str] = None
) -> LabelStudioConfig:
    """更新 Label Studio 配置"""
    config = get_config()

    if enabled is not None:
        config.labelstudio.enabled = enabled
    if url is not None:
        config.labelstudio.url = url
    if api_key is not None:
        config.labelstudio.api_key = api_key
    if default_project_name is not None:
        config.labelstudio.default_project_name = default_project_name
    if default_slice_step is not None:
        config.labelstudio.default_slice_step = default_slice_step
    if default_axis is not None:
        config.labelstudio.default_axis = default_axis
    if minio_public_url is not None:
        config.labelstudio.minio_public_url = minio_public_url

    save_config(config)
    return config.labelstudio


def get_labelstudio_config() -> LabelStudioConfig:
    """获取 Label Studio 配置（优先从配置文件，其次从环境变量）"""
    config = get_config()
    ls_config = config.labelstudio

    # 如果配置为空，尝试从环境变量读取
    if not ls_config.url or ls_config.url == "http://localhost:8080":
        env_url = os.getenv("LABEL_STUDIO_URL")
        if env_url:
            ls_config.url = env_url

    if not ls_config.api_key:
        env_key = os.getenv("LABEL_STUDIO_API_KEY")
        if env_key:
            ls_config.api_key = env_key
            ls_config.enabled = True

    if not ls_config.minio_public_url:
        env_minio = os.getenv("MINIO_PUBLIC_URL")
        if env_minio:
            ls_config.minio_public_url = env_minio

    return ls_config


# ============ Session 持久化 ============

def save_sessions(sessions: Dict[str, Any]) -> None:
    """保存所有 sessions 到文件"""
    # 只保存可序列化的数据
    serializable = {}
    for sid, session in sessions.items():
        try:
            # 测试是否可序列化
            json.dumps(session)
            serializable[sid] = session
        except (TypeError, ValueError):
            # 跳过不可序列化的 session
            print(f"Session {sid} 包含不可序列化数据，跳过保存")
    
    SESSIONS_FILE.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_sessions() -> Dict[str, Any]:
    """从文件加载 sessions"""
    if SESSIONS_FILE.exists():
        try:
            data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
            print(f"已加载 {len(data)} 个 session")
            return data
        except Exception as e:
            print(f"加载 sessions 失败: {e}")
    return {}


def save_single_session(session_id: str, session_data: Dict[str, Any]) -> None:
    """保存单个 session"""
    sessions = load_sessions()
    sessions[session_id] = session_data
    save_sessions(sessions)


def delete_session_file(session_id: str) -> None:
    """从持久化存储中删除 session"""
    sessions = load_sessions()
    if session_id in sessions:
        del sessions[session_id]
        save_sessions(sessions)


# ============ QA 模板配置持久化 ============

QA_TEMPLATES_FILE = CONFIG_DIR / "qa_templates.json"


@dataclass
class QATemplateState:
    """QA 模板选择状态"""
    # 模板选择状态：{category: {"enabled": bool, "priority": bool}}
    selection_state: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    # 自定义模板：{category: [{"q": str, "type": str}]}
    custom_templates: Dict[str, list] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_state": self.selection_state,
            "custom_templates": self.custom_templates
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QATemplateState":
        return cls(
            selection_state=data.get("selection_state", {}),
            custom_templates=data.get("custom_templates", {})
        )


# 全局 QA 模板状态
_qa_template_state: Optional[QATemplateState] = None


def load_qa_template_state() -> QATemplateState:
    """加载 QA 模板状态"""
    global _qa_template_state
    if _qa_template_state is not None:
        return _qa_template_state

    if QA_TEMPLATES_FILE.exists():
        try:
            data = json.loads(QA_TEMPLATES_FILE.read_text(encoding="utf-8"))
            _qa_template_state = QATemplateState.from_dict(data)
            print(f"[QA模板] 已加载模板配置: {len(_qa_template_state.selection_state)} 个类别状态, {len(_qa_template_state.custom_templates)} 个自定义模板")
        except Exception as e:
            print(f"[QA模板] 加载配置失败: {e}，使用默认配置")
            _qa_template_state = QATemplateState()
    else:
        _qa_template_state = QATemplateState()

    return _qa_template_state


def save_qa_template_state(state: QATemplateState) -> None:
    """保存 QA 模板状态"""
    global _qa_template_state
    _qa_template_state = state
    QA_TEMPLATES_FILE.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("[QA模板] 已保存配置")


def get_qa_template_state() -> QATemplateState:
    """获取当前 QA 模板状态"""
    return load_qa_template_state()


def update_template_selection(category: str, enabled: Optional[bool] = None, priority: Optional[bool] = None) -> Dict[str, bool]:
    """更新单个模板类别的选择状态"""
    state = get_qa_template_state()

    if category not in state.selection_state:
        state.selection_state[category] = {"enabled": True, "priority": False}

    if enabled is not None:
        state.selection_state[category]["enabled"] = enabled
    if priority is not None:
        state.selection_state[category]["priority"] = priority

    save_qa_template_state(state)
    return state.selection_state[category]


def update_template_selection_batch(selections: Dict[str, Dict[str, bool]]) -> List[str]:
    """批量更新模板选择状态"""
    state = get_qa_template_state()
    updated = []

    for category, selection in selections.items():
        state.selection_state[category] = {
            "enabled": selection.get("enabled", True),
            "priority": selection.get("priority", False)
        }
        updated.append(category)

    save_qa_template_state(state)
    return updated


def update_custom_templates(category: str, templates: list) -> None:
    """更新自定义模板"""
    state = get_qa_template_state()
    state.custom_templates[category] = templates
    save_qa_template_state(state)


def reset_qa_templates_config(category: Optional[str] = None) -> None:
    """重置 QA 模板配置"""
    state = get_qa_template_state()

    if category:
        if category in state.custom_templates:
            del state.custom_templates[category]
        if category in state.selection_state:
            del state.selection_state[category]
    else:
        state.custom_templates.clear()
        state.selection_state.clear()

    save_qa_template_state(state)


def get_template_selection_state(category: str) -> Dict[str, bool]:
    """获取单个类别的选择状态"""
    state = get_qa_template_state()
    return state.selection_state.get(category, {"enabled": True, "priority": False})


def get_all_template_selection_states() -> Dict[str, Dict[str, bool]]:
    """获取所有模板类别的选择状态"""
    state = get_qa_template_state()
    return state.selection_state.copy()


def get_custom_templates(category: str) -> list:
    """获取自定义模板"""
    state = get_qa_template_state()
    return state.custom_templates.get(category, [])

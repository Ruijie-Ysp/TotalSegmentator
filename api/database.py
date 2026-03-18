"""
数据库连接和模型定义
使用 SQLAlchemy 2.0+ 异步模式
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, AsyncGenerator
from contextlib import asynccontextmanager

# 确保环境变量已加载
from dotenv import load_dotenv
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from sqlalchemy import String, Integer, Text, Boolean, DateTime, JSON, ARRAY, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 数据库连接配置
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "totalseg")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "totalseg_secret")
POSTGRES_DB = os.getenv("POSTGRES_DB", "totalseg")

DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# 创建异步引擎
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DEBUG", "false").lower() == "true",
    pool_size=10,
    max_overflow=20,
)

# 创建会话工厂
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    """SQLAlchemy 模型基类"""
    pass


class Task(Base):
    """任务模型 - 存储 CT 分割任务信息"""
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[Optional[str]] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="uploaded", index=True)
    task_type: Mapped[str] = mapped_column(String(32), default="total")

    # 处理进度
    progress: Mapped[int] = mapped_column(Integer, default=0)
    progress_message: Mapped[Optional[str]] = mapped_column(String(255))

    # 处理结果
    shape: Mapped[Optional[List[int]]] = mapped_column(ARRAY(Integer))
    organs: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    device_used: Mapped[Optional[str]] = mapped_column(String(32))
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # 调试信息
    debug_info: Mapped[Optional[dict]] = mapped_column(JSON)

    # DICOM 选择信息
    dicom_selection: Mapped[Optional[dict]] = mapped_column(JSON)

    # 存储路径 (MinIO object keys 或本地路径)
    input_path: Mapped[Optional[str]] = mapped_column(String(512))
    output_path: Mapped[Optional[str]] = mapped_column(String(512))
    converted_nifti: Mapped[Optional[str]] = mapped_column(String(512))
    processed_input: Mapped[Optional[str]] = mapped_column(String(512))
    session_dir: Mapped[Optional[str]] = mapped_column(String(512))

    # 活动状态（用于前端显示）
    activity_status: Mapped[str] = mapped_column(String(32), default="idle")
    activity_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_preview_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    def to_dict(self) -> dict:
        """转换为字典（兼容原 SESSIONS 格式）"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "status": self.status,
            "task": self.task_type,  # 兼容原格式
            "task_type": self.task_type,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "shape": self.shape,
            "organs": self.organs,
            "device_used": self.device_used,
            "error": self.error_message,  # 兼容原格式
            "error_message": self.error_message,
            "debug_info": self.debug_info,
            "dicom_selection": self.dicom_selection,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "converted_nifti": self.converted_nifti,
            "processed_input": self.processed_input,
            "session_dir": self.session_dir,
            "activity_status": self.activity_status,
            "activity_updated_at": self.activity_updated_at.isoformat() if self.activity_updated_at else None,
            "last_preview_at": self.last_preview_at.isoformat() if self.last_preview_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def update_from_dict(self, data: dict) -> None:
        """从字典更新字段"""
        field_mapping = {
            "filename": "filename",
            "file_type": "file_type",
            "status": "status",
            "task": "task_type",
            "task_type": "task_type",
            "progress": "progress",
            "progress_message": "progress_message",
            "shape": "shape",
            "organs": "organs",
            "device_used": "device_used",
            "error": "error_message",
            "error_message": "error_message",
            "debug_info": "debug_info",
            "dicom_selection": "dicom_selection",
            "input_path": "input_path",
            "output_path": "output_path",
            "converted_nifti": "converted_nifti",
            "processed_input": "processed_input",
            "session_dir": "session_dir",
            "activity_status": "activity_status",
        }
        for dict_key, attr_name in field_mapping.items():
            if dict_key in data:
                setattr(self, attr_name, data[dict_key])


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话的上下文管理器"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """初始化数据库表"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ 数据库表初始化完成")


async def check_db_connection() -> bool:
    """检查数据库连接"""
    try:
        async with async_session_maker() as session:
            await session.execute(func.now())
        return True
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return False


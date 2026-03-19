"""
任务服务层
提供任务的 CRUD 操作，支持数据库和内存双模式
"""
import os
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from sqlalchemy import select, func, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession

# 检查是否启用数据库模式
USE_DATABASE = os.getenv("USE_DATABASE", "false").lower() == "true"

if USE_DATABASE:
    from api.database import Task, get_db_session, async_session_maker
    from api.storage import delete_folder as minio_delete_folder
    from api.cache import cache_delete_pattern


class TaskService:
    """任务服务 - 支持数据库和内存双模式"""
    
    def __init__(self, sessions: Dict[str, Any], data_dir: Path):
        """
        初始化任务服务
        
        Args:
            sessions: 内存中的 session 字典（兼容现有代码）
            data_dir: 数据目录
        """
        self.sessions = sessions
        self.data_dir = data_dir
    
    async def list_tasks(
        self,
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        获取任务列表
        
        Returns:
            (任务列表, 总数)
        """
        if USE_DATABASE:
            return await self._list_tasks_db(page, page_size, status)
        else:
            return self._list_tasks_memory(page, page_size, status)
    
    def _list_tasks_memory(
        self,
        page: int,
        page_size: int,
        status: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """从内存获取任务列表"""
        # 转换为列表并排序
        tasks = []
        for session_id, session in self.sessions.items():
            task = {
                "session_id": session_id,
                "filename": session.get("filename", "unknown"),
                "file_type": session.get("file_type"),
                "status": session.get("status", "unknown"),
                "task_type": session.get("task", "total"),
                "shape": session.get("shape"),
                "organs": session.get("organs"),
                "device_used": session.get("device_used"),
                "error_message": session.get("error"),
                "created_at": session.get("created_at"),
                "completed_at": None,
            }
            # 过滤状态
            if status and task["status"] != status:
                continue
            tasks.append(task)
        
        # 按创建时间倒序
        tasks.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        
        total = len(tasks)
        start = (page - 1) * page_size
        end = start + page_size
        
        return tasks[start:end], total
    
    async def _list_tasks_db(
        self,
        page: int,
        page_size: int,
        status: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """从数据库获取任务列表"""
        async with get_db_session() as session:
            # 构建查询
            query = select(Task)
            count_query = select(func.count(Task.id))
            
            if status:
                query = query.where(Task.status == status)
                count_query = count_query.where(Task.status == status)
            
            # 排序和分页
            query = query.order_by(desc(Task.created_at))
            query = query.offset((page - 1) * page_size).limit(page_size)
            
            # 执行查询
            result = await session.execute(query)
            tasks = [task.to_dict() for task in result.scalars().all()]
            
            count_result = await session.execute(count_query)
            total = count_result.scalar() or 0
            
            return tasks, total
    
    async def get_task(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取单个任务详情"""
        if USE_DATABASE:
            return await self._get_task_db(session_id)
        else:
            return self._get_task_memory(session_id)
    
    def _get_task_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """从内存获取任务"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return {
            "session_id": session_id,
            "filename": session.get("filename", "unknown"),
            "file_type": session.get("file_type"),
            "status": session.get("status", "unknown"),
            "task_type": session.get("task", "total"),
            "shape": session.get("shape"),
            "organs": session.get("organs"),
            "device_used": session.get("device_used"),
            "error_message": session.get("error"),
            "dicom_selection": session.get("dicom_selection"),
            "created_at": session.get("created_at"),
        }
    
    async def _get_task_db(self, session_id: str) -> Optional[Dict[str, Any]]:
        """从数据库获取任务"""
        async with get_db_session() as session:
            result = await session.execute(
                select(Task).where(Task.session_id == session_id)
            )
            task = result.scalar_one_or_none()
            return task.to_dict() if task else None

    async def delete_task(self, session_id: str) -> bool:
        """
        删除单个任务（硬删除）

        删除：
        1. 数据库记录 / 内存记录
        2. 本地文件目录
        3. MinIO 对象（如果使用）
        4. Redis 缓存（如果使用）
        """
        if USE_DATABASE:
            return await self._delete_task_db(session_id)
        else:
            return self._delete_task_memory(session_id)

    def _delete_task_memory(self, session_id: str) -> bool:
        """从内存删除任务"""
        if session_id not in self.sessions:
            return False

        # 删除本地文件
        session_dir = self.data_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)

        # 从内存删除
        del self.sessions[session_id]

        # 更新持久化文件
        from api.config import delete_session_file
        delete_session_file(session_id)

        return True

    async def _delete_task_db(self, session_id: str) -> bool:
        """从数据库删除任务"""
        async with get_db_session() as session:
            # 先获取任务信息
            result = await session.execute(
                select(Task).where(Task.session_id == session_id)
            )
            task = result.scalar_one_or_none()

            if not task:
                return False

            # 删除本地文件
            if task.session_dir:
                session_dir = Path(task.session_dir)
                if session_dir.exists():
                    shutil.rmtree(session_dir)

            # 删除 MinIO 文件
            try:
                minio_delete_folder(f"{session_id}/")
            except Exception as e:
                print(f"MinIO 删除失败: {e}")

            # 删除 Redis 缓存
            try:
                await cache_delete_pattern(f"task:{session_id}:*")
            except Exception as e:
                print(f"Redis 删除失败: {e}")

            # 删除数据库记录
            await session.execute(
                delete(Task).where(Task.session_id == session_id)
            )

            # 同时从内存删除（兼容性）
            if session_id in self.sessions:
                del self.sessions[session_id]

            return True

    async def batch_delete_tasks(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        批量删除任务

        Returns:
            {"deleted": [...], "failed": [...]}
        """
        deleted = []
        failed = []

        for session_id in session_ids:
            try:
                success = await self.delete_task(session_id)
                if success:
                    deleted.append(session_id)
                else:
                    failed.append({"session_id": session_id, "error": "任务不存在"})
            except Exception as e:
                failed.append({"session_id": session_id, "error": str(e)})

        return {"deleted": deleted, "failed": failed}

    async def get_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        if USE_DATABASE:
            return await self._get_statistics_db()
        else:
            return self._get_statistics_memory()

    def _get_statistics_memory(self) -> Dict[str, Any]:
        """从内存获取统计"""
        stats = {
            "total": len(self.sessions),
            "completed": 0,
            "queued": 0,
            "processing": 0,
            "uploaded": 0,
            "error": 0,
        }

        for session in self.sessions.values():
            status = session.get("status", "unknown")
            if status in stats:
                stats[status] += 1

        return stats

    async def _get_statistics_db(self) -> Dict[str, Any]:
        """从数据库获取统计"""
        async with get_db_session() as session:
            # 总数
            total_result = await session.execute(select(func.count(Task.id)))
            total = total_result.scalar() or 0

            # 按状态统计
            stats_query = select(
                Task.status,
                func.count(Task.id)
            ).group_by(Task.status)

            stats_result = await session.execute(stats_query)
            stats = {"total": total}

            for status, count in stats_result:
                stats[status] = count

            return stats


# 全局任务服务实例（延迟初始化）
_task_service: Optional[TaskService] = None


def get_task_service(sessions: Dict[str, Any], data_dir: Path) -> TaskService:
    """获取任务服务实例"""
    global _task_service
    if _task_service is None:
        _task_service = TaskService(sessions, data_dir)
    return _task_service

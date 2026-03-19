"""
TaskRepository - 任务数据访问层
提供统一的数据库操作接口，支持与原 SESSIONS 字典相同的操作模式
同时集成 Redis 缓存以提高热数据访问性能
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy import select, func, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import Task, get_db_session, async_session_maker
from api.cache import cache_get_json, cache_set_json, cache_delete, cache_delete_pattern

# 缓存过期时间（秒）
CACHE_TTL_TASK = 300  # 任务详情缓存 5 分钟
CACHE_TTL_ACTIVE = 60  # 活跃任务缓存 1 分钟


class TaskRepository:
    """任务仓库 - 提供数据库 CRUD 操作"""
    
    @staticmethod
    def _cache_key(session_id: str) -> str:
        """生成缓存键"""
        return f"task:{session_id}"
    
    @staticmethod
    async def create(session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建新任务
        
        Args:
            session_id: 任务会话ID
            data: 任务数据字典
            
        Returns:
            创建的任务字典
        """
        async with get_db_session() as session:
            task = Task(
                session_id=session_id,
                filename=data.get("filename", "unknown"),
                file_type=data.get("file_type"),
                status=data.get("status", "uploaded"),
                task_type=data.get("task", data.get("task_type", "total")),
                input_path=data.get("input_path"),
                session_dir=data.get("session_dir"),
                progress=data.get("progress", 0),
                progress_message=data.get("progress_message"),
            )
            session.add(task)
            await session.flush()
            result = task.to_dict()
            
        # 写入缓存
        await cache_set_json(TaskRepository._cache_key(session_id), result, CACHE_TTL_TASK)
        return result
    
    @staticmethod
    async def get(session_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取任务详情
        
        Args:
            session_id: 任务会话ID
            use_cache: 是否使用缓存
            
        Returns:
            任务字典，不存在则返回 None
        """
        cache_key = TaskRepository._cache_key(session_id)
        
        # 尝试从缓存获取
        if use_cache:
            cached = await cache_get_json(cache_key)
            if cached:
                return cached
        
        # 从数据库获取
        async with get_db_session() as session:
            result = await session.execute(
                select(Task).where(Task.session_id == session_id)
            )
            task = result.scalar_one_or_none()
            if task:
                data = task.to_dict()
                # 更新缓存
                await cache_set_json(cache_key, data, CACHE_TTL_TASK)
                return data
        return None
    
    @staticmethod
    async def exists(session_id: str) -> bool:
        """检查任务是否存在"""
        async with get_db_session() as session:
            result = await session.execute(
                select(func.count(Task.id)).where(Task.session_id == session_id)
            )
            count = result.scalar() or 0
            return count > 0
    
    @staticmethod
    async def update(session_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        更新任务数据
        
        Args:
            session_id: 任务会话ID
            data: 要更新的字段字典
            
        Returns:
            更新后的任务字典
        """
        async with get_db_session() as session:
            result = await session.execute(
                select(Task).where(Task.session_id == session_id)
            )
            task = result.scalar_one_or_none()
            if not task:
                return None
            
            # 更新字段
            task.update_from_dict(data)
            task.updated_at = datetime.now()
            
            # 处理特殊时间字段
            if data.get("activity_status"):
                task.activity_updated_at = datetime.now()
            if data.get("status") == "completed":
                task.completed_at = datetime.now()
                
            await session.flush()
            result_data = task.to_dict()
        
        # 更新缓存
        cache_key = TaskRepository._cache_key(session_id)
        await cache_set_json(cache_key, result_data, CACHE_TTL_TASK)
        return result_data

    @staticmethod
    async def delete(session_id: str) -> bool:
        """
        删除任务

        Args:
            session_id: 任务会话ID

        Returns:
            是否删除成功
        """
        async with get_db_session() as session:
            result = await session.execute(
                delete(Task).where(Task.session_id == session_id)
            )
            deleted = result.rowcount > 0

        # 删除缓存
        if deleted:
            await cache_delete(TaskRepository._cache_key(session_id))
        return deleted

    @staticmethod
    async def list_tasks(
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        获取任务列表（分页）

        Returns:
            (任务列表, 总数)
        """
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

    @staticmethod
    async def get_active_tasks() -> List[Dict[str, Any]]:
        """获取所有进行中的任务"""
        async with get_db_session() as session:
            result = await session.execute(
                select(Task)
                .where(Task.status.in_(["queued", "uploading", "processing"]))
                .order_by(desc(Task.created_at))
            )
            return [task.to_dict() for task in result.scalars().all()]

    @staticmethod
    async def get_active_sessions() -> List[Dict[str, Any]]:
        """
        获取所有活动状态不为 idle 的会话
        用于任务列表显示"生成预览中"、"导出中"等状态
        """
        async with get_db_session() as session:
            result = await session.execute(
                select(Task)
                .where(Task.activity_status != "idle")
                .where(Task.activity_status.isnot(None))
            )
            return [task.to_dict() for task in result.scalars().all()]

    @staticmethod
    async def get_statistics() -> Dict[str, Any]:
        """获取任务统计信息"""
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

    @staticmethod
    async def batch_delete(session_ids: List[str]) -> Dict[str, Any]:
        """
        批量删除任务

        Returns:
            {"deleted": [...], "failed": [...]}
        """
        deleted = []
        failed = []

        for session_id in session_ids:
            try:
                success = await TaskRepository.delete(session_id)
                if success:
                    deleted.append(session_id)
                else:
                    failed.append({"session_id": session_id, "error": "任务不存在"})
            except Exception as e:
                failed.append({"session_id": session_id, "error": str(e)})

        return {"deleted": deleted, "failed": failed}


# 便捷函数，用于替代原 SESSIONS 字典操作
async def get_task(session_id: str) -> Optional[Dict[str, Any]]:
    """获取任务（便捷函数）"""
    return await TaskRepository.get(session_id)


async def task_exists(session_id: str) -> bool:
    """检查任务是否存在（便捷函数）"""
    return await TaskRepository.exists(session_id)


async def create_task(session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """创建任务（便捷函数）"""
    return await TaskRepository.create(session_id, data)


async def update_task(session_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """更新任务（便捷函数）"""
    return await TaskRepository.update(session_id, data)


async def delete_task(session_id: str) -> bool:
    """删除任务（便捷函数）"""
    return await TaskRepository.delete(session_id)

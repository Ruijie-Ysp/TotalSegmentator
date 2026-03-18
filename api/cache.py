"""
Redis 缓存服务
用于缓存热数据和管理任务队列
"""
import os
import json
from pathlib import Path
from typing import Optional, Any
from datetime import timedelta

# 确保环境变量已加载
from dotenv import load_dotenv
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import redis.asyncio as redis

# Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Redis 连接池（延迟初始化）
_redis_pool: Optional[redis.ConnectionPool] = None


def get_redis_pool() -> redis.ConnectionPool:
    """获取 Redis 连接池"""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
            max_connections=20,
        )
    return _redis_pool


def get_redis_client() -> redis.Redis:
    """获取 Redis 客户端"""
    return redis.Redis(connection_pool=get_redis_pool())


async def cache_get(key: str) -> Optional[str]:
    """获取缓存值"""
    try:
        client = get_redis_client()
        value = await client.get(key)
        return value
    except Exception as e:
        print(f"Redis GET 失败: {e}")
        return None


async def cache_set(key: str, value: str, expire_seconds: Optional[int] = None) -> bool:
    """设置缓存值"""
    try:
        client = get_redis_client()
        if expire_seconds:
            await client.setex(key, expire_seconds, value)
        else:
            await client.set(key, value)
        return True
    except Exception as e:
        print(f"Redis SET 失败: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """删除缓存"""
    try:
        client = get_redis_client()
        await client.delete(key)
        return True
    except Exception as e:
        print(f"Redis DELETE 失败: {e}")
        return False


async def cache_get_json(key: str) -> Optional[Any]:
    """获取 JSON 缓存值"""
    value = await cache_get(key)
    if value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


async def cache_set_json(key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
    """设置 JSON 缓存值"""
    try:
        json_str = json.dumps(value, ensure_ascii=False)
        return await cache_set(key, json_str, expire_seconds)
    except (TypeError, ValueError) as e:
        print(f"JSON 序列化失败: {e}")
        return False


async def cache_delete_pattern(pattern: str) -> int:
    """删除匹配模式的所有缓存"""
    try:
        client = get_redis_client()
        keys = []
        async for key in client.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            await client.delete(*keys)
        return len(keys)
    except Exception as e:
        print(f"Redis 批量删除失败: {e}")
        return 0


async def check_redis_connection() -> bool:
    """检查 Redis 连接"""
    try:
        client = get_redis_client()
        await client.ping()
        return True
    except Exception as e:
        print(f"Redis 连接检查失败: {e}")
        return False


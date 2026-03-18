"""
MinIO 对象存储服务
用于存储 CT 影像和分割结果等大文件
"""
import os
import io
from typing import Optional, BinaryIO
from pathlib import Path

from minio import Minio
from minio.error import S3Error

# MinIO 配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "totalseg-data")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# MinIO 客户端（延迟初始化）
_minio_client: Optional[Minio] = None


def get_minio_client() -> Minio:
    """获取 MinIO 客户端"""
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
    return _minio_client


def init_minio_bucket() -> bool:
    """初始化 MinIO bucket"""
    try:
        client = get_minio_client()
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            print(f"✓ MinIO bucket '{MINIO_BUCKET}' 创建成功")
        else:
            print(f"✓ MinIO bucket '{MINIO_BUCKET}' 已存在")
        return True
    except S3Error as e:
        print(f"MinIO 初始化失败: {e}")
        return False
    except Exception as e:
        print(f"MinIO 连接失败: {e}")
        return False


def upload_file(object_name: str, file_path: str) -> bool:
    """上传文件到 MinIO"""
    try:
        client = get_minio_client()
        client.fput_object(MINIO_BUCKET, object_name, file_path)
        return True
    except S3Error as e:
        print(f"上传文件失败: {e}")
        return False


def upload_bytes(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
    """上传字节数据到 MinIO"""
    try:
        client = get_minio_client()
        client.put_object(
            MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return True
    except S3Error as e:
        print(f"上传数据失败: {e}")
        return False


def download_file(object_name: str, file_path: str) -> bool:
    """从 MinIO 下载文件"""
    try:
        client = get_minio_client()
        client.fget_object(MINIO_BUCKET, object_name, file_path)
        return True
    except S3Error as e:
        print(f"下载文件失败: {e}")
        return False


def get_file_stream(object_name: str) -> Optional[BinaryIO]:
    """获取文件流"""
    try:
        client = get_minio_client()
        response = client.get_object(MINIO_BUCKET, object_name)
        return response
    except S3Error as e:
        print(f"获取文件流失败: {e}")
        return None


def delete_file(object_name: str) -> bool:
    """删除 MinIO 中的文件"""
    try:
        client = get_minio_client()
        client.remove_object(MINIO_BUCKET, object_name)
        return True
    except S3Error as e:
        print(f"删除文件失败: {e}")
        return False


def delete_folder(prefix: str) -> int:
    """删除 MinIO 中指定前缀的所有文件"""
    try:
        client = get_minio_client()
        objects = client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
        count = 0
        for obj in objects:
            client.remove_object(MINIO_BUCKET, obj.object_name)
            count += 1
        return count
    except S3Error as e:
        print(f"删除文件夹失败: {e}")
        return 0


def check_minio_connection() -> bool:
    """检查 MinIO 连接"""
    try:
        client = get_minio_client()
        client.list_buckets()
        return True
    except Exception as e:
        print(f"MinIO 连接检查失败: {e}")
        return False


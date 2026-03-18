#!/usr/bin/env python
"""设置 MinIO bucket 权限"""
import os
from minio import Minio
import json

# MinIO 配置
endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:11000')
access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
bucket_name = os.getenv('MINIO_BUCKET', 'totalseg-data')

print(f"连接 MinIO: {endpoint}")

client = Minio(
    endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False
)

# 列出现有 buckets
print('\n现有 Buckets:')
for bucket in client.list_buckets():
    print(f'  - {bucket.name}')

# 创建 bucket（如果不存在）
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)
    print(f'\n✓ 已创建 bucket: {bucket_name}')
else:
    print(f'\n✓ bucket 已存在: {bucket_name}')

# 设置公开读取策略
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
        }
    ]
}

client.set_bucket_policy(bucket_name, json.dumps(policy))
print(f'✓ 已设置 {bucket_name} 公开读取权限')

# 列出 bucket 内容
print(f'\n{bucket_name} 中的对象:')
objects = list(client.list_objects(bucket_name, recursive=True))
if objects:
    for obj in objects[:10]:
        print(f'  - {obj.object_name} ({obj.size} bytes)')
    if len(objects) > 10:
        print(f'  ... 共 {len(objects)} 个对象')
else:
    print('  (空)')

print('\n✓ MinIO 配置完成!')


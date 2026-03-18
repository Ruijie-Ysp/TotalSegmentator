#!/usr/bin/env python
"""测试 Label Studio 任务导入"""
import os
os.environ['LABEL_STUDIO_URL'] = 'http://localhost:8888'
os.environ['LABEL_STUDIO_API_KEY'] = '9e3df408cd28709a7d2691bed750cbf16996d32b'

from api.labelstudio_service import get_labelstudio_service, LABEL_STUDIO_SDK_V2

print(f"SDK v2: {LABEL_STUDIO_SDK_V2}")

service = get_labelstudio_service()
service.connect()

# 测试导入任务到项目 42
project_id = 42
tasks = [
    {
        "data": {
            "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            "question": "请描述这张图像中的内容。",
            "ai_answer": "这是一张测试图像。",
            "session_id": "test",
            "filename": "test.nii.gz",
            "axis": "axial",
            "slice_idx": 0
        }
    }
]

print(f"准备导入 {len(tasks)} 个任务到项目 {project_id}")

try:
    if LABEL_STUDIO_SDK_V2:
        result = service.client.projects.import_tasks(id=project_id, request=tasks)
        print(f"导入成功! 结果: {result}")
    else:
        project = service.client.get_project(project_id)
        project.import_tasks(tasks)
        print("导入成功!")
except Exception as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()


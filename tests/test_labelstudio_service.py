"""
Label Studio 服务测试
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLabelStudioService:
    """Label Studio 服务测试"""
    
    def test_import_service(self):
        """测试服务模块导入"""
        from api.labelstudio_service import (
            LabelStudioService,
            generate_slice_images,
            create_labelstudio_tasks,
            get_vqa_label_config,
            get_labelstudio_service
        )
        assert LabelStudioService is not None
        assert generate_slice_images is not None
        assert create_labelstudio_tasks is not None
    
    def test_label_config(self):
        """测试标注配置生成"""
        from api.labelstudio_service import get_vqa_label_config
        
        config = get_vqa_label_config()
        assert "<View>" in config
        assert "BrushLabels" in config
        assert "未知器官" in config
    
    def test_create_tasks(self):
        """测试任务创建"""
        from api.labelstudio_service import create_labelstudio_tasks
        
        slices_info = [
            {
                "slice_idx": 0,
                "axis": "axial",
                "object_name": "test/slice_0.png",
                "url": "s3://bucket/test/slice_0.png",
                "public_url": "http://localhost:9000/bucket/test/slice_0.png"
            },
            {
                "slice_idx": 10,
                "axis": "axial",
                "object_name": "test/slice_10.png",
                "url": "s3://bucket/test/slice_10.png",
                "public_url": "http://localhost:9000/bucket/test/slice_10.png"
            }
        ]
        
        tasks = create_labelstudio_tasks(
            session_id="test-session",
            slices_info=slices_info,
            organs=["liver", "spleen"],
            filename="test.nii.gz",
            use_public_url=True
        )
        
        assert len(tasks) == 2
        assert tasks[0]["data"]["session_id"] == "test-session"
        assert tasks[0]["data"]["slice_idx"] == 0
        assert "predictions" in tasks[0]
    
    def test_service_singleton(self):
        """测试服务单例"""
        from api.labelstudio_service import get_labelstudio_service
        
        service1 = get_labelstudio_service()
        service2 = get_labelstudio_service()
        
        assert service1 is service2


class TestStorageIntegration:
    """存储集成测试"""
    
    def test_minio_connection_check(self):
        """测试 MinIO 连接检查"""
        from api.storage import check_minio_connection
        
        # 这个测试可能失败如果 MinIO 未运行
        result = check_minio_connection()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

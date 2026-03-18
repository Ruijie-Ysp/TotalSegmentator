#!/bin/bash
# TotalSegmentator Viewer 启动脚本

echo "🩻 启动 TotalSegmentator Viewer..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python3${NC}"
    exit 1
fi

# 检查 Node.js 环境
if ! command -v npm &> /dev/null; then
    echo -e "${RED}错误: 未找到 npm${NC}"
    exit 1
fi

# 创建 API 数据目录
mkdir -p api_data

echo -e "${YELLOW}[1/2] 启动后端 API (端口 8000)...${NC}"
cd api
# 后台启动 API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
cd ..

# 等待 API 启动
sleep 3

echo -e "${YELLOW}[2/2] 启动前端 (端口 3000)...${NC}"
cd web
npm run dev &
WEB_PID=$!
cd ..

echo ""
echo -e "${GREEN}✅ TotalSegmentator Viewer 已启动!${NC}"
echo ""
echo "  📡 后端 API:  http://localhost:8000"
echo "  🌐 前端界面:  http://localhost:3000"
echo "  📚 API 文档:  http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止所有服务${NC}"

# 捕获退出信号
trap "echo '正在停止服务...'; kill $API_PID $WEB_PID 2>/dev/null; exit" SIGINT SIGTERM

# 等待进程
wait


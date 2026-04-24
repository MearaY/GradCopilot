@echo off
echo ========================================
echo GradCopilot - Agentic RAG 学术助手 (FastAPI 后端)
echo ========================================
echo.
echo 正在启动 FastAPI 服务器...
echo 服务器地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务器
echo.

cd /d "%~dp0"
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

pause

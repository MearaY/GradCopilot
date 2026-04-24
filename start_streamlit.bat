@echo off
echo ========================================
echo GradCopilot - Agentic RAG 学术助手 (Web UI 前端)
echo ========================================
echo.

echo 正在启动 Streamlit 前端...
echo.
echo 请确保后端服务已启动！
echo 如果需要启动后端，请运行: start_fastapi.bat
echo.

cd /d "%~dp0"
streamlit run streamlit_app.py

pause

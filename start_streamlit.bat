@echo off
echo ========================================
echo 论文知识库问答系统 - Streamlit 前端
echo ========================================
echo.

echo 正在启动 Streamlit 前端...
echo.
echo 请确保后端服务已启动！
echo 如果需要启动后端，请运行: start_fastapi.bat
echo.

streamlit run streamlit_app.py

pause

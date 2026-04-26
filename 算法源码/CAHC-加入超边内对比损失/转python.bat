@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title 提取代码上下文 (纯净版)

echo ===================================================
echo ⚠️ 准备扫描当前目录及其子目录下的代码文件 (.py, .yaml, .yml)
echo ⚠️ 提取的内容将汇总并保存到 project_context.md
echo ===================================================
echo.

set /p "choice=是否确认执行？(输入 Y 继续，其他按键取消): "
if /I not "%choice%"=="Y" (
    echo.
    echo 🛑 已取消操作。
    pause
    exit /b
)

set "OUTPUT=project_context.md"
echo # Project Context Summary > "%OUTPUT%"
echo --- >> "%OUTPUT%"
echo. >> "%OUTPUT%"

echo 🚀 正在处理中...

:: 使用 dir 命令递归寻找文件，并排除常见的无关目录
for /f "delims=" %%i in ('dir /s /b *.py *.yaml *.yml') do (
    set "abspath=%%i"
    set "relpath=!abspath:%cd%\=!"
    
    :: 简单的目录过滤（跳过包含 .git, venv, __pycache__ 的路径）
    set "skip="
    echo !abspath! | findstr /i "\\.git\\ \\venv\\ \\env\\ \\__pycache__\\ \\.vscode\\" >nul && set "skip=1"
    
    if not defined skip (
        if /I "%%~nxi" neq "%OUTPUT%" (
            echo ✅ 正在提取: !relpath!
            
            :: 写入文件名
            echo ## File: `!relpath!` >> "%OUTPUT%"
            echo. >> "%OUTPUT%"
            
            :: 根据后缀确定语言
            set "ext=%%~xi"
            if /I "!ext!"==".py" ( echo ```python >> "%OUTPUT%" ) else ( echo ```yaml >> "%OUTPUT%" )
            
            :: 写入文件内容
            type "%%i" >> "%OUTPUT%"
            
            :: 闭合代码块
            echo. >> "%OUTPUT%"
            echo ``` >> "%OUTPUT%"
            echo. >> "%OUTPUT%"
        )
    )
)

echo.
echo ✨ 汇总完成！文件已保存为 %OUTPUT%
pause
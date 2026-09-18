# package_project.py - 简单的一键打包脚本
import os
import sys
import shutil
import zipfile
import json
from pathlib import Path


def main():
    """主打包函数"""
    print("=" * 60)
    print("🤖 AI聊天机器人项目打包工具")
    print("=" * 60)

    # 获取当前目录
    current_dir = Path(__file__).parent.absolute()
    print(f"📁 当前项目目录: {current_dir}")

    # 创建临时打包目录
    temp_dir = current_dir / "_package_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)

    # 检查关键文件是否存在
    essential_files = [
        "scripts/api_server.py",
        "models/Qwen2-0.5B-Instruct",
        "models/finetuned_model",
        "requirements.txt"
    ]

    print("\n🔍 检查项目文件...")
    for file_path in essential_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (文件不存在)")

    # 复制核心文件
    print("\n📋 复制核心文件...")

    # 1. 复制models目录
    models_src = current_dir / "models"
    models_dst = temp_dir / "models"

    if models_src.exists():
        # 只复制必要的模型文件
        for model_dir in ["Qwen2-0.5B-Instruct", "finetuned_model"]:
            src_dir = models_src / model_dir
            if src_dir.exists():
                print(f"  正在复制模型: {model_dir}")

                # 跳过.git等不必要文件
                def ignore_func(src, names):
                    ignore = ['.git', '__pycache__', '.gitignore', '.pytest_cache']
                    return [n for n in names if any(i in n for i in ignore)]

                shutil.copytree(src_dir, models_dst / model_dir, ignore=ignore_func)

    # 2. 复制scripts目录
    scripts_src = current_dir / "scripts"
    scripts_dst = temp_dir / "scripts"

    if scripts_src.exists():
        print("  复制scripts目录...")
        shutil.copytree(scripts_src, scripts_dst,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    # 3. 复制其他文件
    other_files = [
        "requirements.txt",
        "TRAINING_SUCCESS.txt",
        "visualization_new.py",
        "main.py"
    ]

    for file_name in other_files:
        file_path = current_dir / file_name
        if file_path.exists():
            shutil.copy2(file_path, temp_dir / file_name)
            print(f"  复制: {file_name}")

    # 创建启动脚本
    print("\n⚡ 创建启动脚本...")

    # Windows启动脚本
    bat_content = '''@echo off
chcp 65001 > nul
title 🤖 AI聊天机器人便携版
color 0A

echo ========================================
echo      AI聊天机器人 - 一键启动
echo ========================================
echo.

echo 🔍 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python，请先安装Python 3.8+
    echo 📥 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python环境正常
echo.

echo 📦 安装依赖包...
python -m pip install --upgrade pip --quiet
python -m pip install transformers torch fastapi uvicorn pydantic --quiet
if errorlevel 1 (
    echo ⚠️  依赖安装失败，尝试继续运行...
)

echo.
echo 🚀 启动AI聊天机器人...
echo 📍 聊天界面: http://localhost:8000
echo 📊 监控面板: http://localhost:8001
echo.

echo 正在启动API服务器...
start "API Server" python scripts/api_server.py

timeout /t 3 /nobreak >nul

echo 正在启动监控面板...
start "Monitor" python visualization_new.py

timeout /t 3 /nobreak >nul

echo 🌐 打开浏览器...
start http://localhost:8000
start http://localhost:8001

echo.
echo ✅ 所有服务已启动！
echo 📝 按Ctrl+C在命令行中停止服务
echo.

pause
'''

    with open(temp_dir / "start.bat", 'w', encoding='utf-8') as f:
        f.write(bat_content)

    # Linux启动脚本
    sh_content = '''#!/bin/bash

echo "========================================"
echo "    AI聊天机器人 - 一键启动"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到Python3，请先安装Python 3.8+"
    echo "📥 下载地址: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python环境正常"
echo ""

# 安装依赖
echo "📦 安装依赖包..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install transformers torch fastapi uvicorn pydantic --quiet

echo ""
echo "🚀 启动AI聊天机器人..."
echo "📍 聊天界面: http://localhost:8000"
echo "📊 监控面板: http://localhost:8001"
echo ""

# 后台启动服务
python3 scripts/api_server.py &
API_PID=$!

sleep 3

python3 visualization_new.py &
MONITOR_PID=$!

sleep 3

# 打开浏览器
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000
    xdg-open http://localhost:8001
elif command -v open &> /dev/null; then
    open http://localhost:8000
    open http://localhost:8001
fi

echo ""
echo "✅ 所有服务已启动！"
echo "📝 运行以下命令停止服务:"
echo "kill $API_PID $MONITOR_PID"
echo ""
echo "按回车键退出..."
read

# 停止服务
kill $API_PID $MONITOR_PID 2>/dev/null
'''

    with open(temp_dir / "start.sh", 'w', encoding='utf-8') as f:
        f.write(sh_content)

    # 设置为可执行
    if sys.platform != 'win32':
        os.chmod(temp_dir / "start.sh", 0o755)

    # 创建使用说明
    readme_content = '''# AI聊天机器人便携版

## 快速开始

### Windows用户：
1. 双击运行 `start.bat`
2. 首次运行会自动安装依赖
3. 浏览器会自动打开聊天界面

### Linux/Mac用户：
1. 打开终端
2. 运行：`chmod +x start.sh`
3. 运行：`./start.sh`

## 访问地址
- 聊天界面：http://localhost:8000
- 监控面板：http://localhost:8001

## 功能特点
- 🤖 本地AI对话聊天
- 🔒 完全离线运行
- ⚡ 一键启动，无需配置
- 📊 实时性能监控

## 系统要求
- Python 3.8+
- 4GB+ 内存
- 5GB+ 磁盘空间

## 注意事项
1. 首次运行需要联网下载依赖
2. 确保防火墙允许访问localhost
3. 不要删除models文件夹

## 技术支持
遇到问题请：
1. 确保Python已正确安装
2. 检查网络连接
3. 重新运行启动脚本
'''

    with open(temp_dir / "README.txt", 'w', encoding='utf-8') as f:
        f.write(readme_content)

    # 创建压缩包
    print("\n📦 创建压缩包...")
    zip_path = current_dir / "AI_ChatBot_Portable.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    # 清理临时目录
    shutil.rmtree(temp_dir)

    # 计算文件大小
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("🎉 打包完成！")
    print("=" * 60)
    print(f"📦 压缩包: {zip_path}")
    print(f"📏 文件大小: {zip_size:.2f} MB")
    print("\n💡 使用方法:")
    print("1. 将压缩包发送给用户")
    print("2. 用户解压后双击 start.bat")
    print("3. 首次运行会自动安装依赖")
    print("4. 访问 http://localhost:8000")
    print("=" * 60)

    # 在Windows资源管理器中打开文件
    if sys.platform == 'win32':
        os.startfile(current_dir)
        print("\n📂 已打开项目文件夹查看压缩包")


if __name__ == "__main__":
    main()
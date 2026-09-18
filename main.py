#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聊天机器人项目主入口 - 解决CS模块依赖问题
"""

import subprocess
import os
import sys

# 添加调试信息
print("========== 启动调试信息 ==========")
print(f"Python解释器路径: {sys.executable}")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print("=================================")


# 模拟CS模块的功能，防止导入错误
class ConfigManager:
    def __init__(self):
        self.config = {
            'data_dir': 'data',
            'model_dir': 'models',
            'results_dir': 'results',
            'language': 'chinese'
        }

    def load_config(self, config_file):
        print(f"加载配置文件: {config_file}")
        return self.config

    def save_config(self, config_file):
        print(f"保存配置文件: {config_file}")
        return True


class EnvironmentManager:
    def __init__(self):
        self.env = os.environ

    def setup_environment(self):
        print("设置环境变量")
        return True


def check_dependencies():
    """
    检查依赖是否安装
    """
    print("\n========== 检查依赖 ==========")
    dependencies = [
        'chatterbot',
        'chatterbot_corpus',
        'spacy',
        'transformers',
        'torch',
        'pandas',
        'numpy',
        'matplotlib',
        'nltk',
        'rouge_score',  # 修改这里，使用正确的包名
        'huggingface_hub'
    ]

    all_ok = True
    for dep in dependencies:
        try:
            imported = __import__(dep)
            print(f"✓ {dep} 已安装，版本: {getattr(imported, '__version__', '未知')}")
        except ImportError as e:
            print(f"✗ {dep} 未安装，错误: {e}")
            all_ok = False

    if all_ok:
        print("\n✅ 所有依赖检查通过!")
    else:
        print("\n❌ 部分依赖未安装，请运行:")
        print("pip install -r requirements.txt")

    return all_ok


def run_script(script_name, script_args=None):
    """
    执行指定的Python脚本
    :param script_name: 脚本文件名
    :param script_args: 脚本参数
    """
    print(f"\n{'=' * 50}")
    print(f"开始执行 {script_name}")
    print(f"{'=' * 50}")

    # 检查脚本文件是否存在
    if not os.path.exists(script_name):
        print(f"❌ 脚本文件不存在: {script_name}")
        return False

    cmd = [sys.executable, script_name]
    if script_args:
        cmd.extend(script_args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        print(result.stdout)
        if result.stderr:
            print(f"警告信息: {result.stderr}")

        print(f"\n{'=' * 50}")
        print(f"{script_name} 执行成功")
        print(f"{'=' * 50}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行 {script_name} 失败，返回码: {e.returncode}")
        print(f"错误信息: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ 执行 {script_name} 发生未知错误: {e}")
        return False


def main():
    """
    主函数 - 处理命令行参数并执行相应任务
    """
    # 初始化配置管理器
    config = ConfigManager()
    env_manager = EnvironmentManager()

    # 设置环境
    env_manager.setup_environment()

    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查不通过，请先安装依赖")
        return

    # 创建必要的目录
    dirs = ['data', 'models', 'results']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"确保目录存在: {dir_name}")

    # 执行实验流程
    print("\n========== 开始执行实验 ==========")

    # 1. 数据处理
    print("\n❶ 数据处理")
    if not run_script('scripts/data_processing.py'):
        print("❌ 数据处理失败，停止执行")
        return

    # 2. 模型训练
    print("\n❷ 模型训练")
    if not run_script('scripts/model_training.py'):
        print("❌ 模型训练失败，停止执行")
        return

    # 3. 模型测试
    print("\n❸ 模型测试")
    if not run_script('scripts/model_testing.py'):
        print("❌ 模型测试失败，停止执行")
        return

    # 4. 结果可视化
    print("\n❹ 结果可视化")
    if not run_script('scripts/visualization.py'):
        print("❌ 可视化失败，停止执行")
        return

    print("\n🎉 整个实验流程执行完成！")
    print("\n📊 实验结果保存在 results/ 目录下")
    print("🤖 模型保存在 models/ 目录下")
    print("\n下一次可以直接运行:")
    print("python scripts/model_testing.py")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序运行时发生错误: {e}")
        import traceback

        traceback.print_exc()
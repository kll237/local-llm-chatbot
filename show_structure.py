import os
from pathlib import Path


def print_directory_tree(start_path, max_depth=4, current_depth=0, prefix=""):
    """打印目录树结构"""
    if current_depth > max_depth:
        return

    # 获取目录内容
    try:
        items = list(os.scandir(start_path))
    except (PermissionError, OSError):
        return

    # 排序：先目录后文件，按字母顺序
    items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

    for i, item in enumerate(items):
        is_last = i == len(items) - 1

        # 当前前缀
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{item.name}")

        # 如果是目录，递归显示
        if item.is_dir():
            extension = "    " if is_last else "│   "
            print_directory_tree(item.path, max_depth, current_depth + 1, prefix + extension)


def main():
    """主函数"""
    print("=" * 60)
    print("项目目录结构")
    print("=" * 60)

    # 获取项目根目录
    project_root = Path(__file__).parent
    print(f"项目根目录: {project_root}")
    print(f"当前工作目录: {Path.cwd()}")
    print()

    # 打印目录树
    print("📁 目录结构:")
    print_directory_tree(project_root, max_depth=3)

    print("\n📊 目录统计:")

    # 统计各目录文件数量
    directories_to_check = [
        ("data", "数据目录"),
        ("models", "模型目录"),
        ("results", "结果目录"),
        ("scripts", "脚本目录"),
    ]

    for dir_name, description in directories_to_check:
        dir_path = project_root / dir_name
        if dir_path.exists():
            files = list(dir_path.rglob("*"))
            files_count = len([f for f in files if f.is_file()])
            dirs_count = len([f for f in files if f.is_dir()])
            print(f"  {description:10s} {dir_name:15s} 文件: {files_count:3d}  子目录: {dirs_count:2d}")
        else:
            print(f"  {description:10s} {dir_name:15s} ❌ 不存在")

    print("\n📁 关键文件检查:")

    # 检查关键文件
    key_files = [
        ("main.py", "主程序"),
        ("requirements.txt", "依赖文件"),
        ("results/processed_train.csv", "训练数据"),
        ("results/processed_test.csv", "测试数据"),
        ("models/Qwen2-0.5B-Instruct/", "原始模型"),
        ("models/finetuned_model/", "微调模型"),
        ("TRAINING_SUCCESS.txt", "训练成功标记"),
    ]

    for file_path, description in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            if full_path.is_dir():
                file_count = len(list(full_path.rglob("*")))
                print(f"  ✓ {description:15s} {file_path:30s} (目录, {file_count}个文件)")
            else:
                size_kb = full_path.stat().st_size / 1024
                print(f"  ✓ {description:15s} {file_path:30s} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {description:15s} {file_path:30s} 不存在")


if __name__ == "__main__":
    main()
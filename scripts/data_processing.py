#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理模块 - 清理、格式化和准备训练数据
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def get_project_root():
    """获取项目根目录"""
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 返回项目根目录（上一级目录）
    return os.path.dirname(script_dir)


def load_data():
    """加载原始数据"""
    print("加载训练数据...")

    # 获取项目根目录
    project_root = get_project_root()

    # 构建绝对路径
    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path = os.path.join(project_root, 'data', 'test.csv')

    # 检查文件是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """预处理数据"""
    print("预处理数据...")

    # 检查数据格式
    print(f"训练集列名: {train_df.columns.tolist()}")
    print(f"测试集列名: {test_df.columns.tolist()}")
    print(f"训练集行数: {len(train_df)}")
    print(f"测试集行数: {len(test_df)}")

    # 检查是否有缺失值
    print(f"训练集缺失值: {train_df.isnull().sum().sum()}")
    print(f"测试集缺失值: {test_df.isnull().sum().sum()}")

    # 去重
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()

    print(f"去重后训练集行数: {len(train_df)}")
    print(f"去重后测试集行数: {len(test_df)}")

    # 统计信息
    print(f"平均输入长度: {train_df['input'].apply(len).mean():.2f}")
    print(f"平均回复长度: {train_df['response'].apply(len).mean():.2f}")

    return train_df, test_df


def save_processed_data(train_df, test_df):
    """保存处理后的数据"""
    print("保存处理后的数据...")

    # 获取项目根目录
    project_root = get_project_root()

    # 构建results目录路径
    results_dir = os.path.join(project_root, 'results')

    # 确保results目录存在
    os.makedirs(results_dir, exist_ok=True)

    # 保存处理后的数据
    train_path = os.path.join(results_dir, 'processed_train.csv')
    test_path = os.path.join(results_dir, 'processed_test.csv')

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')

    print(f"处理后的数据已保存到: {train_path}")
    print(f"处理后的数据已保存到: {test_path}")
    print("数据处理完成!")


def main():
    try:
        # 加载数据
        train_df, test_df = load_data()

        # 预处理数据
        processed_train_df, processed_test_df = preprocess_data(train_df, test_df)

        # 保存处理后的数据
        save_processed_data(processed_train_df, processed_test_df)

        return 0

    except Exception as e:
        print(f"错误：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
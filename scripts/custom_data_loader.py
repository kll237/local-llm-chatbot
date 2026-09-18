#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义数据集加载脚本 - 无需chatterbot_corpus
"""

import os
import pandas as pd


def load_custom_dataset(data_dir='data'):
    """
    加载自定义数据集
    :param data_dir: 数据集目录
    :return: 处理后的对话数据集
    """
    # 如果没有自定义数据集，使用示例数据
    if not os.path.exists(os.path.join(data_dir, 'train.csv')):
        # 创建示例数据
        data = {
            'input': ['你好', '天气如何', '再见'],
            'response': ['嗨', '今天天气不错', '再见'],
            'category': ['general', 'general', 'general']
        }
        dataset = pd.DataFrame(data)
        dataset.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
        dataset.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

    # 加载数据集
    train_dataset = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_dataset = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    return train_dataset, test_dataset


if __name__ == '__main__':
    # 创建数据目录
    os.makedirs('data', exist_ok=True)

    # 加载自定义数据集
    train_dataset, test_dataset = load_custom_dataset()

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print("\n示例数据:")
    print(train_dataset.head())
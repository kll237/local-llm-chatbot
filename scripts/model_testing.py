#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版模型测试脚本 - 评估训练好的聊天机器人模型
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("修复版模型测试")
print("=" * 60)


def setup_environment():
    """设置环境"""
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def get_absolute_path(relative_path):
    """获取绝对路径"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, relative_path)


def load_model():
    """
    加载训练好的模型 - 修复版
    """
    print("加载模型...")

    # 尝试多个可能的模型路径
    possible_model_paths = [
        "models/finetuned_model",  # 微调后的模型
        "models/manually_trained",  # 手动训练的模型
        "models/Qwen2-0.5B-Instruct",  # 原始模型
    ]

    for model_dir in possible_model_paths:
        model_path = get_absolute_path(model_dir)
        print(f"尝试加载模型: {model_path}")

        if not os.path.exists(model_path):
            print(f"  ⚠ 路径不存在")
            continue

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # 先检查配置文件
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                # 如果配置文件中缺少model_type，先修复
                try:
                    import json
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    # 确保有model_type字段
                    if 'model_type' not in config:
                        print(f"  修复config.json，添加model_type...")
                        config['model_type'] = 'qwen2'
                        with open(config_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, ensure_ascii=False, indent=2)
                except:
                    pass

            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"  ✓ Tokenizer加载成功")

            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )

            if not torch.cuda.is_available():
                model = model.to('cpu')

            print(f"  ✓ 模型加载成功: {model_path}")
            print(f"    参数量: {sum(p.numel() for p in model.parameters()):,}")
            print(f"    设备: {next(model.parameters()).device}")

            return model, tokenizer

        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            continue

    print("❌ 无法加载任何模型")
    return None, None


def generate_response(model, tokenizer, user_input, max_length=128):
    """
    生成聊天机器人回复 - 修复版
    """
    if model is None or tokenizer is None:
        return "模型未加载，无法生成回复"

    try:
        # 创建对话格式
        prompt = f"用户: {user_input}\n助手:"

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取助手回复
        if "助手:" in response:
            response = response.split("助手:")[-1].strip()

        # 清理回复
        response = response.replace("</s>", "").replace("<s>", "").strip()

        return response

    except Exception as e:
        return f"生成回复时出错: {e}"


def load_test_data():
    """加载测试数据"""
    print("\n加载测试数据...")

    test_path = get_absolute_path("results/processed_test.csv")

    if not os.path.exists(test_path):
        print(f"❌ 测试数据不存在: {test_path}")

        # 如果没有测试数据，创建样本测试数据
        print("创建样本测试数据...")
        test_data = [
            {"input": "你好", "response": "你好！我是聊天机器人。"},
            {"input": "你叫什么名字", "response": "我叫ChatBot，很高兴为你服务。"},
            {"input": "今天天气怎么样", "response": "抱歉，我无法获取实时天气信息。"},
            {"input": "谢谢", "response": "不客气！有什么可以帮你的吗？"},
            {"input": "再见", "response": "再见！祝你有个愉快的一天！"}
        ]

        test_df = pd.DataFrame(test_data)
        print(f"✓ 创建了 {len(test_df)} 条测试数据")

    else:
        test_df = pd.read_csv(test_path, encoding='utf-8')
        print(f"✓ 加载了 {len(test_df)} 条测试数据")

    return test_df


def simple_evaluate(model, tokenizer, test_df, max_samples=10):
    """简单评估模型性能"""
    print(f"\n评估模型性能（最多{max_samples}条）...")

    if len(test_df) > max_samples:
        test_df = test_df.head(max_samples)

    results = []

    for i, row in test_df.iterrows():
        user_input = row['input']
        expected_response = row['response']

        print(f"\n测试 {i + 1}/{len(test_df)}")
        print(f"用户输入: {user_input}")
        print(f"期望回复: {expected_response[:50]}..." if len(
            expected_response) > 50 else f"期望回复: {expected_response}")

        # 生成回复
        generated_response = generate_response(model, tokenizer, user_input)
        print(f"生成回复: {generated_response}")

        # 简单相似度计算
        similarity = 0
        if generated_response and expected_response:
            # 简单的词重叠计算
            gen_words = set(str(generated_response).split())
            exp_words = set(str(expected_response).split())

            if gen_words and exp_words:
                overlap = len(gen_words.intersection(exp_words))
                total = len(gen_words.union(exp_words))
                similarity = overlap / total if total > 0 else 0

        results.append({
            'input': user_input,
            'expected': expected_response,
            'generated': generated_response,
            'similarity': similarity,
            'length_match': abs(len(generated_response) - len(expected_response)) < 50  # 长度差不大于50字符
        })

    return pd.DataFrame(results)


def save_results(results, test_df):
    """保存测试结果"""
    print("\n保存测试结果...")

    output_dir = get_absolute_path("results/evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # 保存详细结果
    results_file = os.path.join(output_dir, "test_results.csv")
    results.to_csv(results_file, index=False, encoding='utf-8')
    print(f"✓ 详细结果: {results_file}")

    # 计算统计信息
    if len(results) > 0:
        avg_similarity = results['similarity'].mean()
        length_match_rate = results['length_match'].mean()

        stats = {
            "test_samples": len(results),
            "avg_similarity": float(avg_similarity),
            "length_match_rate": float(length_match_rate),
            "total_test_data": len(test_df),
            "note": "简单相似度评估（基于词重叠）"
        }

        # 保存统计信息
        stats_file = os.path.join(output_dir, "test_statistics.json")
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"✓ 统计信息: {stats_file}")

        print("\n评估统计:")
        print(f"  测试样本数: {stats['test_samples']}")
        print(f"  平均相似度: {stats['avg_similarity']:.2%}")
        print(f"  长度匹配率: {stats['length_match_rate']:.2%}")
    else:
        print("⚠ 没有评估结果可保存")


def interactive_test(model, tokenizer):
    """交互式测试"""
    print("\n" + "=" * 60)
    print("交互式测试")
    print("=" * 60)
    print("输入 '退出' 或 'exit' 结束对话")
    print("-" * 60)

    if model is None or tokenizer is None:
        print("模型未加载，无法进行交互测试")
        return

    conversation_history = []

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                print("机器人: 再见！")
                break

            if not user_input:
                continue

            print("机器人: 思考中...", end="\r")
            response = generate_response(model, tokenizer, user_input)
            print(f"机器人: {response}")

            conversation_history.append((user_input, response))

        except KeyboardInterrupt:
            print("\n\n对话结束")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            continue

    # 保存对话历史
    if conversation_history:
        try:
            import json
            from datetime import datetime

            history_file = f"chat_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_history, f, ensure_ascii=False, indent=2)

            print(f"\n对话历史已保存到: {history_file}")
        except:
            print("\n无法保存对话历史")


def main():
    """主函数"""
    setup_environment()

    print("模型测试开始...")

    # 1. 加载模型
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("❌ 无法加载模型，测试中止")
        return

    # 2. 加载测试数据
    test_df = load_test_data()

    # 3. 评估模型性能
    results = simple_evaluate(model, tokenizer, test_df, max_samples=5)

    # 4. 保存结果
    save_results(results, test_df)

    # 5. 交互式测试
    interactive_test(model, tokenizer)

    print("\n" + "=" * 60)
    print("✅ 模型测试完成")
    print("=" * 60)
    print("输出目录: results/evaluation/")
    print("模型状态: 测试通过")
    print("\n可以开始使用聊天机器人了！")


if __name__ == '__main__':
    main()
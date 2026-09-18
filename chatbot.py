# chatbot.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class ChatBot:
    def __init__(self, model_path='models/finetuned_model'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.conversation_history = []

    def generate_reply(self, user_input):
        # 添加用户输入到历史
        self.conversation_history.append({'role': 'user', 'content': user_input})

        # 构建prompt
        prompt = ''
        for message in self.conversation_history:
            prompt += f"{message['role']}: {message['content']}\n"
        prompt += 'assistant:'

        # 生成回复
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        reply = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split('assistant:')[-1].strip()

        # 添加回复到历史
        self.conversation_history.append({'role': 'assistant', 'content': reply})
        return reply


if __name__ == '__main__':
    bot = ChatBot()
    print("AI小助手: 你好！我是你的AI小助手。请问我可以帮你什么？")
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("AI小助手: 再见！")
            break
        reply = bot.generate_reply(user_input)
        print(f"AI小助手: {reply}")
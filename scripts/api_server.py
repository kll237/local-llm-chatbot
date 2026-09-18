# api_server.py - 针对 Qwen2-0.5B 优化的完整版本
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import os
import time
import re

# 配置
# 模型路径：相对项目根目录自动解析，避免硬编码绝对路径，便于克隆后直接运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'Qwen2-0.5B-Instruct')

# 检查模型路径
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ 警告: 模型路径不存在: {MODEL_PATH}")
    print("请确保模型文件已下载到该目录")

# 全局变量
tokenizer = None
model = None
model_loaded = False


# 数据模型
class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class ChatResponse(BaseModel):
    reply: str


# 创建 FastAPI 应用
app = FastAPI(
    title="Qwen2-0.5B 聊天机器人 API",
    description="本地部署的聊天机器人服务",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 自定义文档页面 ==========
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )


# ========== 加载模型函数 ==========
def load_model():
    """加载 Qwen2 模型"""
    global tokenizer, model, model_loaded

    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ 错误: 模型路径不存在: {MODEL_PATH}")
            return False

        print(f"正在加载 Qwen2-0.5B 模型: {MODEL_PATH}")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
            padding_side='left'  # 对于生成，建议左侧填充
        )

        # 确保有 pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = '<|endoftext|>'

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        # 设置为评估模式
        model.eval()

        # 移动到设备
        if torch.cuda.is_available():
            model = model.to('cuda')
            print(f"✅ 模型加载到 GPU")
        else:
            model = model.to('cpu')
            print(f"✅ 模型加载到 CPU")

        model_loaded = True
        print(f"✅ 模型加载成功！")
        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ========== 启动时加载模型 ==========
@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    if os.path.exists(MODEL_PATH):
        success = load_model()
        if success:
            print("✅ 模型初始化完成")
        else:
            print("❌ 模型初始化失败")
    else:
        print("⚠️ 模型文件不存在，跳过加载")


# ========== 辅助函数 ==========
def build_qwen2_prompt(messages):
    """为 Qwen2 模型构建正确的提示格式"""
    prompt = ""

    # Qwen2 的标准格式
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    # 添加 assistant 的开始标记
    prompt += "<|im_start|>assistant\n"
    return prompt


def clean_response(text, prompt):
    """清理模型响应"""
    # 移除 prompt 部分
    if prompt in text:
        response = text[len(prompt):]
    else:
        response = text

    # 移除特殊标记
    special_tokens = [
        '<|im_start|>', '<|im_end|>', '<|endoftext|>',
        'system\n', 'user\n', 'assistant\n',
        'System:', 'User:', 'Assistant:',
        '系统:', '用户:', '助手:'
    ]

    for token in special_tokens:
        response = response.replace(token, '')

    # 移除多余的空格和换行
    response = re.sub(r'\n\s*\n', '\n\n', response)  # 多个空行合并为一个
    response = response.strip()

    return response


# ========== 路由定义 ==========

@app.get("/")
async def root():
    """API首页"""
    return {
        "service": "Qwen2-0.5B 聊天机器人 API",
        "status": "running",
        "model_loaded": model_loaded,
        "model_info": "Qwen2-0.5B-Instruct (小型模型，适合简单对话)",
        "endpoints": {
            "GET /": "此页面",
            "GET /health": "健康检查",
            "GET /info": "服务器信息",
            "POST /api/chat": "聊天接口",
            "GET /test-page": "完整测试页面",
            "GET /docs": "API文档",
            "POST /api/test": "简单测试接口"
        },
        "note": "注意：0.5B模型较小，回答可能不够准确"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": time.time(),
        "message": "服务正常运行" if model_loaded else "服务运行但模型未加载"
    }


@app.get("/info")
async def server_info():
    """服务器信息"""
    return {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": model_loaded,
        "model_name": "Qwen2-0.5B-Instruct",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(model.device) if model else "none"
    }


# ========== 核心聊天接口 ==========

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口 - 针对 Qwen2-0.5B 优化"""

    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请检查模型文件")

    try:
        start_time = time.time()

        # 1. 准备消息
        messages = []

        # 如果没有 system 消息，添加一个默认的
        has_system = any(msg.role == "system" for msg in request.messages)
        if not has_system:
            messages.append(Message(
                role="system",
                content="你是一个有帮助的AI助手。请用中文简洁明了地回答用户的问题。如果不知道答案，请诚实地说明。"
            ))

        # 添加用户消息（只保留最近的2条用户消息，避免太长）
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        recent_user_messages = user_messages[-2:] if len(user_messages) > 2 else user_messages

        for msg in recent_user_messages:
            messages.append(msg)

        # 2. 构建 Qwen2 格式的 prompt
        prompt = build_qwen2_prompt(messages)

        print(f"📝 Prompt预览: {prompt[:200]}...")

        # 3. 编码
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )

        # 4. 移动到正确的设备
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 5. 生成参数（针对小模型优化）
        max_new_tokens = min(request.max_tokens, 300)  # 小模型限制长度
        temperature = max(min(request.temperature, 1.0), 0.1)  # 合理范围

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=request.top_p,
                do_sample=True,
                repetition_penalty=1.2,  # 防止重复
                no_repeat_ngram_size=3,  # 防止3-gram重复
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
                early_stopping=True
            )

        # 6. 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 7. 清理响应
        reply = clean_response(generated_text, prompt)

        # 8. 后处理
        if not reply or len(reply.strip()) < 3:
            # 如果回复太短或无意义，提供默认回复
            last_user_message = recent_user_messages[-1].content if recent_user_messages else "这个问题"
            reply = f"关于'{last_user_message}'，这是一个很好的问题。作为一个小型AI模型，我的回答可能不够详尽，建议你：\n1. 查阅相关文档\n2. 搜索更多资料\n3. 尝试询问更具体的问题"

        # 限制回复长度
        reply = reply[:500]

        elapsed = time.time() - start_time
        print(f"✅ 生成完成 - 耗时: {elapsed:.2f}s, 回复长度: {len(reply)}")
        print(f"🤖 回复: {reply[:100]}...")

        return ChatResponse(reply=reply)

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU内存不足，尝试减少max_tokens")
    except Exception as e:
        print(f"❌ 生成错误: {str(e)}")
        import traceback
        traceback.print_exc()

        # 返回友好的错误信息
        error_msg = f"生成失败: {str(e)[:100]}"
        return ChatResponse(reply=f"抱歉，处理请求时出现错误。{error_msg}")


# ========== 简单测试接口 ==========

@app.post("/api/test")
async def test_chat():
    """简单测试接口"""
    test_request = ChatRequest(
        messages=[Message(role="user", content="你好，请介绍一下你自己")],
        max_tokens=100,
        temperature=0.7
    )

    try:
        response = await chat(test_request)
        return {
            "test": "success",
            "request": "你好，请介绍一下你自己",
            "response": response.reply,
            "model_loaded": model_loaded,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "test": "failed",
            "error": str(e),
            "model_loaded": model_loaded
        }


# ========== 前端测试页面 ==========

@app.get("/test-page")
async def test_page():
    """完整的前端测试页面"""
    html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Qwen2-0.5B 聊天测试</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Microsoft YaHei', sans-serif;
                background: #f0f2f5;
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #1890ff 0%, #096dd9 100%);
                color: white;
                padding: 25px 30px;
            }
            .header h1 {
                font-size: 28px;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .header h1:before {
                content: "🤖";
                font-size: 32px;
            }
            .model-info {
                font-size: 14px;
                opacity: 0.9;
                background: rgba(255,255,255,0.1);
                padding: 8px 15px;
                border-radius: 20px;
                display: inline-block;
                margin-top: 5px;
            }
            .status-bar {
                padding: 15px 30px;
                background: #fafafa;
                border-bottom: 1px solid #e8e8e8;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .status {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .status-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #52c41a;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .controls {
                display: flex;
                gap: 10px;
            }
            .btn {
                padding: 6px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }
            .btn-primary {
                background: #1890ff;
                color: white;
            }
            .btn-secondary {
                background: #f5f5f5;
                color: #666;
                border: 1px solid #d9d9d9;
            }
            .chat-container {
                display: flex;
                height: 600px;
            }
            .chat-history {
                width: 250px;
                background: #fafafa;
                border-right: 1px solid #e8e8e8;
                padding: 20px;
                overflow-y: auto;
            }
            .chat-history h3 {
                margin-bottom: 15px;
                color: #666;
            }
            .chat-history-item {
                padding: 10px;
                background: white;
                border-radius: 6px;
                margin-bottom: 10px;
                border: 1px solid #e8e8e8;
                cursor: pointer;
                font-size: 14px;
            }
            .chat-history-item:hover {
                border-color: #1890ff;
            }
            .chat-main {
                flex: 1;
                display: flex;
                flex-direction: column;
            }
            .messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #fcfcfc;
            }
            .message {
                margin-bottom: 20px;
                max-width: 80%;
            }
            .message-user {
                margin-left: auto;
            }
            .message-bot {
                margin-right: auto;
            }
            .message-content {
                padding: 12px 18px;
                border-radius: 18px;
                line-height: 1.5;
                word-wrap: break-word;
            }
            .user .message-content {
                background: #1890ff;
                color: white;
                border-bottom-right-radius: 4px;
            }
            .bot .message-content {
                background: white;
                color: #333;
                border: 1px solid #e8e8e8;
                border-bottom-left-radius: 4px;
            }
            .message-time {
                font-size: 12px;
                color: #999;
                margin-top: 4px;
                text-align: right;
            }
            .bot .message-time {
                text-align: left;
            }
            .input-area {
                padding: 20px;
                background: white;
                border-top: 1px solid #e8e8e8;
            }
            .input-container {
                display: flex;
                gap: 10px;
            }
            .input-container input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e8e8e8;
                border-radius: 8px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            .input-container input:focus {
                border-color: #1890ff;
            }
            .input-container button {
                padding: 12px 30px;
                background: #1890ff;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: background 0.3s;
            }
            .input-container button:hover {
                background: #096dd9;
            }
            .input-container button:disabled {
                background: #d9d9d9;
                cursor: not-allowed;
            }
            .info-panel {
                padding: 20px;
                background: #fafafa;
                border-top: 1px solid #e8e8e8;
            }
            .info-panel h3 {
                margin-bottom: 15px;
                color: #666;
            }
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }
            .info-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e8e8e8;
            }
            .info-label {
                font-size: 12px;
                color: #999;
                margin-bottom: 5px;
            }
            .info-value {
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }
            .typing-indicator {
                display: none;
                padding: 12px 18px;
                background: white;
                border: 1px solid #e8e8e8;
                border-radius: 18px;
                margin-bottom: 20px;
                max-width: 80px;
                margin-right: auto;
            }
            .typing-dots {
                display: flex;
                gap: 4px;
            }
            .typing-dots span {
                width: 8px;
                height: 8px;
                background: #999;
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }
            .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
            .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-5px); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Qwen2-0.5B 聊天机器人</h1>
                <div class="model-info">小型模型测试版 - 适合简单对话和测试</div>
            </div>

            <div class="status-bar">
                <div class="status">
                    <span class="status-dot"></span>
                    <span id="status-text">连接中...</span>
                </div>
                <div class="controls">
                    <button class="btn btn-secondary" onclick="checkStatus()">检查状态</button>
                    <button class="btn btn-secondary" onclick="clearChat()">清空聊天</button>
                    <button class="btn btn-primary" onclick="runTest()">运行测试</button>
                </div>
            </div>

            <div class="chat-container">
                <div class="chat-history">
                    <h3>示例问题</h3>
                    <div class="chat-history-item" onclick="sendExample('你好，请介绍一下你自己')">介绍自己</div>
                    <div class="chat-history-item" onclick="sendExample('什么是人工智能？')">什么是AI</div>
                    <div class="chat-history-item" onclick="sendExample('Python有哪些特点？')">Python特点</div>
                    <div class="chat-history-item" onclick="sendExample('写一个简单的问候函数')">代码示例</div>
                    <div class="chat-history-item" onclick="sendExample('今天的天气怎么样？')">天气查询</div>
                </div>

                <div class="chat-main">
                    <div class="messages" id="messages">
                        <div class="message bot">
                            <div class="message-content">
                                你好！我是基于 Qwen2-0.5B 模型的 AI 助手。由于我是小型模型，回答可能比较简短，但我会尽力帮助你！
                            </div>
                            <div class="message-time">系统</div>
                        </div>
                    </div>

                    <div class="typing-indicator" id="typingIndicator">
                        <div class="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>

                    <div class="input-area">
                        <div class="input-container">
                            <input type="text" id="messageInput" placeholder="输入你的问题..." 
                                   onkeypress="if(event.key==='Enter') sendMessage()">
                            <button onclick="sendMessage()" id="sendButton">发送</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="info-panel">
                <h3>系统信息</h3>
                <div class="info-grid" id="infoGrid">
                    <!-- 动态填充 -->
                </div>
            </div>
        </div>

        <script>
            const API_BASE = 'http://localhost:8000';
            let isTyping = false;

            // 页面加载时初始化
            window.onload = function() {
                checkStatus();
                loadSystemInfo();

                // 聚焦输入框
                document.getElementById('messageInput').focus();
            };

            // 检查API状态
            async function checkStatus() {
                const statusText = document.getElementById('status-text');

                try {
                    const response = await fetch(API_BASE + '/health');
                    const data = await response.json();

                    if (data.model_loaded) {
                        statusText.innerHTML = '<span style="color:#52c41a">✅ 连接正常 - 模型已加载</span>';
                    } else {
                        statusText.innerHTML = '<span style="color:#faad14">⚠️ 连接正常 - 但模型未加载</span>';
                    }

                    // 更新系统信息
                    loadSystemInfo();

                } catch (error) {
                    statusText.innerHTML = '<span style="color:#ff4d4f">❌ 无法连接到API服务器</span>';
                    console.error('连接错误:', error);
                }
            }

            // 加载系统信息
            async function loadSystemInfo() {
                const infoGrid = document.getElementById('infoGrid');

                try {
                    const response = await fetch(API_BASE + '/info');
                    const data = await response.json();

                    infoGrid.innerHTML = `
                        <div class="info-item">
                            <div class="info-label">模型状态</div>
                            <div class="info-value">${data.model_loaded ? '已加载' : '未加载'}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">计算设备</div>
                            <div class="info-value">${data.device}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">CUDA</div>
                            <div class="info-value">${data.cuda_available ? '可用' : '不可用'}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">PyTorch</div>
                            <div class="info-value">${data.torch_version}</div>
                        </div>
                    `;

                } catch (error) {
                    infoGrid.innerHTML = `
                        <div class="info-item">
                            <div class="info-label">错误</div>
                            <div class="info-value">无法获取信息</div>
                        </div>
                    `;
                }
            }

            // 发送消息
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();

                if (!message || isTyping) return;

                // 显示用户消息
                addMessage('user', message);
                input.value = '';

                // 显示输入中状态
                showTyping(true);

                try {
                    // 发送请求
                    const response = await fetch(API_BASE + '/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            messages: [
                                {
                                    role: 'system',
                                    content: '你是一个有帮助的AI助手。请用中文简洁明了地回答用户的问题。如果不知道答案，请诚实地说明。'
                                },
                                {
                                    role: 'user',
                                    content: message
                                }
                            ],
                            max_tokens: 200,
                            temperature: 0.7
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP错误: ${response.status}`);
                    }

                    const data = await response.json();

                    // 隐藏输入中状态
                    showTyping(false);

                    // 显示回复
                    addMessage('bot', data.reply);

                } catch (error) {
                    showTyping(false);
                    console.error('发送消息失败:', error);
                    addMessage('bot', '抱歉，出错了: ' + error.message);
                }
            }

            // 发送示例问题
            function sendExample(question) {
                document.getElementById('messageInput').value = question;
                sendMessage();
            }

            // 运行测试
            async function runTest() {
                const testQuestions = [
                    "你好",
                    "介绍一下你自己",
                    "什么是机器学习",
                    "写一个Python的hello world程序"
                ];

                for (const question of testQuestions) {
                    addMessage('user', question);
                    showTyping(true);

                    try {
                        const response = await fetch(API_BASE + '/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                messages: [{role: 'user', content: question}],
                                max_tokens: 150
                            })
                        });

                        const data = await response.json();
                        showTyping(false);
                        addMessage('bot', data.reply);

                        // 等待1秒
                        await new Promise(resolve => setTimeout(resolve, 1000));

                    } catch (error) {
                        showTyping(false);
                        addMessage('bot', '测试失败: ' + error.message);
                        break;
                    }
                }
            }

            // 添加消息到聊天区域
            function addMessage(type, content) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');

                const timeStr = new Date().toLocaleTimeString('zh-CN', { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                });

                messageDiv.className = `message ${type}`;
                messageDiv.innerHTML = `
                    <div class="message-content">${content}</div>
                    <div class="message-time">${timeStr}</div>
                `;

                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            // 显示/隐藏输入中状态
            function showTyping(show) {
                const typingIndicator = document.getElementById('typingIndicator');
                const sendButton = document.getElementById('sendButton');

                isTyping = show;

                if (show) {
                    typingIndicator.style.display = 'block';
                    sendButton.disabled = true;
                    sendButton.textContent = '发送中...';
                } else {
                    typingIndicator.style.display = 'none';
                    sendButton.disabled = false;
                    sendButton.textContent = '发送';
                }

                // 滚动到底部
                const messagesDiv = document.getElementById('messages');
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            // 清空聊天
            function clearChat() {
                const messagesDiv = document.getElementById('messages');
                messagesDiv.innerHTML = `
                    <div class="message bot">
                        <div class="message-content">
                            聊天已清空！开始新的对话吧。
                        </div>
                        <div class="message-time">系统</div>
                    </div>
                `;
            }

            // 监听输入框回车键
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            // 自动调整输入框高度
            document.getElementById('messageInput').addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ========== 运行服务器 ==========
if __name__ == '__main__':
    print("=" * 70)
    print("🤖 Qwen2-0.5B 聊天机器人 API 服务器")
    print("=" * 70)
    print(f"📁 模型路径: {MODEL_PATH}")
    print(f"📦 模型存在: {'✅' if os.path.exists(MODEL_PATH) else '❌'}")
    print(f"⚡ CUDA 可用: {'✅' if torch.cuda.is_available() else '❌'}")
    print(f"🔧 PyTorch 版本: {torch.__version__}")
    print("=" * 70)
    print("🌐 重要访问地址:")
    print("1. 🏠 API首页: http://localhost:8000/")
    print("2. 💬 聊天测试页: http://localhost:8000/test-page")
    print("3. 📖 API文档: http://localhost:8000/docs")
    print("=" * 70)
    print("💡 使用说明:")
    print("• 这是 0.5B 小型模型，回答会比较简短")
    print("• 建议问一些简单直接的问题")
    print("• 如果回答不理想，可以尝试重新提问")
    print("=" * 70)
    print("🛠️ 快速测试命令:")
    print('curl -X POST http://localhost:8000/api/chat \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"messages":[{"role":"user","content":"你好"}], "max_tokens":100}\'')
    print("=" * 70)
    print("按 Ctrl+C 停止服务器")
    print("=" * 70)

    # 运行服务器
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level="info",
        reload=False
    )
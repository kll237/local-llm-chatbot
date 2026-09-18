# api_server.py - 针对 Qwen2-0.5B 优化的完整版本
# 新增能力：
#   1) Redis 对话上下文缓存（可选，连不上时自动回退进程内存）
#   2) SSE 流式生成（/api/chat/stream）
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import uvicorn
import os
import time
import re
import json
import uuid
import threading
import asyncio

# redis 为可选依赖：未安装或连接失败时自动回退到进程内存，保证服务可独立运行
try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

# ========== 配置 ==========
# 模型路径：相对项目根目录自动解析，避免硬编码绝对路径，便于克隆后直接运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'Qwen2-0.5B-Instruct')

# 上下文存储（Redis / 内存）相关配置，均可通过环境变量覆盖
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "auto").lower()  # true / false / auto
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_TTL = int(os.environ.get("REDIS_TTL", "3600"))          # 会话上下文过期时间（秒）
CONTEXT_MAX_TURNS = int(os.environ.get("CONTEXT_MAX_TURNS", "20"))  # 单会话保留的最大消息条数

DEFAULT_SYSTEM = "你是一个有帮助的AI助手。请用中文简洁明了地回答用户的问题。如果不知道答案，请诚实地说明。"

# 检查模型路径
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ 警告: 模型路径不存在: {MODEL_PATH}")
    print("请确保模型文件已下载到该目录")

# 全局变量
tokenizer = None
model = None
model_loaded = False


# ========== 上下文存储（Redis / 内存） ==========
class ContextStore:
    """多轮对话上下文存储。

    - REDIS_ENABLED=true/false 强制开启/关闭 Redis；
    - REDIS_ENABLED=auto（默认）时，探测 Redis 是否可达，可达则用 Redis，否则回退内存。
    Redis 中以 `chat:ctx:{session_id}` 为 key 保存 JSON 化的消息列表，并带 TTL。
    """

    def __init__(self):
        self.backend = "memory"
        self._mem = {}  # session_id -> list[{role, content}]
        self.redis = None

        if REDIS_ENABLED == "false":
            print("🧠 上下文存储后端: memory（已通过 REDIS_ENABLED=false 禁用 Redis）")
            return

        if redis_lib is None:
            print("🧠 上下文存储后端: memory（未安装 redis 包，pip install redis 后可启用）")
            return

        try:
            self.redis = redis_lib.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                decode_responses=True, socket_connect_timeout=2, socket_timeout=2,
            )
            self.redis.ping()
            self.backend = "redis"
            print(f"🧠 上下文存储后端: redis（{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}，TTL={REDIS_TTL}s）")
        except Exception as e:
            self.redis = None
            print(f"⚠️ Redis 不可用（{e}），回退到进程内存存储（重启服务后上下文清空）")

    def _key(self, sid):
        return f"chat:ctx:{sid}"

    def get(self, sid):
        if self.backend == "redis" and self.redis:
            raw = self.redis.get(self._key(sid))
            return json.loads(raw) if raw else []
        return self._mem.get(sid, [])

    def set(self, sid, history):
        # 超出上限时裁剪：保留 system 消息 + 最近的 (MAX_TURNS-1) 条
        if len(history) > CONTEXT_MAX_TURNS:
            sys_msgs = [m for m in history if m.get("role") == "system"]
            rest = [m for m in history if m.get("role") != "system"]
            keep = rest[-(CONTEXT_MAX_TURNS - 1):] if sys_msgs else rest[-CONTEXT_MAX_TURNS:]
            history = sys_msgs + keep
        if self.backend == "redis" and self.redis:
            self.redis.set(self._key(sid), json.dumps(history, ensure_ascii=False), ex=REDIS_TTL)
        else:
            self._mem[sid] = history
        return history

    def clear(self, sid):
        if self.backend == "redis" and self.redis:
            self.redis.delete(self._key(sid))
        else:
            self._mem.pop(sid, None)

    def ttl(self, sid):
        if self.backend == "redis" and self.redis:
            return self.redis.ttl(self._key(sid))
        return None


store = ContextStore()


# ========== 数据模型 ==========
class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    # 会话 ID：传入后服务端在 Redis/内存中维护多轮上下文；不传则为无状态单轮
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None


# 创建 FastAPI 应用
app = FastAPI(
    title="Qwen2-0.5B 聊天机器人 API",
    description="本地部署的聊天机器人服务",
    version="2.0.0",
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
            dtype=torch.float32,
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


def default_reply(topic):
    """当模型输出过短或为空时的兜底回复"""
    return (
        f"关于'{topic}'，这是一个很好的问题。作为一个小型AI模型，我的回答可能不够详尽，建议你：\n"
        "1. 查阅相关文档\n2. 搜索更多资料\n3. 尝试询问更具体的问题"
    )


def build_context_messages(request: ChatRequest) -> List[Message]:
    """根据是否带 session_id 解析本次生成所用的消息列表。

    带 session_id：从 Redis/内存读取历史，拼接本次请求中的用户消息（建议每轮只发新消息），
                  首次请求可用 system 消息设定人设，后续请求中的 system 会被忽略。
    不带 session_id：无状态，沿用原逻辑仅保留最近 2 条用户消息。
    """
    if request.session_id:
        hist = store.get(request.session_id)
        msgs = [Message(role=m["role"], content=m["content"]) for m in hist]
        req_system = next((m for m in request.messages if m.role == "system"), None)
        if not any(m.role == "system" for m in msgs):
            sys_content = req_system.content if req_system else DEFAULT_SYSTEM
            msgs.insert(0, Message(role="system", content=sys_content))
        # 仅追加请求中非 system 的消息（约定每轮只传新消息）
        for m in request.messages:
            if m.role != "system":
                msgs.append(m)
        return msgs

    # 无状态路径：原逻辑
    messages = []
    if not any(m.role == "system" for m in request.messages):
        messages.append(Message(role="system", content=DEFAULT_SYSTEM))
    user_messages = [m for m in request.messages if m.role == "user"]
    recent_user_messages = user_messages[-2:] if len(user_messages) > 2 else user_messages
    messages.extend(recent_user_messages)
    return messages


def persist_context(request: ChatRequest, prior_msgs: List[Message], reply: str):
    """将本轮（历史 + 新用户消息 + 模型回复）写回 Redis/内存"""
    if not request.session_id:
        return
    history = [{"role": m.role, "content": m.content} for m in prior_msgs]
    history.append({"role": "assistant", "content": reply})
    store.set(request.session_id, history)


def generate_reply(messages, max_tokens, temperature, top_p):
    """非流式生成：整句返回并做后处理"""
    prompt = build_qwen2_prompt(messages)
    print(f"📝 Prompt预览: {prompt[:200]}...")

    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=1024, truncation=True, padding=True
    )
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    max_new_tokens = min(max_tokens, 300)
    temperature = max(min(temperature, 1.0), 0.1)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    reply = clean_response(generated_text, prompt)

    last_user = next((m.content for m in messages if m.role == "user"), "这个问题")
    if not reply or len(reply.strip()) < 3:
        reply = default_reply(last_user)
    reply = reply[:500]
    return reply


def stream_reply(messages, max_tokens, temperature, top_p):
    """流式生成：在后台线程执行 model.generate，通过 TextIteratorStreamer 逐 token 产出。
    返回生成器，产出经过后处理（去除特殊标记）的文本片段。"""
    prompt = build_qwen2_prompt(messages)
    print(f"📝 [stream] Prompt预览: {prompt[:200]}...")

    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=1024, truncation=True, padding=True
    )
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    max_new_tokens = min(max_tokens, 300)
    temperature = max(min(temperature, 1.0), 0.1)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token_text in streamer:
        yield token_text

    thread.join()


def event_generator(request: ChatRequest, messages: List[Message]):
    """SSE 事件生成器：逐 token 推送 {token, done}，结束时推送 {done:true, reply}。"""
    accumulated = []
    for piece in stream_reply(messages, request.max_tokens, request.temperature, request.top_p):
        accumulated.append(piece)
        data = json.dumps({"token": piece, "done": False}, ensure_ascii=False)
        yield f"data: {data}\n\n"

    reply = "".join(accumulated)
    last_user = next((m.content for m in messages if m.role == "user"), "这个问题")
    if not reply or len(reply.strip()) < 3:
        reply = default_reply(last_user)
    reply = reply[:500]

    persist_context(request, messages, reply)

    data = json.dumps(
        {"token": "", "done": True, "reply": reply, "session_id": request.session_id},
        ensure_ascii=False,
    )
    yield f"data: {data}\n\n"


# ========== 路由定义 ==========

@app.get("/")
async def root():
    """API首页"""
    return {
        "service": "Qwen2-0.5B 聊天机器人 API",
        "status": "running",
        "model_loaded": model_loaded,
        "context_backend": store.backend,
        "model_info": "Qwen2-0.5B-Instruct (小型模型，适合简单对话)",
        "endpoints": {
            "GET /": "此页面",
            "GET /health": "健康检查",
            "GET /info": "服务器信息",
            "POST /api/chat": "聊天接口（整句返回，支持 session_id 多轮上下文）",
            "POST /api/chat/stream": "流式聊天接口（SSE，逐 token 返回）",
            "GET /api/session/{id}": "查看会话上下文",
            "DELETE /api/session/{id}": "清理会话上下文",
            "GET /test-page": "完整测试页面（支持流式+多轮）",
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
        "context_backend": store.backend,
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
        "device": str(model.device) if model else "none",
        "context_backend": store.backend,
        "redis_enabled": REDIS_ENABLED,
    }


# ========== 核心聊天接口（整句返回） ==========

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口 - 支持 session_id 多轮上下文"""

    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请检查模型文件")

    try:
        start_time = time.time()
        messages = build_context_messages(request)
        reply = generate_reply(messages, request.max_tokens, request.temperature, request.top_p)
        persist_context(request, messages, reply)

        elapsed = time.time() - start_time
        print(f"✅ 生成完成 - 耗时: {elapsed:.2f}s, 回复长度: {len(reply)}")
        print(f"🤖 回复: {reply[:100]}...")

        return ChatResponse(reply=reply, session_id=request.session_id)

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=500, detail="GPU内存不足，尝试减少max_tokens")
    except Exception as e:
        print(f"❌ 生成错误: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"生成失败: {str(e)[:100]}"
        return ChatResponse(reply=f"抱歉，处理请求时出现错误。{error_msg}")


# ========== 流式聊天接口（SSE） ==========

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口 - SSE 逐 token 推送。

    请求体同 /api/chat（建议携带 session_id 以启用多轮上下文）。
    响应格式为 text/event-stream，每个事件形如：
        data: {"token": "你", "done": false}
        data: {"token": "好", "done": false}
        ...
        data: {"token": "", "done": true, "reply": "你好...", "session_id": "xxx"}
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请检查模型文件")

    try:
        messages = build_context_messages(request)
        return StreamingResponse(
            event_generator(request, messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 关闭代理缓冲，保证逐 token 到达
            },
        )
    except Exception as e:
        print(f"❌ 流式生成错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"流式生成失败: {str(e)[:120]}")


# ========== 会话上下文管理 ==========

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """查看某会话的上下文消息列表"""
    import re as _re
    sid = _re.sub(r'[^A-Za-z0-9_\-]', '', session_id) or "default"
    history = store.get(sid)
    return {
        "session_id": sid,
        "backend": store.backend,
        "ttl": store.ttl(sid),
        "turns": len(history),
        "messages": history,
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """清理某会话的上下文"""
    import re as _re
    sid = _re.sub(r'[^A-Za-z0-9_\-]', '', session_id) or "default"
    store.clear(sid)
    return {"session_id": sid, "cleared": True}


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
    """完整的前端测试页面（支持流式输出与多轮会话）"""
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
                white-space: pre-wrap;
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
                <div class="model-info">支持流式输出 + 多轮会话（Redis/内存上下文）</div>
            </div>

            <div class="status-bar">
                <div class="status">
                    <span class="status-dot"></span>
                    <span id="status-text">连接中...</span>
                </div>
                <div class="controls">
                    <button class="btn btn-secondary" onclick="checkStatus()">检查状态</button>
                    <button class="btn btn-secondary" onclick="newSession()">新建会话</button>
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
                                你好！我是基于 Qwen2-0.5B 模型的 AI 助手。支持流式逐字输出，并会记住本会话的上下文。
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

            // 会话 ID：本地保存，用于多轮上下文（服务端 Redis/内存中维护）
            function getSessionId() {
                let s = localStorage.getItem('chat_session');
                if (!s) {
                    s = 'sess-' + Math.random().toString(36).slice(2, 10);
                    localStorage.setItem('chat_session', s);
                }
                return s;
            }
            function newSession() {
                localStorage.removeItem('chat_session');
                fetch(API_BASE + '/api/session/' + encodeURIComponent(getSessionId()), {method:'DELETE'}).catch(()=>{});
                clearChat();
            }

            // 页面加载时初始化
            window.onload = function() {
                checkStatus();
                loadSystemInfo();
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
                            <div class="info-label">上下文后端</div>
                            <div class="info-value">${data.context_backend}</div>
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

            function scrollBottom() {
                const m = document.getElementById('messages');
                m.scrollTop = m.scrollHeight;
            }

            // 发送（流式）
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message || isTyping) return;
                addMessage('user', message);
                input.value = '';
                showTyping(true);
                try {
                    await streamChat(message);
                } catch (error) {
                    showTyping(false);
                    addMessage('bot', '抱歉，出错了: ' + error.message);
                }
            }

            // 流式调用 /api/chat/stream，逐 token 渲染
            async function streamChat(userText) {
                const sessionId = getSessionId();
                const res = await fetch(API_BASE + '/api/chat/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        messages: [{role: 'user', content: userText}],
                        max_tokens: 200,
                        temperature: 0.7
                    })
                });
                if (!res.ok) {
                    showTyping(false);
                    addMessage('bot', '请求失败: HTTP ' + res.status);
                    return;
                }
                const reader = res.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buf = '';
                let full = '';
                const contentEl = addStreamingMessage();
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    buf += decoder.decode(value, {stream: true});
                    let idx;
                    while ((idx = buf.indexOf('\\n\\n')) !== -1) {
                        const chunk = buf.slice(0, idx);
                        buf = buf.slice(idx + 2);
                        const dataLine = chunk.split('\\n').find(l => l.startsWith('data:'));
                        if (!dataLine) continue;
                        let payload;
                        try { payload = JSON.parse(dataLine.slice(5).trim()); } catch(e) { continue; }
                        if (payload.token) {
                            full += payload.token;
                            contentEl.textContent = full;
                            scrollBottom();
                        }
                    }
                }
                showTyping(false);
            }

            // 发送示例问题
            function sendExample(question) {
                document.getElementById('messageInput').value = question;
                sendMessage();
            }

            // 运行测试（非流式，无状态）
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
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    } catch (error) {
                        showTyping(false);
                        addMessage('bot', '测试失败: ' + error.message);
                        break;
                    }
                }
            }

            // 添加普通消息
            function addMessage(type, content) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                const timeStr = new Date().toLocaleTimeString('zh-CN', {hour:'2-digit', minute:'2-digit'});
                messageDiv.className = `message ${type}`;
                messageDiv.innerHTML = `
                    <div class="message-content">${content}</div>
                    <div class="message-time">${timeStr}</div>
                `;
                messagesDiv.appendChild(messageDiv);
                scrollBottom();
            }

            // 添加流式消息占位（返回内容元素，便于逐 token 更新）
            function addStreamingMessage() {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                const timeStr = new Date().toLocaleTimeString('zh-CN', {hour:'2-digit', minute:'2-digit'});
                messageDiv.className = 'message bot';
                messageDiv.innerHTML = `
                    <div class="message-content"></div>
                    <div class="message-time">${timeStr}</div>
                `;
                messagesDiv.appendChild(messageDiv);
                scrollBottom();
                return messageDiv.querySelector('.message-content');
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
                scrollBottom();
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

            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
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
    print(f"🧠 上下文后端: {store.backend}")
    print("=" * 70)
    print("🌐 重要访问地址:")
    print("1. 🏠 API首页: http://localhost:8000/")
    print("2. 💬 聊天测试页: http://localhost:8000/test-page")
    print("3. 📖 API文档: http://localhost:8000/docs")
    print("=" * 70)
    print("💡 使用说明:")
    print("• 这是 0.5B 小型模型，回答会比较简短")
    print("• 建议问一些简单直接的问题")
    print("• 流式接口: POST /api/chat/stream；多轮上下文: 传入 session_id")
    print("=" * 70)
    print("🛠️ 快速测试命令（流式）:")
    print('curl -N -X POST http://localhost:8000/api/chat/stream \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"session_id":"demo","messages":[{"role":"user","content":"你好"}],"max_tokens":120}\'')
    print("=" * 70)
    print("按 Ctrl+C 停止服务器")
    print("=" * 70)

    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level="info",
        reload=False
    )

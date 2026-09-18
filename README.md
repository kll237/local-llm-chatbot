# 本地智能对话机器人 · 基于 Qwen2-0.5B-Instruct

一个**完全本地运行**的轻量化大模型对话系统：以 Qwen2-0.5B-Instruct 为底座，覆盖「数据准备 → 模型微调 → FastAPI 推理服务 → 效果评估与可视化 → 一键打包部署」的端到端流程。模型权重与推理均在本地完成，**无需联网调用云端大模型 API**，适合在普通笔记本 / 台式机上离线运行。

> 说明：本项目为学习与演示性质。Qwen2-0.5B 是极小参数模型，回答会比较简短、偶有不准确，更适合作为「本地大模型应用工程化」的完整范式示例，而非生产级对话产品。

---

## 一、核心功能与界面

### 1) 本地大模型推理（FastAPI 服务）
- 基于 `Transformers + PyTorch` 本地加载 Qwen2-0.5B-Instruct，离线推理。
- 提供 `/api/chat` 对话接口、内置 HTML 测试页 `/test-page`、健康检查 `/health` 与接口信息 `/info`。
- 兼容 Qwen2 的 ChatML 对话模板（`<|im_start|>...<|im_end|>`），支持多轮上下文。

### 2) 命令行对话（CLI）
- `chatbot.py` 提供终端交互式对话，内部维护 `conversation_history` 实现多轮记忆。

### 3) 模型微调
- `scripts/model_training.py` 基于 `Trainer` 对 Qwen2-0.5B 做有监督微调（SFT），训练阶段强制离线模式（`TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1`）。

### 4) 一键打包部署
- `package_project.py` 将项目打包为 `AI_ChatBot_Portable.zip`，内置 Windows 启动脚本 `start.bat` 与 Linux 启动脚本 `start.sh`。

### 5) 评估与可视化
- `scripts/model_testing.py` 输出词重叠基线评估；`visualization_new.py` 生成训练 / 准确率 / 混淆矩阵 / 响应时间等展示图表。

---

## 二、技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户 / 前端                           │
│   chat_interface.html  ·  /test-page 内置页面  ·  CLI       │
└───────────────────────────┬─────────────────────────────────┘
                            │  HTTP (FastAPI + CORS)
┌───────────────────────────▼─────────────────────────────────┐
│                   scripts/api_server.py (FastAPI)            │
│   接收 messages 列表 → 构建 ChatML prompt → model.generate   │
│   端点: /api/chat  /health  /info  /docs  /test-page        │
└───────────────────────────┬─────────────────────────────────┘
                            │  Transformers + PyTorch
┌───────────────────────────▼─────────────────────────────────┐
│               Qwen2-0.5B-Instruct (本地权重)                 │
│   models/Qwen2-0.5B-Instruct  ·  models/finetuned_model     │
└─────────────────────────────────────────────────────────────┘

离线训练链路（main.py 编排）:
  data_processing.py → model_training.py → model_testing.py → visualization_new.py
```

| 层级 | 技术 |
|------|------|
| 模型 | Qwen2-0.5B-Instruct（阿里通义千问，0.5B 参数） |
| 推理框架 | Transformers 4.44.2 + PyTorch 2.9.1 |
| 服务 | FastAPI + Uvicorn + Pydantic |
| 训练 | HuggingFace `Trainer` / `DataCollatorForLanguageModeling` |
| 可视化 | Matplotlib + Seaborn |
| 数据 | Pandas + scikit-learn（清洗 / 划分） |
| 部署 | Python 标准库 `zipfile` + 批处理 / Shell 启动脚本 |

---

## 三、目录结构

```
chatbot_project/
├── chatbot.py                 # 命令行对话（ChatBot 类，本地 CLI）
├── main.py                    # 实验流程编排入口（依赖校验 + 数据→训练→测试→可视化）
├── chat_interface.html        # 独立 Web 对话界面
├── show_structure.py          # 打印项目结构
├── package_project.py         # 一键打包为跨平台便携包
├── visualization_new.py       # 生成结果展示图表（示例数据）
├── requirements.txt           # 依赖清单（已补全，见第十节）
├── EXPERIMENT_REPORT.md       # 实验说明
├── TRAINING_SUCCESS.txt       # 训练完成标记
├── data/
│   ├── train.csv              # 训练集（input, response 两列）
│   └── test.csv               # 测试集
├── scripts/
│   ├── api_server.py          # FastAPI 推理服务（核心）
│   ├── model_training.py      # Qwen2-0.5B 微调脚本
│   ├── model_testing.py       # 推理测试与词重叠基线评估
│   ├── data_processing.py     # 数据清洗 / 去重
│   ├── custom_data_loader.py  # 自定义数据集加载（免 chatterbot_corpus）
│   └── visualization.py       # BLEU/ROUGE 报告脚本（依赖 evaluation_results.csv）
├── models/                    # 【不入库】需自行下载 Qwen2-0.5B-Instruct
│   ├── Qwen2-0.5B-Instruct/   #   基础模型权重
│   └── finetuned_model/       #   微调后模型权重
├── results/                   # 评估与可视化产物
│   ├── evaluation/            #   test_results.csv / test_statistics.json
│   ├── visualization/         #   *.png 展示图
│   └── processed_train.csv / processed_test.csv
└── AI_ChatBot_Portable.zip    # 【不入库】package_project.py 打包产物
```

> `models/` 与 `AI_ChatBot_Portable.zip` 体积较大（模型权重约 1GB 级），**不纳入版本库**，请按第十节自行获取。

---

## 四、环境要求与依赖

- **Python**：3.10 及以上（本项目在 3.11/3.12 下验证）
- **硬件**：CPU 即可运行；有 CUDA 时自动使用 GPU
- **磁盘**：模型权重 + 虚拟环境约需 5GB 以上

依赖以 `requirements.txt` 为准（已补全实际 import 所需的全部包）：

```
torch==2.9.1
transformers==4.44.2
datasets
fastapi
uvicorn
pydantic
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.8.4
seaborn
```

安装：

```bash
pip install -r requirements.txt
```

---

## 五、快速开始

### 步骤 1：准备模型权重

将 Qwen2-0.5B-Instruct 下载到 `models/Qwen2-0.5B-Instruct/`：

- HuggingFace：`https://huggingface.co/Qwen/Qwen2-0.5B-Instruct`
- 或使用 ModelScope（国内镜像）下载后放置到相同目录

目录结构应为：

```
models/Qwen2-0.5B-Instruct/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

### 步骤 2：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 3：启动推理服务（Web 对话）

```bash
python scripts/api_server.py
```

启动后访问：

- 聊天测试页：<http://localhost:8000/test-page>
- API 首页：<http://localhost:8000/>
- 接口文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

### 步骤 4：命令行对话

```bash
python chatbot.py
```

### 步骤 5：微调模型

```bash
python main.py                 # 依次执行 数据处理→训练→测试→可视化
# 或单独运行训练：
python scripts/model_training.py
```

### 步骤 6：测试与评估

```bash
python scripts/model_testing.py
```

### 步骤 7：一键打包

```bash
python package_project.py      # 生成 AI_ChatBot_Portable.zip（含 start.bat / start.sh）
```

---

## 六、配置说明

### 推理服务（`scripts/api_server.py`）

- `MODEL_PATH`：模型目录。默认指向 `models/Qwen2-0.5B-Instruct`（相对项目根目录自动解析），如需加载微调模型改为 `models/finetuned_model`。
- 监听地址：`0.0.0.0:8000`（可在文件末尾 `uvicorn.run` 处修改）。
- CORS：当前为 `allow_origins=["*"]`，便于本地联调；生产环境请收紧来源。

### 微调（`scripts/model_training.py`）

- 训练数据：`results/processed_train.csv`（由 `data_processing.py` 从 `data/train.csv` 生成）。
- 关键超参：`num_train_epochs=1`、`learning_rate=5e-6`、`per_device_train_batch_size=1`。
- 说明：脚本内当前默认仅取前 20 条样本用于最小可运行验证（`if len(train_dataset) > 20: train_dataset = train_dataset.select(range(20))`）。如需完整训练，请注释该截断逻辑。

### 对话接口（`POST /api/chat`）

请求体：

```json
{
  "messages": [
    { "role": "user", "content": "你好" }
  ],
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9
}
```

响应：

```json
{ "reply": "你好！我是你的AI助手……" }
```

> 多轮上下文：`api_server.py` 接收 `messages` 列表，但为控制长度默认仅保留**最近 2 条用户消息**；`chatbot.py` 则完整维护 `conversation_history`。

---

## 七、API 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息（模型状态、端点列表） |
| GET | `/health` | 健康检查 |
| GET | `/info` | 服务器信息（模型路径、设备、CUDA、PyTorch 版本） |
| POST | `/api/chat` | 核心对话接口 |
| POST | `/api/test` | 内置自测接口 |
| GET | `/test-page` | 内置 HTML 对话测试页 |
| GET | `/docs` | Swagger 文档 |

---

## 八、项目优势

- **完全本地 / 离线**：模型权重与推理均在本地，不调用任何云端大模型 API，数据不出本机。
- **轻量可跑**：0.5B 参数模型在普通 CPU 笔记本上即可运行，资源占用低。
- **端到端闭环**：从数据集、微调、推理服务、评估到打包部署，全流程脚本化、可复现。
- **开箱即用的界面**：内置 HTML 测试页与命令行两种交互方式，无需额外前端工程。
- **跨平台启动脚本**：提供 Windows（`start.bat`）与 Linux（`start.sh`）一键启动。

---

## 九、已交付能力对照（重要 · 与原始需求描述的一致性说明）

为避免在面试 / 评审中被追问时对不上，以下将「原始项目描述」与「代码实际实现」逐一对照。**本仓库严格以代码为准，未实现的功能不作夸大表述**。

| 原始描述 | 代码实际情况 | 结论 |
|----------|--------------|------|
| 基于 Qwen2-0.5B-Instruct，Transformers+PyTorch | `api_server.py` / `chatbot.py` / `model_training.py` 均使用 Transformers + PyTorch 加载 Qwen2-0.5B-Instruct | ✅ 已实现 |
| 多轮对话管理 | `api_server.py` 接收 `messages` 列表；`chatbot.py` 维护 `conversation_history` | ✅ 已实现（API 侧默认仅保留最近 2 条用户消息） |
| **流式生成** | 当前 `/api/chat` 为整句返回，未使用 SSE / `StreamingResponse` 逐 token 流式输出 | ⚠️ **未实现**（如需可后续补充 `StreamingResponse`） |
| 模型微调、推理部署全流程 | `model_training.py` 微调 + `api_server.py` 部署 | ✅ 已实现（微调默认仅 20 条样本、1 epoch，属最小验证） |
| **引入 Redis 缓存对话上下文，优化时延与并发** | 全仓库无任何 Redis 客户端或对话上下文缓存；上下文由进程内 `messages` 列表维护 | ⚠️ **未实现**（Redis 未接入） |
| 工程化启动程序、环境校验、依赖管理、一键打包 | `main.py` 做依赖校验；`package_project.py` 生成 `AI_ChatBot_Portable.zip`（含 start.bat / start.sh） | ✅ 已实现（见下方已知限制） |
| Windows / Linux 跨平台部署包 | 已生成 `start.bat` 与 `start.sh` | ✅ 已生成，但**非完全离线 / 可移植**（见已知限制） |
| 训练与性能可视化报告、全链路评估 | `visualization_new.py` 出图；`model_testing.py` 输出词重叠基线评估 | ✅ 已实现（评估为基线级，指标为例示数据，见已知限制） |

---

## 十、复现本项目所需配置与依赖库

**模型权重（必须，不入库）**：将 Qwen2-0.5B-Instruct 放置到 `models/Qwen2-0.5B-Instruct/`（来源见第五节步骤 1）。微调产物保存至 `models/finetuned_model/`。

**依赖库**：见第四节 `requirements.txt`（已补全 `fastapi` / `uvicorn` / `pydantic` / `datasets` / `seaborn` 等原清单缺失项，确保克隆后可安装运行）。

**数据**：`data/train.csv` 与 `data/test.csv` 为 `input,response` 两列的对话样本（ChatterBot 风格中文语料），已随仓库提供；运行 `data_processing.py` 生成 `results/processed_*.csv`。

---

## 十一、已知限制与未来发展方向

**已知限制（如实说明）**

1. **未接入 Redis**：对话上下文由进程内列表维护，重启即丢失，不支持跨进程 / 跨实例共享。
2. **未实现流式输出**：`/api/chat` 整句返回，长回复时首字延迟较高。
3. **可视化指标为例示数据**：`visualization_new.py` 中的损失、准确率、混淆矩阵为固定示例数组，并非真实测评结果；`model_testing.py` 的自动评估仅为词重叠基线（5 条样本，相似度接近 0），用于演示流程。
4. **打包产物非完全离线 / 可移植**：`start.bat` / `start.sh` 仍依赖首次联网 `pip install`；`api_server.py` 早期版本模型路径为硬编码绝对路径（已改为相对路径自动解析）；打包脚本原引用了不存在的 `visualization_server.py`（已修正为 `visualization_new.py`）。
5. **微调规模小**：默认仅 20 条样本、1 epoch，属于可运行验证，非充分训练。

**未来发展方向**

- 接入 Redis 做对话上下文缓存与会话管理，提升并发与重启恢复能力。
- 使用 `StreamingResponse` + SSE 实现逐 token 流式输出，改善交互体验。
- 引入 LoRA / QLoRA 降低微调显存占用，并扩大训练数据规模。
- 将可视化与评估报告对接真实测评指标（BLEU / ROUGE / 人工打分）。
- 打包时内置依赖与模型下载引导，做到真正离线可分发的便携包。

---

## 十二、许可证

本项目用于学习与演示，模型权重请遵循 Qwen2-0.5B-Instruct 原作者的许可协议（Apache 2.0）使用。

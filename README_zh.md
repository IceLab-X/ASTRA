# ASTRA: 基于推理智能体的晶体管自动尺寸优化框架

## 概述

ASTRA (Automatic Sizing of Transistors with Reasoning Agents) 是一个专为模拟集成电路（Analog IC）设计而构建的先进优化框架。它通过实现模型上下文协议 (Model Context Protocol, MCP)，在大型语言模型 (LLM)、领域知识库和贝叶斯优化 (BO) 之间建立了结构化的推理路径。

ASTRA 创新地引入了一个两阶段流程来解决复杂的晶体管尺寸优化问题：

1. ASTRA-FastInitial (阶段一：快速初始化)：由 MCP 引导，利用检索增强生成 (RAG) 和 gm/ID 方法学，从知识库中获取设计启发，快速识别并收敛到可行的设计区域。

2. ASTRA-FocalOpt (阶段二：聚焦优化)：结合 LLM 的推理能力和互信息分析 (MI) 的数据驱动验证，识别出对性能影响最大的“关键晶体管”，随后对这些关键参数执行聚焦的贝叶斯优化。

ASTRA 是一个通用的优化框架，旨在解决各类模拟电路（如论文中演示的两级运放、三级运放和带隙基准电路）的复杂尺寸优化问题。

本项目中包含的代码示例以两级运算放大器 (Two-Stage Op-Amp) 为例，展示了 ASTRA 的核心流程。其优化目标是在满足增益、相位裕度 (PM) 和增益带宽积 (GBW) 等关键性能约束的前提下，最小化直流功耗 (DC Current)。

## 核心特性

| 序号  | 特性 (Feature)                      | 描述 (Description)                                           |
| :---: | :---------------------------------- | :----------------------------------------------------------- |
| **1** | **ASTRA-FastInitial：快速初始设计** | 结合 RAG 和 gm/ID 方法学，利用 LLM 从领域知识中推理出合理的 gm/ID 初始值，将高维 W/L 空间映射到低维空间，快速定位可行解区域 |
| **2** | **ASTRA-FocalOpt：多阶段聚焦优化**  | 将复杂的 12 维参数优化分解为 4 维、8 维和 12 维的递进阶段，允许优化器先聚焦于关键参数，显著提高在高维设计空间中的收敛速度。 |
| **3** | **MI & LLM 协同引导**               | 动态结合 MI 算法（基于数据敏感度）和 LLM（基于设计知识）的参数重要性排序，以智能地决定在每个优化阶段应该优先调整哪些参数。 |
| **4** | **基于 MCP 的并发架构**             | 使用 `mcp` 客户端/服务器架构，支持耗时的优化任务（如仿真和 BO）在后台并行执行，并通过 MCP 协议管理 LLM、工具和知识库之间的通信。 |
| **5** | **RAG 知识库支持**                  | 集成 ChromaDB 和 Sentence Transformers，建立可检索的专业知识库 (如 Razavi 的教材)，为 LLM 提供电路设计文档和经验数据的上下文，增强其推理决策能力。 |

## **项目结构**

```
├── FocalOpt/
│   ├── focal_opt_main.py         # ASTRA-FocalOpt 的主要流程、LLM 调用与权重更新逻辑。
│   ├── optimization_core.py      # 贝叶斯优化 (BO) 的核心实现：GP训练、采集函数优化、约束判断。
│   ├── mi_analysis.py            # 互信息 (MI) 计算和动态权重评分逻辑。
│   └── utility_functions.py      # 辅助函数：参数范围设置、FoM 计算、数据分组。
├── Find_Initial_Design/
│   └── bo_logic.py               # ASTRA-FastInitial (Stage 1) 逻辑，9维参数 BO，通过 LUT 查找 W/L 值。
├── astra_client.py               # MCP 客户端：用户接口、LLM 交互、工具调用和任务状态轮询。
├── astra_mcp_server.py           # MCP 服务器：后台任务管理 (FastInitial, FocalOpt) 和 RAG 工具服务。
├── build_database.py             # RAG 知识库构建脚本：PDF/TXT/MD 文件处理、嵌入生成和 ChromaDB 填充。
└── pyproject.toml                # 项目依赖清单 (uv 和 pip 均可读取)。
```

## **安装与设置**

### 1.环境依赖

本项目要求使用 Python 3.10 或更高版本。

推荐使用 uv 作为包管理器。uv 会自动读取 pyproject.toml 文件来安装依赖。

建议在虚拟环境中安装

uv 会自动创建并激活虚拟环境（如果尚不存在）

```
uv pip install -e .
```


如果您不使用 uv，也可以使用 pip：

```
pip install -e .
```

或者，如果项目中提供了 `requirements.txt` 文件，您也可以使用以下命令安装：

```
pip install -r requirements.txt
```

主要依赖项包括：torch, gpytorch, botorch, chromadb, sentence-transformers, openai, pandas, mcp, python-dotenv。

### 2.LLM API 配置

ASTRA 依赖 LLM 进行推理和参数重要性排序。您需要配置一个支持 API 接口的 LLM 服务，您可以调整调用不同厂商的LLM API。

在项目根目录下创建 .env 文件，并填入您的 API 凭证：

**.env 文件内容示例**

```
OPENAI_API_KEY="YOUR_API_KEY_HERE"
```



**如果使用非 OpenAI 兼容的 API，请修改 BASE_URL 和 MODEL**

```
BASE_URL="[http://your.api.endpoint/](http://your.api.endpoint/)"
MODEL="gpt-4o" # 或您使用的其他 LLM 模型
```



### 3.RAG 知识库构建

在运行 RAG 查询工具之前，您需要构建知识库。

将您的电路设计文档、规格书、论文等文件（.pdf, .txt, .md）放入相关目录（请确保路径与 build_database.py 中配置一致）。

建议放入.md文件。

运行数据库构建脚本：

```
python build_database.py
```

### 4.LUT 文件

ASTRA-FastInitial (Stage 1) 优化 (bo_logic.py) 依赖查找表 (LUT) 文件来计算晶体管的宽度（W）。请确保 `/gmid_LUT/` 目录下存在 `nmos_gmidX.csv` 和 `pmos_gmidY.csv` 文件。

# 使用指南

本项目通过 `astra_client.py` 脚本与后台的 `astra_mcp_server.py` 服务进行交互。

## 1.启动 MCP 服务器和客户端

打开一个终端，直接运行下面脚本，同时启动服务器和客户端：

```
uv run python astra_client.py astra_mcp_server.py
```

## 2.运行优化任务

在客户端中，您可以通过 LLM 调用的形式启动后台优化任务。整个流程分为两步（以下示例针对 T2 Op-Amp）：

**步骤 A:** ASTRA-FastInitial (查找初始可行设计 - Stage 1)

使用 RAG 工具查询初始的 gm/id 值，然后启动 find_initial_design 任务。

客户端查询示例:

```
Analyse the circuit according to the netlist and constraint，Outputs the results briefly,which can be in json format. Help me fix the best gm/ID value for each transistor（you think which detail value is the best)（1—25）,If any of them have the same w and transistor type, then the gmid is also the same. Total 5 gmid values.Constraint：phase margin(PM) over 60 degrees, gain-bandwidth product(GBW) above 4MHz, and gain exceeding 60dB.

(后续接网表内容)
```

**预期结果：** LLM 将调用 rag_query 工具获取 gm/id 值；

然后调用 find_initial_design 工具，返回一个 `task_id`。

在对话框中输入与“find initial design”语义相关的句子，＋每一个gm/id的值，进行find_initial_design步骤。

最终会得到 初始可行点（指定路径）以及一个`task_id.`



**步骤 B:** ASTRA-FocalOpt (运行聚焦优化 - Stage 2)

一旦 Stage 1 任务完成（可以通过 check_task_status 确认），然后，使用 Stage 1 返回的 task_id 启动 FocalOpt 任务。

客户端查询示例:

```
"Start the FocalOpt optimization using the results from task ID 

$$YOUR\_INITIAL\_TASK\_ID$$

 with 450 total iterations."
```

**预期结果：** LLM 将调用 FocalOpt 工具，返回一个 task_id，并开始执行多阶段聚焦优化。

最终结果会输出到指定路径。

**重要提示：** 电路数据和网表必须根据实际优化的电路进行替换。

许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

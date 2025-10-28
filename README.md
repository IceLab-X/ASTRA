# ASTRA: A Framework for Automatic Sizing of Transistors with Reasoning Agents

## Overview

ASTRA (Automatic Sizing of Transistors with Reasoning Agents) is an advanced optimization framework built specifically for analog integrated circuit (IC) design. It establishes a structured reasoning path among Large Language Models (LLMs), a domain knowledge base, and Bayesian Optimization (BO) by implementing the Model Context Protocol (MCP).

ASTRA innovatively introduces a two-stage process to solve complex transistor sizing problems:

1.  **ASTRA-FastInitial (Stage 1: Rapid Initialization):** Guided by the MCP, this stage leverages Retrieval-Augmented Generation (RAG) and the gm/ID methodology to extract design heuristics from the knowledge base, rapidly identifying and converging on a feasible design region.

2.  **ASTRA-FocalOpt (Stage 2: Focused Optimization):** This stage combines the reasoning capabilities of an LLM with data-driven validation from Mutual Information (MI) analysis to identify the "critical transistors" that have the most significant impact on performance. It then performs focused Bayesian Optimization on these key parameters.

ASTRA is a general-purpose optimization framework designed to address complex sizing problems for various analog circuits, such as the two-stage op-amp, three-stage op-amp, and bandgap reference circuit demonstrated in the paper.

The code examples included in this project use a Two-Stage Op-Amp to demonstrate ASTRA's core workflow. The optimization objective is to minimize DC power consumption (DC Current) while satisfying key performance constraints such as gain, phase margin (PM), and gain-bandwidth product (GBW).

## Core Features

|  No.  | Feature                                              | Description                                                  |
| :---: | :--------------------------------------------------- | :----------------------------------------------------------- |
| **1** | **ASTRA-FastInitial: Rapid Initial Design**          | Combines RAG and the gm/ID methodology, using an LLM to reason from domain knowledge to derive reasonable initial gm/ID values, mapping the high-dimensional W/L space to a lower-dimensional one to quickly locate a feasible solution region. |
| **2** | **ASTRA-FocalOpt: Multi-Stage Focused Optimization** | Decomposes a complex 12-dimensional parameter optimization into progressive 4-D, 8-D, and 12-D stages, allowing the optimizer to focus on critical parameters first, significantly improving convergence speed in high-dimensional design spaces. |
| **3** | **MI & LLM Co-Guidance**                             | Dynamically combines parameter importance rankings from the MI algorithm (based on data sensitivity) and the LLM (based on design knowledge) to intelligently decide which parameters to prioritize for adjustment in each optimization stage. |
| **4** | **MCP-Based Concurrent Architecture**                | Uses an `mcp` client/server architecture to support time-consuming optimization tasks (like simulation and BO) running in parallel in the background, managing communication between the LLM, tools, and knowledge base via the MCP protocol. |
| **5** | **RAG Knowledge Base Support**                       | Integrates ChromaDB and Sentence Transformers to build a searchable professional knowledge base (e.g., from Razavi's textbook), providing the LLM with context from circuit design documents and empirical data to enhance its reasoning and decision-making capabilities. |

## **Project Structure**

```
.
├── FocalOpt/
│   ├── focal_opt_main.py         # Main flow for ASTRA-FocalOpt, including LLM calls and weight updates.
│   ├── optimization_core.py      # Core Bayesian Optimization (BO) implementation: GP training, acquisition, and constraint handling.
│   ├── mi_analysis.py            # Mutual Information (MI) calculation and dynamic weighting logic.
│   └── utility_functions.py      # Helper functions: parameter range setting, FoM calculation, and data grouping.
├── Find_Initial_Design/
│   └── bo_logic.py               # ASTRA-FastInitial (Stage 1) logic, 9D BO using LUTs for W/L lookups.
├── astra_client.py               # MCP Client: User interface, LLM interaction, tool calling, and task status polling.
├── astra_mcp_server.py           # MCP Server: Backend task management (FastInitial, FocalOpt) and RAG tool service provider.
├── build_database.py             # RAG Knowledge Base builder: PDF/TXT/MD processing, embedding generation, and ChromaDB population.
└── pyproject.toml                # Project dependencies manifest (used by uv and pip).
```

## Installation and Setup

### 1. Environment Dependencies

This project requires **Python 3.10 or higher**.

**Recommended package manager is `uv`**. `uv` automatically reads the `pyproject.toml` file to install dependencies.

Installation is recommended within a virtual environment.

**Option 1 (Recommended): Using `uv`** `uv` will automatically create and activate a virtual environment if one is not present.

```
uv pip install -e .
```

**Option 2: Using `pip` with `pyproject.toml`** If you prefer not to use `uv`, you can use `pip`:

```
pip install -e .
```

**Option 3: Using `pip` with `requirements.txt`** Alternatively, if a `requirements.txt` file is provided, you can use `pip` to install dependencies from it:

```
pip install -r requirements.txt
```

Key dependencies include: `torch`, `gpytorch`, `botorch`, `chromadb`, `sentence-transformers`, `openai`, `pandas`, `mcp`, `python-dotenv`.

### 2. LLM API Configuration

ASTRA relies on an LLM for reasoning and parameter importance ranking. You need to configure an LLM service that supports the OpenAI API standard. You may adjust the call to use APIs from different vendors.

Create a **`.env`** file in the project root directory and populate it with your API credentials:

**Example `.env` file content**

```
OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

**If using a non-OpenAI compatible API, modify BASE_URL and MODEL**

```
BASE_URL="[http://your.api.endpoint/](http://your.api.endpoint/)"
MODEL="gpt-4o" # Or another LLM model you are using
```

### 3. RAG Knowledge Base Construction

You must build the knowledge base before running the RAG query tool.

1. Place your circuit design documents, specifications, papers, etc. (`.pdf`, `.txt`, `.md` files) into the relevant directory (ensure the path is consistent with the configuration in `build_database.py`).
2. Markdown files are recommended.
3. Run the database building script:

```
python build_database.py
```

### 4. LUT Files

The ASTRA-FastInitial (Stage 1) optimization (`bo_logic.py`) relies on Lookup Table (LUT) files to calculate transistor widths (W). Ensure the `/gmid_LUT/` directory contains the required `nmos_gmidX.csv` and `pmos_gmidY.csv` files.

## Usage Guide

This project interacts through the `astra_client.py` script with the backend `astra_mcp_server.py` service.

### 1. Start the MCP Server and Client

Open one terminal and run the following command to start both the server and the client simultaneously:

```
uv run python astra_client.py astra_mcp_server.py
```

### 2. Running the Optimization Tasks

From the client, you can launch background optimization tasks via LLM calls. The entire process consists of two main steps (example given for the Two-Stage Op-Amp):

#### Step A: ASTRA-FastInitial (Find Initial Feasible Design - Stage 1)

Use the RAG tool to query the initial gm/id values, then start the `find_initial_design` task.

**Client Query Example:**

```
Analyse the circuit according to the netlist and constraint，Outputs the results briefly,which can be in json format. Help me fix the best gm/ID value for each transistor（you think which detail value is the best)（1—25）,If any of them have the same w and transistor type, then the gmid is also the same. Total 5 gmid values.Constraint：phase margin(PM) over 60 degrees, gain-bandwidth product(GBW) above 4MHz, and gain exceeding 60dB.

(Netlist content to follow)
```

- **Expected Result:** The LLM will call the `rag_query` tool to obtain the gm/id values. Then, it will call the `find_initial_design` tool, returning a `task_id`.
- *Note:* Enter a sentence semantically related to "find initial design," plus the value for each gm/id, to proceed with the `find_initial_design` step.
- The final result will be an initial feasible point (saved to the specified path) and a `task_id`.

#### Step B: ASTRA-FocalOpt (Run Focused Optimization - Stage 2-4)

Once the Stage 1 task is complete (confirmed via `check_task_status`), use the `task_id` returned from Stage 1 to initiate the FocalOpt task.

**Client Query Example:**

```
Start the FocalOpt optimization using the results from task ID

$$YOUR\_INITIAL\_TASK\_ID$$

with 450 total iterations.
```

- **Expected Result:** The LLM will call the `FocalOpt` tool, return a `task_id`, and begin the multi-stage focused optimization.
- The final result will be outputted to the specified path.

## License

This project is licensed under the **MIT License**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

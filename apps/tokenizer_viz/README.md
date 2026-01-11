# Tokenizer Visualizer

A simple web application for visualizing how different tokenizers break down text into tokens.

## Features

- Support for 16 tokenizers across major LLM families:

  **OpenAI (tiktoken - fast, Rust-based):**
  - **GPT-2** (gpt2)
  - **GPT-4** (cl100k_base)
  - **Codex** (p50k_base)
  - **GPT-3** (r50k_base)
  - **GPT-4o** (o200k_base)

  **Meta & Mistral:**
  - **OPT** (facebook/opt-125m)
  - **LLaMA 3** (meta-llama/Meta-Llama-3-8B)
  - **Mistral 7B** (mistralai/Mistral-7B-v0.1)

  **Google:**
  - **Gemma 2** (google/gemma-2-2b-it)
  - **Gemma 3** (google/gemma-3-1b-it)

  **Chinese LLMs:**
  - **Qwen 3** (Qwen/Qwen3-8B)
  - **DeepSeek V3** (deepseek-ai/DeepSeek-V3)

  **Other Leading Models:**
  - **Phi-3** (microsoft/Phi-3-mini-4k-instruct)
  - **Command R** (CohereForAI/c4ai-command-r-v01)
  - **Jamba** (ai21labs/Jamba-v0.1)
  - **BLOOM** (bigscience/bloom-560m)

- **Comparison Mode**: Select "All (compare)" to tokenize with all 16 tokenizers simultaneously and compare their performance
- Interactive visualization with color-coded tokens
- Real-time tokenization as you type or paste text
- Detailed statistics: token count, character count, latency, chars/token ratio, compression ratio
- Library information (tiktoken vs HuggingFace)

## Usage

### Running with Bazel

```bash
bazel run //apps/tokenizer_viz:tokenizer_viz
```

The server will start on `http://127.0.0.1:8081` by default.

### Command-line Options

```bash
bazel run //apps/tokenizer_viz:tokenizer_viz -- --host 0.0.0.0 --port 8080 --debug
```

- `--host`: Bind host (default: 127.0.0.1)
- `--port`: Bind port (default: 8081)
- `--debug`: Enable verbose request logging

## How It Works

1. Enter or paste text in the text area
2. Select a tokenizer from the radio buttons
3. Click "Tokenize" or press Ctrl/Cmd+Enter
4. View the color-coded tokenization output

Each token is highlighted with alternating colors to make it easy to see token boundaries. Hover over tokens to see their token ID and index.

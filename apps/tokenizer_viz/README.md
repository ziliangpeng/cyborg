# Tokenizer Visualizer

A simple web application for visualizing how different tokenizers break down text into tokens.

## Features

- Support for multiple tokenizers:
  - **GPT-2** (tiktoken)
  - **GPT-4 / cl100k_base** (tiktoken)
  - **p50k_base** (tiktoken)
  - **r50k_base** (tiktoken)
  - **OPT-125M** (HuggingFace)
  - **OPT-350M** (HuggingFace)
  - **OPT-1.3B** (HuggingFace)

- Interactive visualization with color-coded tokens
- Real-time tokenization as you type or paste text
- Token count display

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

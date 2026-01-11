/* global fetch */

const textInput = document.getElementById("textInput");
const tokenizerGroup = document.getElementById("tokenizerGroup");
const tokenizeBtn = document.getElementById("tokenizeBtn");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const output = document.getElementById("output");

let currentTokens = [];
let originalText = "";

function setStatus(text) {
  statusEl.textContent = text;
}

function clearStatus() {
  statusEl.textContent = "";
}

function setStats(stats) {
  if (!stats) {
    statsEl.textContent = "";
    statsEl.innerHTML = "";
    return;
  }

  const parts = [
    `${stats.tokenCount} token${stats.tokenCount !== 1 ? "s" : ""}`,
    `${stats.charCount} char${stats.charCount !== 1 ? "s" : ""}`,
    `${stats.latencyMs}ms`,
    `${stats.avgCharsPerToken} chars/token`,
    `${stats.compressionRatio}x compression`,
    `[${stats.library}]`
  ];

  statsEl.textContent = parts.join(" • ");
}

function renderAllStats(results) {
  // Build comparison table
  let html = '<table class="stats-table">';
  html += '<thead><tr>';
  html += '<th>Tokenizer</th>';
  html += '<th>Library</th>';
  html += '<th>Tokens</th>';
  html += '<th>Latency</th>';
  html += '<th>Chars/Token</th>';
  html += '<th>Compression</th>';
  html += '</tr></thead>';
  html += '<tbody>';

  // Order tokenizers for display
  const orderedTokenizers = [
    { key: 'gpt2', name: 'GPT-2' },
    { key: 'cl100k_base', name: 'GPT-4 (cl100k)' },
    { key: 'p50k_base', name: 'Codex (p50k)' },
    { key: 'r50k_base', name: 'GPT-3 (r50k)' },
    { key: 'o200k_base', name: 'GPT-4o (o200k)' },
    { key: 'opt', name: 'OPT' },
    { key: 'llama3', name: 'LLaMA 3' },
    { key: 'mistral', name: 'Mistral' },
    { key: 'gemma2', name: 'Gemma 2' },
    { key: 'gemma3', name: 'Gemma 3' },
    { key: 'qwen3', name: 'Qwen3' },
    { key: 'deepseek', name: 'DeepSeek V3' },
    { key: 'phi3', name: 'Phi-3' },
    { key: 'command', name: 'Command R' },
    { key: 'jamba', name: 'Jamba' },
    { key: 'bloom', name: 'BLOOM' }
  ];

  orderedTokenizers.forEach(({ key, name }) => {
    const stats = results[key];
    if (!stats) return;

    if (stats.error) {
      html += `<tr><td>${name}</td><td colspan="5" class="error">Error: ${escapeHtml(stats.error)}</td></tr>`;
    } else {
      html += '<tr>';
      html += `<td>${name}</td>`;
      html += `<td>${stats.library}</td>`;
      html += `<td>${stats.tokenCount}</td>`;
      html += `<td>${stats.latencyMs}ms</td>`;
      html += `<td>${stats.avgCharsPerToken}</td>`;
      html += `<td>${stats.compressionRatio}x</td>`;
      html += '</tr>';
    }
  });

  html += '</tbody></table>';
  statsEl.innerHTML = html;
}

function renderTokens(tokens, text) {
  if (!tokens || tokens.length === 0) {
    output.textContent = text || "";
    return;
  }

  // Sort tokens by start position
  const sortedTokens = [...tokens].sort((a, b) => a.start - b.start);

  // Build HTML with colored tokens
  let html = "";
  let lastEnd = 0;
  const numColors = 6;

  sortedTokens.forEach((token, index) => {
    // Add any text before this token
    if (token.start > lastEnd) {
      const beforeText = text.substring(lastEnd, token.start);
      html += escapeHtml(beforeText);
    }

    // Add the token with color
    const colorIndex = token.index % numColors;
    const tokenText = escapeHtml(token.text);
    html += `<span class="token token-color-${colorIndex}" data-token-id="${token.id}" data-token-index="${token.index}" title="Token ID: ${token.id}, Index: ${token.index}">${tokenText}</span>`;

    lastEnd = Math.max(lastEnd, token.end);
  });

  // Add any remaining text
  if (lastEnd < text.length) {
    const afterText = text.substring(lastEnd);
    html += escapeHtml(afterText);
  }

  output.innerHTML = html;
  currentTokens = tokens;
  originalText = text;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function tokenize() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus("Please enter some text");
    output.textContent = "";
    setStats(null);
    return;
  }

  const selectedTokenizer = document.querySelector('input[name="tokenizer"]:checked')?.value;
  if (!selectedTokenizer) {
    setStatus("Please select a tokenizer");
    return;
  }

  tokenizeBtn.disabled = true;
  setStatus("Tokenizing...");
  setStats(null);
  output.textContent = "";

  try {
    if (selectedTokenizer === "all") {
      // Tokenize with all tokenizers
      const response = await fetch("/api/tokenize-all", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: text }),
      });

      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      renderAllStats(data.results);
      output.textContent = "";
      clearStatus();
    } else {
      // Tokenize with single tokenizer
      const response = await fetch("/api/tokenize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
          tokenizer: selectedTokenizer,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      renderTokens(data.tokens, text);
      setStats(data.stats);
      clearStatus();
    }
  } catch (error) {
    setStatus(`Error: ${error.message}`);
    output.textContent = "";
    setStats(null);
    console.error("Tokenization error:", error);
  } finally {
    tokenizeBtn.disabled = false;
  }
}

// Event listeners
tokenizeBtn.addEventListener("click", tokenize);

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    tokenize();
  }
});

// Auto-tokenize on tokenizer change if text exists
tokenizerGroup.addEventListener("change", () => {
  if (textInput.value.trim()) {
    tokenize();
  }
});

// Initial focus
textInput.focus();

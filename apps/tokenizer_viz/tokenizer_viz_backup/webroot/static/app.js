/* global fetch */

const textInput = document.getElementById("textInput");
const tokenizerGroup = document.getElementById("tokenizerGroup");
const tokenizeBtn = document.getElementById("tokenizeBtn");
const tokenCount = document.getElementById("tokenCount");
const statusEl = document.getElementById("status");
const output = document.getElementById("output");

let currentTokens = [];
let originalText = "";

function setStatus(text) {
  statusEl.textContent = text;
}

function clearStatus() {
  statusEl.textContent = "";
}

function setTokenCount(count) {
  if (count !== null && count !== undefined) {
    tokenCount.textContent = `${count} token${count !== 1 ? "s" : ""}`;
  } else {
    tokenCount.textContent = "";
  }
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
    setTokenCount(null);
    return;
  }

  const selectedTokenizer = document.querySelector('input[name="tokenizer"]:checked')?.value;
  if (!selectedTokenizer) {
    setStatus("Please select a tokenizer");
    return;
  }

  tokenizeBtn.disabled = true;
  setStatus("Tokenizing...");
  setTokenCount(null);
  output.textContent = "";

  try {
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
    setTokenCount(data.tokenCount);
    clearStatus();
  } catch (error) {
    setStatus(`Error: ${error.message}`);
    output.textContent = "";
    setTokenCount(null);
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

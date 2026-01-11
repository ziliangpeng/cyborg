/* global fetch */

const textInput = document.getElementById("textInput");
const tokenizerGroup = document.getElementById("tokenizerGroup");
const tokenizeBtn = document.getElementById("tokenizeBtn");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const output = document.getElementById("output");

let currentTokens = [];
let originalText = "";
let availableTokenizers = []; // Will be populated from API

// Sample text demonstrating multilingual tokenization
const SAMPLE_TEXT = `Vibe coding 🚀 is revolutionizing software development! Développeurs can now create applications by simply describing their ideas to AI assistants. 开发者无需深入理解每一行代码，而是通过自然语言交流来构建软件。これは従来のプログラミング方法とは大きく異なります 💻. The AI generates code, and developers iterate based on "vibes" ✨ rather than traditional debugging. C'est une nouvelle façon de programmer!`;

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

  // Use the tokenizers from API (in the order they were fetched)
  availableTokenizers.forEach(({ key, name }) => {
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

// Initialize: fetch available tokenizers and populate UI
async function initializeTokenizers() {
  try {
    const response = await fetch("/api/tokenizers");
    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    availableTokenizers = data.tokenizers;

    // Populate radio buttons
    tokenizerGroup.innerHTML = "";

    // Add individual tokenizer options
    availableTokenizers.forEach(({ key, name }, index) => {
      const label = document.createElement("label");
      label.className = "radio-label";

      const input = document.createElement("input");
      input.type = "radio";
      input.name = "tokenizer";
      input.value = key;
      if (index === 0) {
        input.checked = true; // First one is default
      }

      const span = document.createElement("span");
      span.textContent = name;

      label.appendChild(input);
      label.appendChild(span);
      tokenizerGroup.appendChild(label);
    });

    // Add "All (compare)" option
    const allLabel = document.createElement("label");
    allLabel.className = "radio-label";

    const allInput = document.createElement("input");
    allInput.type = "radio";
    allInput.name = "tokenizer";
    allInput.value = "all";

    const allSpan = document.createElement("span");
    allSpan.textContent = "All (compare)";

    allLabel.appendChild(allInput);
    allLabel.appendChild(allSpan);
    tokenizerGroup.appendChild(allLabel);

    // Set sample text and auto-tokenize
    textInput.value = SAMPLE_TEXT;
    tokenize();

  } catch (error) {
    setStatus(`Error loading tokenizers: ${error.message}`);
    console.error("Failed to load tokenizers:", error);
  }
}

// Initialize on page load
initializeTokenizers().then(() => {
  textInput.focus();
});

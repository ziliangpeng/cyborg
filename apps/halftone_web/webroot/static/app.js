/* global fetch */

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const submitBtn = document.getElementById("submitBtn");
const fileMeta = document.getElementById("fileMeta");
const statusEl = document.getElementById("status");
const resultsGrid = document.getElementById("resultsGrid");

let selectedFile = null;

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function clearResults() {
  resultsGrid.innerHTML = "";
}

function setSelectedFile(file) {
  selectedFile = file;
  submitBtn.disabled = !selectedFile;
  clearResults();

  if (!file) {
    fileMeta.textContent = "";
    setStatus("");
    return;
  }

  fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;
  setStatus("Ready.");
}

function isImageFile(file) {
  return !!file && (file.type?.startsWith("image/") || /\.(png|jpe?g|gif|webp|bmp)$/i.test(file.name));
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  if (!isImageFile(file)) {
    setStatus("Please select an image file.");
    return;
  }
  setSelectedFile(file);
});

function handleDroppedFiles(files) {
  if (!files || files.length === 0) return;
  const file = files[0];
  if (!isImageFile(file)) {
    setStatus("Please drop an image file.");
    return;
  }
  setSelectedFile(file);
}

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("active");
  });
});

dropZone.addEventListener("drop", (e) => {
  handleDroppedFiles(e.dataTransfer.files);
});

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    if (selectedFile && !submitBtn.disabled) {
      submitBtn.click();
    } else {
      fileInput.click();
    }
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  if (!selectedFile || submitBtn.disabled) return;

  const target = e.target;
  const tag = target && target.tagName ? target.tagName.toLowerCase() : "";
  if (tag === "textarea" || tag === "input") return;

  e.preventDefault();
  submitBtn.click();
});

document.addEventListener("paste", (e) => {
  const clipboard = e.clipboardData;
  if (!clipboard) return;

  const items = clipboard.items || [];
  for (const item of items) {
    if (!item.type || !item.type.startsWith("image/")) continue;
    const blob = item.getAsFile();
    if (!blob) continue;

    const ext = blob.type === "image/jpeg" ? "jpg" : blob.type.split("/")[1] || "png";
    const file = new File([blob], `pasted.${ext}`, { type: blob.type });
    setSelectedFile(file);
    setStatus("Pasted image.");
    break;
  }
});

submitBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  submitBtn.disabled = true;
  setStatus("Uploading and processing…");
  clearResults();

  try {
    const form = new FormData();
    form.append("image", selectedFile, selectedFile.name);

    const res = await fetch("/api/halftone", { method: "POST", body: form });
    const payload = await res.json();
    if (!res.ok || !payload.ok) {
      throw new Error(payload.error || `Request failed (${res.status})`);
    }

    setStatus(`Done. Rendered ${payload.results.length} styles.`);
    for (const item of payload.results) {
      const card = document.createElement("div");
      card.className = "card";

      const label = document.createElement("div");
      label.className = "card-title";
      label.textContent = item.style;

      const img = document.createElement("img");
      img.className = "preview";
      img.alt = item.style;
      img.loading = "lazy";
      img.src = item.dataUrl;

      card.appendChild(label);
      card.appendChild(img);
      resultsGrid.appendChild(card);
    }
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    submitBtn.disabled = !selectedFile;
  }
});

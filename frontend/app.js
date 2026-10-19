(() => {
  const API_BASE = ""; // same origin as the FastAPI backend

  // --- DOM elements -------------------------------------------------------
  const status = document.getElementById("status");
  const statusText = document.getElementById("status-text");
  const statusIcon = document.getElementById("status-icon");

  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("preview");
  const modeSelect = document.getElementById("modeSelect");
  const uploadBtn = document.getElementById("uploadBtn");

  const resultCard = document.getElementById("resultCard");
  const resultText = document.getElementById("resultText");
  const copyBtn = document.getElementById("copyBtn");
  const editBtn = document.getElementById("editBtn");

  const metaMode = document.getElementById("metaMode");
  const metaConfidence = document.getElementById("metaConfidence");
  const metaPrediction = document.getElementById("metaPrediction");
  const metaLowConf = document.getElementById("metaLowConf");
  const metaLowConfText = document.getElementById("metaLowConfText");

  const linesPanel = document.getElementById("linesPanel");
  const linesList = document.getElementById("linesList");
  const errorBox = document.getElementById("errorBox");

  let editing = false;

  // --- Backend health ------------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
      const data = await res.json();
      const cnn = data.cnn_ready;
      const crnn = data.crnn_ready;
      if (data.predictor_loaded && (cnn || crnn)) {
        setStatus("ok", `Backend ready — ${cnn && crnn ? "char + line" : cnn ? "char (CNN)" : "line (CRNN)"} mode available`);
        updateModeOptions(cnn, crnn);
      } else {
        setStatus("warn", "Backend running, but no OCR model is loaded.");
        updateModeOptions(false, false);
      }
    } catch (e) {
      setStatus("error", "Cannot reach the backend. Start it with `python -m uvicorn src.api.app:app`.");
      updateModeOptions(false, false);
    }
  }

  function setStatus(kind, text) {
    status.className = `status status--${kind}`;
    statusIcon.textContent = kind === "ok" ? "●" : kind === "warn" ? "⚠" : "✖";
    statusText.textContent = text;
  }

  function updateModeOptions(cnn, crnn) {
    const charOpt = modeSelect.querySelector('option[value="char"]');
    const lineOpt = modeSelect.querySelector('option[value="line"]');
    if (charOpt) charOpt.disabled = !cnn;
    if (lineOpt) lineOpt.disabled = !crnn;
    if (!cnn && !crnn) {
      modeSelect.disabled = true;
      uploadBtn.disabled = true;
    } else {
      modeSelect.disabled = false;
      uploadBtn.disabled = false;
    }
  }

  // --- File handling -------------------------------------------------------
  let currentFile = null;

  function showPreview(file) {
    const url = URL.createObjectURL(file);
    preview.innerHTML = `<img src="${url}" alt="Preview" />`;
    preview.classList.remove("hidden");
    currentFile = file;
    uploadBtn.disabled = false;
  }

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      showError("Please select an image file.");
      return;
    }
    showPreview(file);
    errorBox.classList.add("hidden");
    resultCard.classList.remove("shown");
  }

  // --- Recognition ---------------------------------------------------------
  uploadBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    uploadBtn.disabled = true;
    uploadBtn.textContent = "Recognizing…";
    errorBox.classList.add("hidden");
    resultCard.classList.remove("shown");

    const form = new FormData();
    form.append("file", currentFile);
    form.append("mode", modeSelect.value);

    try {
      const res = await fetch(`${API_BASE}/ocr`, { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok || !data.ok) {
        showError(data.error || data.message || `Request failed (HTTP ${res.status})`);
        return;
      }

      renderResult(data);
    } catch (e) {
      showError("Network error while contacting the OCR backend.");
    } finally {
      uploadBtn.disabled = false;
      uploadBtn.textContent = "Recognize";
    }
  });

  function renderResult(data) {
    resultText.value = data.text || "";
    editing = false;
    resultText.readOnly = true;
    editBtn.textContent = "Edit";
    resultText.classList.remove("editable");

    metaMode.textContent = data.mode;
    metaConfidence.textContent = (data.confidence * 100).toFixed(1) + "%";
    if (data.prediction) {
      metaPrediction.textContent = data.prediction;
      if (data.topk && data.topk.length) {
        metaPrediction.title = data.topk.map((t) => `${t[0]}: ${(t[1] * 100).toFixed(1)}%`).join("; ");
      }
    } else {
      metaPrediction.textContent = "—";
    }

    if (data.low_confidence) {
      metaLowConf.classList.remove("hidden");
      metaLowConfText.textContent = "Low confidence — result may be inaccurate.";
    } else {
      metaLowConf.classList.add("hidden");
    }

    if (data.lines && data.lines.length) {
      linesPanel.classList.remove("hidden");
      linesList.innerHTML = data.lines
        .map((ln, i) => `<li><b>Line ${i + 1}</b> — ${escapeHtml(ln.text)} (${(ln.confidence * 100).toFixed(1)}%)</li>`)
        .join("");
    } else {
      linesPanel.classList.add("hidden");
    }

    resultCard.classList.add("shown");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
    resultCard.classList.remove("shown");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
    })[c]);
  }

  // --- Copy / edit ---------------------------------------------------------
  copyBtn.addEventListener("click", async () => {
    const text = resultText.value;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      const old = copyBtn.textContent;
      copyBtn.textContent = "Copied!";
      setTimeout(() => { copyBtn.textContent = old; }, 1200);
    } catch (e) {
      resultText.select();
      document.execCommand("copy");
    }
  });

  editBtn.addEventListener("click", () => {
    editing = !editing;
    resultText.readOnly = !editing;
    editBtn.textContent = editing ? "Done" : "Edit";
    resultText.classList.toggle("editable", editing);
    if (editing) resultText.focus();
  });

  // --- Boot ----------------------------------------------------------------
  checkHealth();
})();
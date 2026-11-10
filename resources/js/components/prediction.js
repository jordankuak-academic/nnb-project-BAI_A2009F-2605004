import { postForm, formatPercent } from "./api.js";

export class PredictionPanel {
  constructor({ apiBase }) {
    this.apiBase = apiBase;
    this.selectedFile = null;
    this.previewUrl = null;
    this.elements = {};
  }

  mount() {
    this.elements.area = document.getElementById("upload-area");
    this.elements.input = document.getElementById("file-input");
    this.elements.inner = document.getElementById("upload-inner");
    this.elements.preview = document.getElementById("preview-image");
    this.elements.btnPredict = document.getElementById("btn-predict");
    this.elements.btnClear = document.getElementById("btn-clear");
    this.elements.spinner = document.getElementById("predict-spinner");
    this.elements.label = document.getElementById("predict-label");
    this.elements.result = document.getElementById("prediction-result");
    this.elements.predictedClass = document.getElementById("predicted-class");
    this.elements.confidence = document.getElementById("predicted-confidence");
    this.elements.confidenceFill = document.getElementById("confidence-fill");
    this.elements.top3List = document.getElementById("top3-list");
    this.elements.error = document.getElementById("predict-error");

    this.elements.area.addEventListener("click", () => this.elements.input.click());
    this.elements.preview.addEventListener("click", e => {
      e.stopPropagation();
      this.elements.input.click();
    });
    this.elements.input.addEventListener("change", e => this.handleFile(e.target.files[0]));

    ["dragenter", "dragover"].forEach(evt => {
      this.elements.area.addEventListener(evt, e => {
        e.preventDefault();
        this.elements.area.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach(evt => {
      this.elements.area.addEventListener(evt, e => {
        e.preventDefault();
        this.elements.area.classList.remove("is-dragover");
      });
    });
    this.elements.area.addEventListener("drop", e => {
      const file = e.dataTransfer.files?.[0];
      if (file) this.handleFile(file);
    });

    this.elements.preview.hidden = true;
    this.elements.preview.removeAttribute("src");
    this.elements.area.classList.remove("upload-area--preview");

    this.elements.btnPredict.addEventListener("click", () => this.predict());
    this.elements.btnClear.addEventListener("click", () => this.clear());
  }

  handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      this.showError("Please select an image file (JPG / PNG / WebP).");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      this.showError("Image is larger than 10 MB.");
      return;
    }
    this.hideError();
    this.selectedFile = file;
    this.elements.inner.hidden = true;
    this.elements.area.classList.add("upload-area--preview");
    this.elements.result.hidden = true;
    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
    this.previewUrl = URL.createObjectURL(file);
    this.elements.preview.onload = () => {
      this.elements.preview.hidden = false;
    };
    this.elements.preview.onerror = () => {
      this.elements.preview.hidden = true;
      this.elements.preview.removeAttribute("src");
      this.elements.inner.hidden = false;
      this.elements.area.classList.remove("upload-area--preview");
      this.showError("Preview could not be loaded.");
    };
    this.elements.preview.hidden = true;
    this.elements.preview.src = this.previewUrl;
    this.elements.btnPredict.disabled = false;
    this.elements.btnClear.hidden = false;
  }

  async predict() {
    if (!this.selectedFile) return;
    this.setLoading(true);
    this.hideError();

    try {
      const form = new FormData();
      form.append("file", this.selectedFile);
      const result = await postForm(`${this.apiBase}/api/predict`, form);
      this.renderResult(result);
    } catch (err) {
      this.showError(err.message);
    } finally {
      this.setLoading(false);
    }
  }

  setLoading(loading) {
    this.elements.btnPredict.disabled = loading || !this.selectedFile;
    this.elements.spinner.hidden = !loading;
    this.elements.label.textContent = loading ? "Analyzing…" : "Analyze Image";
  }

  renderResult(result) {
    const predictions = result.top_3 || result.top_k || [];
    const top = predictions[0] || { label: result.label, confidence: result.confidence };

    this.elements.result.hidden = false;
    this.elements.predictedClass.textContent = top.label ?? "—";
    const conf = top.confidence ?? 0;
    this.elements.confidence.textContent = formatPercent(conf);
    this.elements.confidenceFill.style.width = `${Math.max(2, Math.min(100, conf * 100))}%`;

    this.elements.top3List.innerHTML = "";
    predictions.slice(0, 3).forEach((p, idx) => {
      const li = document.createElement("li");
      if (idx === 0) li.classList.add("is-top");
      li.innerHTML = `
        <span class="top3-label">${p.label ?? "—"}</span>
        <span class="top3-score">${formatPercent(p.confidence ?? 0)}</span>
      `;
      this.elements.top3List.appendChild(li);
    });
  }

  clear() {
    this.selectedFile = null;
    this.elements.input.value = "";
    if (this.previewUrl) {
      URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = null;
    }
    this.elements.preview.removeAttribute("src");
    this.elements.preview.hidden = true;
    this.elements.inner.hidden = false;
    this.elements.area.classList.remove("upload-area--preview");
    this.elements.btnPredict.disabled = true;
    this.elements.btnClear.hidden = true;
    this.elements.result.hidden = true;
    this.hideError();
  }

  showError(message) {
    this.elements.error.textContent = message;
    this.elements.error.hidden = false;
  }

  hideError() {
    this.elements.error.hidden = true;
  }
}
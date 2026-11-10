import { EvaluationPanel } from "./components/evaluation.js";
import { PredictionPanel } from "./components/prediction.js";

const API_BASE = "";

document.addEventListener("DOMContentLoaded", () => {
  const status = {
    set(state, text) {
      const dot = document.querySelector("#model-status .status-dot");
      const label = document.getElementById("model-status-text");
      dot.className = `status-dot status-dot--${state}`;
      label.textContent = text;
    }
  };

  const evaluation = new EvaluationPanel({
    apiBase: API_BASE,
    onStatus: (state, text) => status.set(state, text)
  });

  const prediction = new PredictionPanel({
    apiBase: API_BASE
  });

  evaluation.mount();
  prediction.mount();
});
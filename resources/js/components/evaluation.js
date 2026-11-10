import { postJson, formatPercent, formatNumber } from "./api.js";

export class EvaluationPanel {
  constructor({ apiBase, onStatus }) {
    this.apiBase = apiBase;
    this.onStatus = onStatus;
    this.elements = {};
  }

  mount() {
    this.elements.button = document.getElementById("btn-evaluate");
    this.elements.label = document.getElementById("evaluate-label");
    this.elements.spinner = document.getElementById("evaluate-spinner");
    this.elements.metricsGrid = document.getElementById("metrics-grid");
    this.elements.tablesGrid = document.getElementById("tables-grid");
    this.elements.baselineCard = document.getElementById("baseline-card");
    this.elements.error = document.getElementById("evaluate-error");

    this.elements.fields = {
      accuracy: document.getElementById("metric-accuracy"),
      precision: document.getElementById("metric-precision"),
      recall: document.getElementById("metric-recall"),
      f1: document.getElementById("metric-f1"),
      weightedF1: document.getElementById("metric-weighted-f1"),
      top3: document.getElementById("metric-top3"),
      inference: document.getElementById("metric-inference"),
      samples: document.getElementById("metric-samples")
    };

    this.elements.perClassBody = document.querySelector("#per-class-table tbody");
    this.elements.confusionTable = document.getElementById("confusion-table");

    this.elements.baseline = {
      acc: document.getElementById("baseline-acc"),
      f1: document.getElementById("baseline-f1"),
      deltaAcc: document.getElementById("delta-acc"),
      deltaF1: document.getElementById("delta-f1")
    };

    this.elements.button.addEventListener("click", () => this.run());
  }

  async run() {
    this.setLoading(true);
    this.elements.error.hidden = true;
    this.onStatus?.("warning", "Evaluating testing set…");

    try {
      const result = await postJson(`${this.apiBase}/api/evaluate`);
      this.render(result);
      this.onStatus?.("success", `Evaluation complete · F1 ${formatNumber(result.macro_f1)}`);
    } catch (err) {
      this.elements.error.textContent = err.message;
      this.elements.error.hidden = false;
      this.onStatus?.("danger", "Evaluation failed");
    } finally {
      this.setLoading(false);
    }
  }

  setLoading(loading) {
    this.elements.button.disabled = loading;
    this.elements.spinner.hidden = !loading;
    this.elements.label.textContent = loading ? "Evaluating…" : "Run Test on Testing Set";
  }

  render(data) {
    this.elements.metricsGrid.hidden = false;
    this.elements.tablesGrid.hidden = false;
    this.elements.baselineCard.hidden = false;

    this.elements.fields.accuracy.textContent = formatPercent(data.accuracy);
    this.elements.fields.precision.textContent = formatNumber(data.macro_precision);
    this.elements.fields.recall.textContent = formatNumber(data.macro_recall);
    this.elements.fields.f1.textContent = formatNumber(data.macro_f1);
    this.elements.fields.weightedF1.textContent = formatNumber(data.weighted_f1);
    this.elements.fields.top3.textContent = formatPercent(data.top_3_accuracy);
    this.elements.fields.inference.textContent = formatNumber(data.average_inference_time_ms, 2);
    this.elements.fields.samples.textContent = String(data.sample_count);

    this.renderPerClass(data.per_class);
    this.renderConfusion(data.confusion_matrix, data.class_names);
    this.renderBaseline(data.baseline, data.task1_comparison);
  }

  renderPerClass(perClass) {
    const tbody = this.elements.perClassBody;
    tbody.innerHTML = "";
    Object.entries(perClass).forEach(([name, stats]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${name}</td>
        <td>${formatNumber(stats.precision)}</td>
        <td>${formatNumber(stats.recall)}</td>
        <td>${formatNumber(stats["f1-score"] ?? stats.f1)}</td>
        <td>${String(Math.round(stats.support))}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  renderConfusion(matrix, classNames) {
    const table = this.elements.confusionTable;
    const shorten = name => (name === "cardboard" ? "cardb" : name);
    table.innerHTML = "";

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    headRow.appendChild(document.createElement("th")).textContent = "True \\ Pred";
    classNames.forEach(name => {
      const th = document.createElement("th");
      th.textContent = shorten(name);
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    matrix.forEach((row, i) => {
      const tr = document.createElement("tr");
      const rowHead = document.createElement("th");
      rowHead.textContent = shorten(classNames[i]);
      tr.appendChild(rowHead);
      row.forEach((value, j) => {
        const td = document.createElement("td");
        td.textContent = String(value);
        if (i === j) td.classList.add("is-diagonal");
        else if (value === 0) td.classList.add("is-zero");
        else td.classList.add("is-wrong");
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  renderBaseline(baseline, comparison) {
    const b = this.elements.baseline;
    if (!baseline) {
      b.acc.textContent = "—";
      b.f1.textContent = "—";
      b.deltaAcc.textContent = "—";
      b.deltaF1.textContent = "—";
      return;
    }
    b.acc.textContent = formatPercent(baseline.accuracy);
    b.f1.textContent = formatNumber(baseline.macro_f1);
    if (comparison) {
      const dAcc = comparison.accuracy_delta_vs_baseline ?? 0;
      const dF1 = comparison.macro_f1_delta_vs_baseline ?? 0;
      b.deltaAcc.textContent = `${dAcc >= 0 ? "+" : ""}${(dAcc * 100).toFixed(2)} pp`;
      b.deltaF1.textContent = `${dF1 >= 0 ? "+" : ""}${formatNumber(dF1)}`;
      b.deltaAcc.className = `baseline-grid__value ${dAcc >= 0 ? "baseline-grid__value--positive" : "baseline-grid__value--negative"}`;
      b.deltaF1.className = `baseline-grid__value ${dF1 >= 0 ? "baseline-grid__value--positive" : "baseline-grid__value--negative"}`;
    }
  }
}
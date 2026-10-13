"use strict";

const byId = id => document.getElementById(id);
const series = ["actual", "lightgbm", "lag_1"];
const format = value => value.toFixed(2);
const timestampPattern = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/;

function validate(data) {
  if (data.schema_version !== 1 || !Array.isArray(data.records)) {
    throw new Error("Unsupported export schema");
  }
  for (const key of ["dataset", "model", "split", "evaluation", "dependencies"]) {
    if (!data[key] || typeof data[key] !== "object") throw new Error(`Missing ${key}`);
  }
  if (!Array.isArray(data.features)) throw new Error("Missing features");
  for (const key of ["lightgbm", "lag_1"]) {
    for (const metric of ["mae", "rmse"]) {
      if (!Number.isFinite(data.metrics?.[key]?.[metric])) throw new Error("Invalid metrics");
    }
  }
  for (const row of data.records) {
    if (!row || !timestampPattern.test(row.timestamp) ||
        !series.every(key => Number.isFinite(row[key]))) throw new Error("Invalid observation");
  }
}

function svgNode(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderWindow(records) {
  const start = Math.max(0, Math.min(records.length - 1, Math.trunc(Number(byId("start").value)) || 0));
  const size = Math.max(1, Math.min(168, Math.trunc(Number(byId("size").value)) || 48));
  byId("start").value = String(start);
  const rows = records.slice(start, start + size);
  const summary = `${rows.length} observations · ${rows[0].timestamp} to ${rows.at(-1).timestamp}`;
  byId("window-summary").textContent = summary;
  const chart = byId("chart");
  chart.setAttribute("role", "img");
  chart.setAttribute("aria-labelledby", "chart-title chart-description");
  chart.replaceChildren(
    svgNode("title", {id: "chart-title"}, "Historical traffic volume comparison"),
    svgNode("desc", {id: "chart-description"}, `${summary}. Actual: solid; LightGBM: dashed; lag-1: dotted. Equally spaced observation rows, not elapsed hours. Exact values follow in the table.`)
  );
  const values = rows.flatMap(row => series.map(key => row[key]));
  const low = Math.min(0, ...values);
  const high = Math.max(1, ...values);
  const x = index => 75 + index / Math.max(1, rows.length - 1) * 860;
  const y = value => 270 - (value - low) / (high - low) * 230;
  for (const value of [low, (high + low) / 2, high]) {
    chart.append(svgNode("line", {x1: 75, x2: 935, y1: y(value), y2: y(value), stroke: "#c3ced6"}));
    chart.append(svgNode("text", {x: 65, y: y(value) + 5, "text-anchor": "end"}, String(Math.round(value))));
  }
  chart.append(svgNode("text", {x: 75, y: 22}, "Traffic volume"));
  chart.append(svgNode("text", {x: 75, y: 305}, `Row ${start}`));
  chart.append(svgNode("text", {x: 935, y: 305, "text-anchor": "end"}, `Row ${start + rows.length - 1}`));
  series.forEach((key, index) => {
    // A short segment keeps a single remaining observation visible.
    let points = rows.map((row, i) => `${x(i)},${y(row[key])}`).join(" ");
    if (rows.length === 1) points += ` ${x(0) + 4},${y(rows[0][key])}`;
    chart.append(svgNode("polyline", {points, class: key, fill: "none", "stroke-width": 2.5,
      "stroke-dasharray": ["none", "9 4", "2 4"][index], role: "presentation"}));
  });
  byId("rows").replaceChildren(...rows.map(row => {
    const tr = document.createElement("tr");
    for (const value of [row.timestamp, ...series.map(key => format(row[key]))]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    return tr;
  }));
}

async function load() {
  try {
    const response = await fetch("data/forecast.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    validate(data);
    byId("metrics").textContent = ["lightgbm", "lag_1"].map(key =>
      `${key === "lightgbm" ? "LightGBM" : "Lag-1 baseline"}: MAE ${format(data.metrics[key].mae)} · RMSE ${format(data.metrics[key].rmse)}`
    ).join(" | ");
    byId("methodology").textContent = [
      data.evaluation.mode, data.evaluation.fit_semantics,
      `Lags: ${data.evaluation.lag_semantics}. Rolling: ${data.evaluation.rolling_semantics}.`,
      data.evaluation.prior_observed_validation_targets
        ? "Validation consumes prior observed targets for subsequent predictions."
        : "Validation does not consume prior observed targets.",
      data.evaluation.timestamp_semantics, data.evaluation.test_file, data.model.selection
    ].join(" ");
    const source = new URL(data.dataset.source);
    if (source.protocol !== "https:") throw new Error("Invalid dataset source");
    const link = document.createElement("a");
    link.href = source.href;
    link.textContent = data.dataset.name;
    byId("attribution").replaceChildren(link, ` · ${data.dataset.license} · ${data.dataset.license_evidence}`);
    byId("metadata").textContent = ["model", "split", "features", "evaluation", "dependencies", "dataset"]
      .map(key => `${key}\n${JSON.stringify(data[key], null, 2)}`).join("\n\n");
    byId("dashboard").hidden = false;
    if (!data.records.length) {
      byId("controls").disabled = true;
      byId("status").textContent = "No historical records are available in this export.";
      return;
    }
    byId("start").max = String(data.records.length - 1);
    byId("size").value = "48";
    for (const id of ["start", "size"]) byId(id).addEventListener("change", () => renderWindow(data.records));
    renderWindow(data.records);
    byId("controls").disabled = false;
    byId("status").textContent = `Loaded ${data.records.length} historical observations. Select a window below.`;
  } catch (error) {
    byId("dashboard").hidden = true;
    byId("controls").disabled = true;
    byId("status").textContent = `Unable to load historical evaluation: ${error.message}. Serve the docs directory over HTTP and reload; check data/forecast.json is available.`;
  }
}

load();

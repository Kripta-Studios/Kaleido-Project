"use strict";

var state = { operations: [], selected: null };

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, function (char) {
    return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[char];
  });
}

function formatMinutes(value) {
  var sign = value < 0 ? "−" : "+";
  var absolute = Math.abs(value);
  if (absolute >= 60) {
    return sign + Math.floor(absolute / 60) + "h " + Math.round(absolute % 60) + "m";
  }
  return sign + Math.round(absolute) + "m";
}

function formatDuration(value) {
  var hours = Math.floor(value / 60);
  var minutes = Math.round(value % 60);
  return hours + "h " + minutes + "m";
}

function localTime(value) {
  return new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit", minute: "2-digit", day: "2-digit", month: "short",
    timeZone: "UTC", timeZoneName: "short"
  }).format(new Date(value));
}

function riskClass(status) {
  if (status === "critical") return "risk-critical";
  if (status === "watch") return "risk-watch";
  return "risk-stable";
}

function operationRow(operation) {
  var selected = operation.operation_id === state.selected ? " selected" : "";
  var deltaClass = operation.plan_delta_minutes < 0 ? "negative" : "positive";
  return "<tr class=\"" + selected.trim() + "\" data-operation=\"" +
    escapeHtml(operation.operation_id) + "\">" +
    "<td><div class=\"operation-name\"><strong>" + escapeHtml(operation.operation_id) +
    " · " + escapeHtml(operation.vessel) + "</strong><span>" +
    escapeHtml(operation.operation_type) + " · " + escapeHtml(operation.berth) + "</span></div></td>" +
    "<td class=\"progress-cell\"><div class=\"progress-meta\"><span>" +
    Math.round(operation.progress * 100) + "%</span><span>" + escapeHtml(operation.shift) +
    "</span></div><div class=\"progress-track\"><i style=\"width:" +
    Math.round(operation.progress * 100) + "%\"></i></div></td>" +
    "<td><span class=\"risk-pill " + riskClass(operation.status) + "\">" +
    operation.risk + "%</span></td>" +
    "<td><span class=\"delta " + deltaClass + "\">" +
    formatMinutes(operation.plan_delta_minutes) + "</span></td>" +
    "<td class=\"row-arrow\">›</td></tr>";
}

function renderOperations(filter) {
  var query = (filter || "").toLowerCase();
  var rows = state.operations.filter(function (operation) {
    return operation.operation_id.toLowerCase().includes(query) ||
      operation.vessel.toLowerCase().includes(query) ||
      operation.operation_type.toLowerCase().includes(query);
  });
  document.getElementById("operations-body").innerHTML = rows.map(operationRow).join("") ||
    "<tr><td colspan=\"5\" class=\"loading-row\">No matching operations</td></tr>";
  document.querySelectorAll("[data-operation]").forEach(function (row) {
    row.addEventListener("click", function () { selectOperation(row.dataset.operation); });
  });
}

function detailMarkup(operation) {
  var timeline = operation.timeline.map(function (event) {
    return "<li class=\"" + escapeHtml(event.state) + "\"><span class=\"timeline-dot\"></span>" +
      "<label>" + escapeHtml(event.label) + "</label><time>" +
      localTime(event.time) + "</time></li>";
  }).join("");
  var bars = operation.risk_horizons.map(function (item) {
    return "<div class=\"bar-column\"><strong>" + item.risk + "%</strong>" +
      "<div class=\"bar\" style=\"height:" + Math.max(6, item.risk) + "%\"></div>" +
      "<label>" + item.hours + "h</label></div>";
  }).join("");
  var reasons = operation.reason_codes.map(function (reason) {
    return "<span>" + escapeHtml(reason.replaceAll("_", " ")) + "</span>";
  }).join("");
  return "<div class=\"detail-head\"><div class=\"detail-title-row\"><div class=\"detail-title\">" +
    "<span class=\"section-label\">SELECTED OPERATION</span><h2>" +
    escapeHtml(operation.operation_id) + "</h2><p>" + escapeHtml(operation.vessel) + " · " +
    escapeHtml(operation.berth) + " · plan revision " + operation.plan_revision + "</p></div>" +
    "<div class=\"detail-risk\"><strong>" + operation.risk + "%</strong><span>8h risk</span></div>" +
    "</div></div><div class=\"prediction-band\"><div class=\"prediction\"><span>Remaining P50</span>" +
    "<strong>" + formatDuration(operation.remaining_p50_minutes) + "</strong><small>point estimate</small>" +
    "</div><div class=\"prediction\"><span>Remaining P90</span><strong>" +
    formatDuration(operation.remaining_p90_minutes) + "</strong><small>uncertainty bound</small></div></div>" +
    "<div class=\"risk-chart\"><div class=\"subheading\"><h3>Deviation risk by horizon</h3>" +
    "<span>synthetic probability</span></div><div class=\"bars\">" + bars + "</div></div>" +
    "<div class=\"timeline-section\"><div class=\"subheading\"><h3>Evidence timeline</h3>" +
    "<span>cutoff-safe events</span></div><ol class=\"timeline\">" + timeline + "</ol></div>" +
    "<div class=\"reason-box\"><h3>Associated factors · not causal attribution</h3>" +
    "<div class=\"reason-tags\">" + reasons + "</div><div class=\"audit-line\">" +
    "Model " + escapeHtml(operation.model_version) + " · " + escapeHtml(operation.claim_state) +
    " · cutoff " + localTime(operation.data_cutoff) + " · source writes disabled</div></div>";
}

async function selectOperation(operationId) {
  state.selected = operationId;
  renderOperations(document.getElementById("operation-search").value);
  var panel = document.getElementById("operation-detail");
  panel.innerHTML = "<div class=\"detail-empty\"><div class=\"spinner\"></div><p>Loading cutoff-safe evidence…</p></div>";
  try {
    var response = await fetch("/v1/demo/operations/" + encodeURIComponent(operationId));
    if (!response.ok) throw new Error("Operation unavailable");
    var operation = await response.json();
    panel.innerHTML = detailMarkup(operation);
    document.getElementById("global-cutoff").textContent = localTime(operation.data_cutoff);
  } catch (error) {
    panel.innerHTML = "<div class=\"detail-empty\"><p>" + escapeHtml(error.message) + "</p></div>";
  }
}

async function boot() {
  try {
    var response = await fetch("/v1/demo/overview");
    if (!response.ok) throw new Error("API did not return the replay");
    var overview = await response.json();
    state.operations = overview.operations;
    document.getElementById("metric-active").textContent = overview.operations_active;
    document.getElementById("metric-risk").textContent = overview.operations_at_risk;
    document.getElementById("metric-delta").textContent = formatMinutes(overview.mean_plan_delta_minutes);
    state.selected = state.operations[0].operation_id;
    renderOperations("");
    await selectOperation(state.selected);
  } catch (error) {
    document.getElementById("operations-body").innerHTML =
      "<tr><td colspan=\"5\" class=\"loading-row\">" + escapeHtml(error.message) + "</td></tr>";
  }
}

document.getElementById("operation-search").addEventListener("input", function (event) {
  renderOperations(event.target.value);
});
document.getElementById("watermark-close").addEventListener("click", function () {
  document.querySelector(".watermark").style.display = "none";
});
document.querySelector(".mobile-menu").addEventListener("click", function () {
  document.querySelector(".sidebar").classList.toggle("open");
});
boot();

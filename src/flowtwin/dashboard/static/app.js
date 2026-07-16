"use strict";

var state = {
  operations: [],
  selected: null,
  currentDetail: null,
  evidence: null,
  overview: null,
  filters: {query: "", status: "all", minRisk: 0, sort: "risk_desc"}
};

function escapeHtml(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, function (char) {
    return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[char];
  });
}

function formatMinutes(value) {
  var sign = Number(value) < 0 ? "−" : "+";
  var absolute = Math.abs(Number(value));
  if (absolute >= 60) return sign + Math.floor(absolute / 60) + "h " + Math.round(absolute % 60) + "m";
  return sign + Math.round(absolute) + "m";
}

function formatDuration(value) {
  var numeric = Number(value || 0);
  return Math.floor(numeric / 60) + "h " + Math.round(numeric % 60) + "m";
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

function showToast(message) {
  var toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timeoutId);
  showToast.timeoutId = window.setTimeout(function () { toast.classList.remove("show"); }, 2400);
}

async function copyText(value) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(value);
    return;
  }
  var textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function downloadBlob(filename, content, contentType) {
  var blob = new Blob([content], {type: contentType});
  var url = URL.createObjectURL(blob);
  var link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  showToast("Export prepared: " + filename);
}

function openDrawer(title, markup) {
  document.getElementById("drawer-title").textContent = title;
  document.getElementById("drawer-body").innerHTML = markup;
  document.getElementById("drawer-backdrop").hidden = false;
  var drawer = document.getElementById("utility-drawer");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  document.getElementById("drawer-backdrop").hidden = true;
  var drawer = document.getElementById("utility-drawer");
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  document.getElementById("alerts-button").setAttribute("aria-expanded", "false");
  document.getElementById("profile-button").setAttribute("aria-expanded", "false");
}

function operationRow(operation) {
  var selected = operation.operation_id === state.selected ? " selected" : "";
  var deltaClass = operation.plan_delta_minutes < 0 ? "negative" : "positive";
  return "<tr class=\"" + selected.trim() + "\" data-operation=\"" +
    escapeHtml(operation.operation_id) + "\" tabindex=\"0\">" +
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

function filteredOperations() {
  var query = state.filters.query.toLowerCase();
  var rows = state.operations.filter(function (operation) {
    var matchesQuery = operation.operation_id.toLowerCase().includes(query) ||
      operation.vessel.toLowerCase().includes(query) ||
      operation.operation_type.toLowerCase().includes(query) ||
      operation.berth.toLowerCase().includes(query);
    var matchesStatus = state.filters.status === "all" || operation.status === state.filters.status;
    return matchesQuery && matchesStatus && operation.risk >= state.filters.minRisk;
  });
  rows.sort(function (left, right) {
    if (state.filters.sort === "progress_desc") return right.progress - left.progress;
    if (state.filters.sort === "delta_asc") return left.plan_delta_minutes - right.plan_delta_minutes;
    if (state.filters.sort === "operation_asc") return left.operation_id.localeCompare(right.operation_id);
    return right.risk - left.risk;
  });
  return rows;
}

function bindOperationRows() {
  document.querySelectorAll("[data-operation]").forEach(function (row) {
    function activate() { selectOperation(row.dataset.operation); }
    row.addEventListener("click", activate);
    row.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") activate();
    });
  });
}

function renderOperations() {
  var rows = filteredOperations();
  document.getElementById("operations-body").innerHTML = rows.map(operationRow).join("") ||
    "<tr><td colspan=\"5\" class=\"loading-row\">No operations match the current filters</td></tr>";
  bindOperationRows();
}

function objectMarkup(operation) {
  return operation.object_links.map(function (item) {
    return "<div class=\"object-chip\"><span>" + escapeHtml(item.type) + "</span><strong>" +
      escapeHtml(item.id) + "</strong></div>";
  }).join("");
}

function scenarioControls(operation) {
  var actions = operation.allowed_scenarios.map(function (action, index) {
    return "<label><input type=\"checkbox\" name=\"scenario-action\" value=\"" +
      escapeHtml(action) + "\"" + (index < 2 ? " checked" : "") + ">" +
      escapeHtml(action.replaceAll("_", " ")) + "</label>";
  }).join("");
  return "<div class=\"scenario-box\"><div class=\"scenario-head\"><h3>Allowed scenarios</h3>" +
    "<span>simulation · no realized saving</span></div><div class=\"scenario-actions\">" + actions +
    "</div><button class=\"scenario-run\" id=\"run-scenarios\" type=\"button\">Rank selected scenarios</button>" +
    "<div class=\"scenario-results\" id=\"scenario-results\"><div class=\"scenario-warning\">" +
    "Results are synthetic advisory comparisons restricted to approved demo actions.</div></div></div>";
}

function detailMarkup(operation) {
  var timeline = operation.timeline.map(function (event) {
    return "<li class=\"" + escapeHtml(event.state) + "\"><span class=\"timeline-dot\"></span>" +
      "<label>" + escapeHtml(event.label) + "</label><time>" + localTime(event.time) + "</time></li>";
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
    "<span class=\"section-label\">SELECTED OPERATION</span><h2>" + escapeHtml(operation.operation_id) +
    "</h2><p>" + escapeHtml(operation.vessel) + " · " + escapeHtml(operation.berth) +
    " · plan revision " + operation.plan_revision + "</p></div><div class=\"detail-risk\"><strong>" +
    operation.risk + "%</strong><span>8h synthetic risk</span></div></div>" +
    "<div class=\"detail-actions\"><button class=\"primary\" id=\"copy-audit\" type=\"button\">Copy audit</button>" +
    "<button id=\"export-operation\" type=\"button\">Export JSON</button>" +
    "<button id=\"open-model-card\" type=\"button\">Model card</button></div></div>" +
    "<div class=\"prediction-band\"><div class=\"prediction\"><span>Remaining P50</span><strong>" +
    formatDuration(operation.remaining_p50_minutes) + "</strong><small>point estimate</small></div>" +
    "<div class=\"prediction\"><span>Remaining P90</span><strong>" +
    formatDuration(operation.remaining_p90_minutes) + "</strong><small>uncertainty bound</small></div></div>" +
    "<div class=\"risk-chart\"><div class=\"subheading\"><h3>Deviation risk by horizon</h3>" +
    "<span>synthetic probability</span></div><div class=\"bars\">" + bars + "</div></div>" +
    "<div class=\"bottleneck-card\"><div><span>SYNTHETIC BOTTLENECK</span><strong>" +
    escapeHtml(operation.bottleneck.object) + "</strong><small>" + escapeHtml(operation.bottleneck.evidence) +
    "</small></div><strong>" + operation.bottleneck.median_wait_minutes + " min</strong></div>" +
    "<div class=\"timeline-section\"><div class=\"subheading\"><h3>Evidence timeline</h3>" +
    "<span>" + operation.source_event_count + " cutoff-safe events</span></div><ol class=\"timeline\">" +
    timeline + "</ol></div><div class=\"reason-box\"><h3>Associated factors · not causal attribution</h3>" +
    "<div class=\"reason-tags\">" + reasons + "</div><div class=\"object-grid\">" +
    objectMarkup(operation) + "</div><div class=\"audit-line\">Audit " + escapeHtml(operation.audit_id) +
    " · model " + escapeHtml(operation.model_version) + " · " + escapeHtml(operation.claim_state) +
    " · cutoff " + localTime(operation.data_cutoff) + " · source writes disabled</div></div>" +
    scenarioControls(operation);
}

async function runScenarios(operation) {
  var selected = Array.from(document.querySelectorAll("input[name=scenario-action]:checked"))
    .map(function (input) { return input.value; });
  var results = document.getElementById("scenario-results");
  if (!selected.length) {
    results.innerHTML = "<div class=\"scenario-warning\">Select at least one approved action.</div>";
    return;
  }
  results.innerHTML = "<div class=\"scenario-warning\">Evaluating synthetic scenarios…</div>";
  try {
    var response = await fetch("/v1/scenarios/rank", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({operation_id: operation.operation_id, approved_actions: selected})
    });
    if (!response.ok) throw new Error("Scenario service unavailable");
    var payload = await response.json();
    results.innerHTML = payload.scenarios.map(function (item) {
      return "<div class=\"scenario-result\"><strong>#" + item.rank + "</strong><span>" +
        escapeHtml(item.action.replaceAll("_", " ")) + "</span><strong>" +
        formatMinutes(item.estimated_p50_delta_minutes) + "</strong></div>";
    }).join("") + "<div class=\"scenario-warning\">" + escapeHtml(payload.evidence_type) +
      " · advisory only · no causal claim</div>";
  } catch (error) {
    results.innerHTML = "<div class=\"scenario-warning\">" + escapeHtml(error.message) + "</div>";
  }
}

function auditMarkup(operation) {
  return "<div class=\"drawer-card\"><span>AUDIT ID</span><strong>" + escapeHtml(operation.audit_id) +
    "</strong><p>Prediction generated from " + operation.source_event_count +
    " cutoff-safe events. Plan revision " + operation.plan_revision + " was visible from " +
    localTime(operation.plan_valid_from) + ".</p></div><ul class=\"drawer-list\">" +
    operation.source_event_ids.map(function (eventId) { return "<li><strong>" + escapeHtml(eventId) +
      "</strong><span>source event included before cutoff</span></li>"; }).join("") +
    "</ul><div class=\"drawer-card\"><span>DATA QUALITY</span><strong>" +
    escapeHtml(operation.data_quality.timestamp_audit) + "</strong><p>Plan cutoff: " +
    escapeHtml(operation.data_quality.plan_cutoff) + ". Missing required fields: " +
    operation.data_quality.missing_required_fields + ". Relationship integrity: " +
    escapeHtml(operation.data_quality.relationship_integrity) + ".</p></div>";
}

function bindDetailActions(operation) {
  document.getElementById("copy-audit").addEventListener("click", async function () {
    await copyText(JSON.stringify({
      audit_id: operation.audit_id, operation_id: operation.operation_id,
      cutoff: operation.data_cutoff, plan_revision: operation.plan_revision,
      model_version: operation.model_version, source_event_ids: operation.source_event_ids,
      claim_state: operation.claim_state, source_writes: false
    }, null, 2));
    showToast("Audit copied to clipboard");
  });
  document.getElementById("export-operation").addEventListener("click", function () {
    downloadBlob(operation.operation_id + "-smoke-only.json", JSON.stringify(operation, null, 2), "application/json");
  });
  document.getElementById("open-model-card").addEventListener("click", openModelCard);
  document.getElementById("run-scenarios").addEventListener("click", function () { runScenarios(operation); });
}

async function selectOperation(operationId) {
  state.selected = operationId;
  renderOperations();
  var panel = document.getElementById("operation-detail");
  panel.innerHTML = "<div class=\"detail-empty\"><div class=\"spinner\"></div><p>Loading cutoff-safe evidence…</p></div>";
  try {
    var response = await fetch("/v1/demo/operations/" + encodeURIComponent(operationId));
    if (!response.ok) throw new Error("Operation unavailable");
    var operation = await response.json();
    state.currentDetail = operation;
    panel.innerHTML = detailMarkup(operation);
    bindDetailActions(operation);
    document.getElementById("global-cutoff").textContent = localTime(operation.data_cutoff);
  } catch (error) {
    panel.innerHTML = "<div class=\"detail-empty\"><p>" + escapeHtml(error.message) + "</p></div>";
  }
}

function metricContextMarkup(context) {
  if (!context || !context.available) return "";
  var comparators = context.comparators.map(function (item) {
    return "<div class=\"metric-row\"><span>" + escapeHtml(item.label) + "</span><strong>" +
      Number(item.mae_hours).toFixed(2) + " h</strong><strong>−" +
      Number(item.selected_gain_percent).toFixed(1) + "%</strong></div>";
  }).join("");
  return "<div class=\"metric-context\"><div class=\"metric-context-head\"><div><span>HOW TO READ THE ETA RESULT</span>" +
    "<strong>" + Number(context.selected_mae_hours).toFixed(2) + " h average absolute error</strong></div>" +
    "<strong class=\"verdict\">" + context.gate_checks_passed + "/" + context.gate_checks_total +
    " frozen public gates</strong></div>" +
    "<div class=\"metric-comparators\">" + comparators + "</div><p class=\"metric-caveat\">" +
    "Median error " + Number(context.median_absolute_error_hours).toFixed(2) + " h · " +
    (context.within_1h * 100).toFixed(1) + "% within ±1 h · " +
    (context.within_2h * 100).toFixed(1) + "% within ±2 h · " +
    (context.within_4h * 100).toFixed(1) + "% within ±4 h. P90 coverage " +
    (context.p90_interval_coverage * 100).toFixed(1) + "% with a " +
    Number(context.p90_interval_width_hours).toFixed(2) + " h-wide interval. " +
    escapeHtml(context.explanation) + "</p><div class=\"metric-question\">" +
    escapeHtml(context.presentation_line) + " Test: " + context.test_trips + " trips / " +
    Number(context.test_prefixes).toLocaleString("en-GB") + " prefixes; " +
    (context.nola_prefix_share * 100).toFixed(1) + "% from New Orleans.</div></div>";
}

function evidenceMetric(stage) {
  if (stage.metric_value == null && stage.mae_minutes == null) return "Pending";
  if (stage.metric_unit === "hours") return Number(stage.metric_value).toFixed(2) + " h";
  return Number(stage.mae_minutes).toFixed(2) + " min";
}

function evidenceMarkup(payload) {
  var rows = payload.stages.map(function (stage, index) {
    var failed = stage.status === "did_not_beat_reference" ? " failed" : "";
    var metric = evidenceMetric(stage);
    var width = Math.max(8, Math.min(100, stage.relative_score || 0));
    return "<div class=\"evidence-stage" + failed + "\" data-evidence-index=\"" + index +
      "\" role=\"button\" tabindex=\"0\"><div class=\"stage-index\">" + escapeHtml(stage.milestone) +
      "</div><div class=\"stage-copy\"><strong>" + escapeHtml(stage.label) + "</strong><span>" +
      escapeHtml(stage.description) + "</span></div><div class=\"stage-metric\"><strong>" + metric +
      "</strong><span>future test MAE · lower is better</span></div><div class=\"evidence-bar\"><i style=\"width:" +
      width + "%\"></i></div></div>";
  }).join("");
  var research = "";
  if (payload.research_finding) {
    var finding = payload.research_finding;
    var metricDisplay = finding.metric_display || ("−" + Number(finding.injected_signal_gain_minutes).toFixed(2) + " min");
    var metricLabel = finding.metric_label || "correct vs shuffled · synthetic";
    research = "<div class=\"research-card\"><div><span>" + escapeHtml(finding.title) + "</span><strong>" +
      escapeHtml(finding.verdict) + "</strong></div><div class=\"research-metric\"><strong>" +
      escapeHtml(metricDisplay) + "</strong><span>" + escapeHtml(metricLabel) + "</span></div><p>" +
      escapeHtml(finding.summary) + "</p></div>";
  }
  return rows + research + metricContextMarkup(payload.metric_context) +
    "<div class=\"evidence-note\">" + escapeHtml(payload.note) + "</div>";
}

function openEvidenceStage(index) {
  var stage = state.evidence.stages[index];
  openDrawer(stage.label, "<div class=\"drawer-card\"><span>" + escapeHtml(stage.milestone) +
    " · " + escapeHtml(stage.status) + "</span><strong>" +
    evidenceMetric(stage) + " MAE" +
    "</strong><p>" + escapeHtml(stage.description) +
    ". Lower is better. This public test result is smoke_only and does not establish Kaleido value.</p></div>");
}

function bindEvidenceStages() {
  document.querySelectorAll("[data-evidence-index]").forEach(function (row) {
    function activate() { openEvidenceStage(Number(row.dataset.evidenceIndex)); }
    row.addEventListener("click", activate);
    row.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") activate();
    });
  });
}

async function loadEvidence() {
  var panel = document.getElementById("model-evidence");
  try {
    var response = await fetch("/v1/demo/evidence");
    if (!response.ok) throw new Error("Evidence artifacts unavailable");
    state.evidence = await response.json();
    panel.innerHTML = evidenceMarkup(state.evidence);
    bindEvidenceStages();
  } catch (error) {
    panel.innerHTML = "<div class=\"evidence-loading\"><span>" + escapeHtml(error.message) + "</span></div>";
  }
}

function renderDataset(dataset) {
  document.getElementById("dataset-name").textContent = dataset.dataset_id || "synthetic_ui_fixture";
  document.getElementById("dataset-operations").textContent = dataset.source_cases_used == null ? "Pending" :
    Number(dataset.source_cases_used).toLocaleString("en-GB");
  document.getElementById("dataset-prefixes").textContent = dataset.prefix_rows == null ? "Pending" :
    Number(dataset.prefix_rows).toLocaleString("en-GB");
  var split = dataset.split_counts_operations || {};
  document.getElementById("dataset-split").textContent = split.train == null ? "Pending" :
    Number(split.train).toLocaleString("en-GB") + " / " + Number(split.validation).toLocaleString("en-GB") +
    " / " + Number(split.test).toLocaleString("en-GB");
  document.getElementById("dataset-hash").textContent = dataset.source_file_sha256 ?
    dataset.source_file_sha256.slice(0, 12) + "…" : "Pending";
  document.getElementById("dataset-entity-label").textContent = dataset.entity_label || "operations";
  document.getElementById("dataset-domain-note").textContent = dataset.domain_note || "public evidence dataset";
}

async function loadHealth() {
  try {
    var response = await fetch("/health");
    if (!response.ok) throw new Error("health unavailable");
    var health = await response.json();
    document.getElementById("api-status").innerHTML = "<i class=\"status-dot\"></i> API healthy · source writes disabled";
    return health;
  } catch (error) {
    document.getElementById("api-status").textContent = "API unavailable";
    return null;
  }
}

async function openAlerts() {
  document.getElementById("alerts-button").setAttribute("aria-expanded", "true");
  openDrawer("Synthetic alerts", "<div class=\"evidence-loading\"><div class=\"spinner\"></div><span>Loading…</span></div>");
  try {
    var response = await fetch("/v1/demo/alerts");
    var payload = await response.json();
    document.getElementById("drawer-body").innerHTML = "<div class=\"drawer-card\"><span>DEMO THRESHOLD</span><strong>" +
      payload.count + " synthetic alerts</strong><p>These are interface fixtures, not Kaleido material-risk alerts.</p></div>" +
      "<ul class=\"drawer-list\">" + payload.alerts.map(function (alert) {
        return "<li><button class=\"alert-link\" type=\"button\" data-alert-operation=\"" +
          escapeHtml(alert.operation_id) + "\"><strong>" + escapeHtml(alert.operation_id) + " · " +
          alert.risk + "%</strong><span>" + escapeHtml(alert.message) + "</span></button></li>";
      }).join("") + "</ul>";
    document.querySelectorAll("[data-alert-operation]").forEach(function (button) {
      button.addEventListener("click", function () {
        closeDrawer(); selectOperation(button.dataset.alertOperation);
        document.getElementById("operations").scrollIntoView();
      });
    });
  } catch (error) {
    document.getElementById("drawer-body").textContent = error.message;
  }
}

async function openProfile() {
  document.getElementById("profile-button").setAttribute("aria-expanded", "true");
  var health = await loadHealth();
  openDrawer("Local demo session", "<div class=\"drawer-card\"><span>SESSION</span><strong>FlowTwin local MVP</strong>" +
    "<p>Read-only: " + escapeHtml(health ? health.read_only : "unknown") +
    " · source write capability: " + escapeHtml(health ? health.source_write_capability : "unknown") +
    " · claim state: " + escapeHtml(health ? health.claim_state : "unknown") +
    ".</p></div><div class=\"drawer-actions\"><button class=\"compact-button\" id=\"download-full-export\" type=\"button\">Full JSON export</button>" +
    "<a class=\"compact-button link-button\" href=\"/docs\" target=\"_blank\" rel=\"noopener\">API docs</a></div>");
  document.getElementById("download-full-export").addEventListener("click", async function () {
    var response = await fetch("/v1/demo/export");
    var payload = await response.json();
    downloadBlob("flowtwin-demo-export-smoke-only.json", JSON.stringify(payload, null, 2), "application/json");
  });
}

async function openModelCard() {
  openDrawer("Prediction model card", "<div class=\"evidence-loading\"><div class=\"spinner\"></div></div>");
  var response = await fetch("/v1/models/latest/card");
  var card = await response.json();
  document.getElementById("drawer-body").innerHTML = "<div class=\"drawer-card\"><span>MODEL VERSION</span><strong>" +
    escapeHtml(card.model_version) + "</strong><p>Dataset: " + escapeHtml(card.dataset_id) +
    " · split: " + escapeHtml(card.split_protocol) + " · claim state: " + escapeHtml(card.claim_state) +
    ".</p></div><ul class=\"drawer-list\">" + card.limitations.map(function (item) {
      return "<li><strong>Limitation</strong><span>" + escapeHtml(item) + "</span></li>";
    }).join("") + "</ul>";
}

function openCurrentAudit() {
  if (!state.currentDetail) { showToast("Select an operation first"); return; }
  openDrawer("Prediction audit", auditMarkup(state.currentDetail));
}

function exportOperationsCsv() {
  var columns = ["operation_id", "vessel", "operation_type", "berth", "shift", "progress", "risk", "status", "plan_delta_minutes", "last_event"];
  var lines = [columns.join(",")].concat(filteredOperations().map(function (row) {
    return columns.map(function (column) {
      return "\"" + String(row[column]).replaceAll("\"", "\"\"") + "\"";
    }).join(",");
  }));
  downloadBlob("flowtwin-filtered-operations-smoke-only.csv", lines.join("\r\n"), "text/csv;charset=utf-8");
}

function processInfo(stage) {
  var descriptions = {
    READY: "Cargo and documents are available at the current cutoff.",
    ASSIGN: "Approved resources and object relationships are visible.",
    HANDLE: "Synthetic waiting-time acceleration marks the current demo bottleneck.",
    DONE: "Completion is future state only; it is never exposed as an input feature."
  };
  openDrawer("Process stage · " + stage, "<div class=\"drawer-card\"><span>OBJECT-CENTRIC FLOW</span><strong>" +
    escapeHtml(stage) + "</strong><p>" + escapeHtml(descriptions[stage]) +
    " This process map is descriptive and preserves object identity.</p></div>");
}

async function boot() {
  var refresh = document.getElementById("refresh-button");
  refresh.disabled = true;
  refresh.textContent = "Loading…";
  try {
    var response = await fetch("/v1/demo/overview");
    if (!response.ok) throw new Error("API did not return the replay");
    var overview = await response.json();
    state.overview = overview;
    state.operations = overview.operations;
    document.getElementById("metric-active").textContent = overview.operations_active;
    document.getElementById("metric-risk").textContent = overview.operations_at_risk;
    document.getElementById("metric-delta").textContent = formatMinutes(overview.mean_plan_delta_minutes);
    document.getElementById("metric-coverage").textContent = overview.p90_coverage == null ?
      "Pending" : (overview.p90_coverage * 100).toFixed(1) + "%";
    document.getElementById("hero-events").textContent = Number(overview.fixture_events || 2185).toLocaleString("en-GB");
    renderDataset(overview.dataset || {});
    if (!state.selected || !state.operations.some(function (item) { return item.operation_id === state.selected; })) {
      state.selected = state.operations[0].operation_id;
    }
    renderOperations();
    await Promise.all([selectOperation(state.selected), loadEvidence(), loadHealth()]);
    var alerts = await fetch("/v1/demo/alerts").then(function (result) { return result.json(); });
    document.getElementById("alerts-button").title = alerts.count + " synthetic alerts";
  } catch (error) {
    document.getElementById("operations-body").innerHTML =
      "<tr><td colspan=\"5\" class=\"loading-row\">" + escapeHtml(error.message) + "</td></tr>";
  } finally {
    refresh.disabled = false;
    refresh.textContent = "Refresh";
  }
}

document.getElementById("operation-search").addEventListener("input", function (event) {
  state.filters.query = event.target.value;
  renderOperations();
});
document.getElementById("status-filter").addEventListener("change", function (event) {
  state.filters.status = event.target.value; renderOperations();
});
document.getElementById("risk-filter").addEventListener("input", function (event) {
  state.filters.minRisk = Number(event.target.value);
  document.getElementById("risk-filter-value").textContent = state.filters.minRisk + "%";
  renderOperations();
});
document.getElementById("sort-operations").addEventListener("change", function (event) {
  state.filters.sort = event.target.value; renderOperations();
});
document.getElementById("filters-toggle").addEventListener("click", function () {
  var panel = document.getElementById("filter-panel");
  panel.hidden = !panel.hidden;
  this.setAttribute("aria-expanded", String(!panel.hidden));
});
document.getElementById("reset-filters").addEventListener("click", function () {
  state.filters = {query: "", status: "all", minRisk: 0, sort: "risk_desc"};
  document.getElementById("operation-search").value = "";
  document.getElementById("status-filter").value = "all";
  document.getElementById("risk-filter").value = "0";
  document.getElementById("risk-filter-value").textContent = "0%";
  document.getElementById("sort-operations").value = "risk_desc";
  renderOperations(); showToast("Filters reset");
});
document.getElementById("export-operations").addEventListener("click", exportOperationsCsv);
document.getElementById("export-evidence").addEventListener("click", function () {
  if (!state.evidence) { showToast("Evidence is still loading"); return; }
  downloadBlob("flowtwin-model-evidence-smoke-only.json", JSON.stringify(state.evidence, null, 2), "application/json");
});
document.getElementById("refresh-button").addEventListener("click", async function () {
  await boot(); showToast("Replay refreshed from generated artifacts");
});
document.getElementById("alerts-button").addEventListener("click", openAlerts);
document.getElementById("profile-button").addEventListener("click", openProfile);
document.getElementById("audit-open").addEventListener("click", openCurrentAudit);
document.getElementById("drawer-close").addEventListener("click", closeDrawer);
document.getElementById("drawer-backdrop").addEventListener("click", closeDrawer);
document.getElementById("watermark-close").addEventListener("click", function () {
  document.querySelector(".watermark").style.display = "none";
});
document.querySelector(".mobile-menu").addEventListener("click", function () {
  document.querySelector(".sidebar").classList.toggle("open");
});
document.querySelectorAll(".nav-item").forEach(function (link) {
  link.addEventListener("click", function () {
    document.querySelectorAll(".nav-item").forEach(function (item) { item.classList.remove("active"); });
    link.classList.add("active");
    document.querySelector(".sidebar").classList.remove("open");
  });
});
document.querySelectorAll("[data-process]").forEach(function (node) {
  function activate() { processInfo(node.dataset.process); }
  node.addEventListener("click", activate);
  node.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") activate();
  });
});
document.addEventListener("keydown", function (event) { if (event.key === "Escape") closeDrawer(); });
boot();

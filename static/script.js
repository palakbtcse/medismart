/* ═══════════════════════════════════════════════
   MediSmart — Frontend Script
   Connects to Flask Backend API at /api/*
   
   API Endpoints used:
     GET  /api/symptoms       → all symptoms list
     POST /api/predict        → predict disease from symptoms
     GET  /api/models/stats   → ML accuracy stats
     GET  /api/diseases       → all diseases (admin table)
═══════════════════════════════════════════════ */

const API_BASE = "";   // same origin — Flask serves both frontend & API

let ALL_SYMPTOMS = [];
let currentStep  = 1;

function fmtSym(s) {
  return s.replace(/_/g, " ").replace(/\s+/g, " ").trim()
    .split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

async function apiFetch(url, options = {}) {
  try {
    const res = await fetch(API_BASE + url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    return await res.json();
  } catch (err) {
    console.error("API error:", err);
    return { success: false, error: "Network error. Is Flask running on port 5000?" };
  }
}

/* ─── LOAD SYMPTOMS FROM API ─── */
async function initSymptoms() {
  const data = await apiFetch("/api/symptoms");
  if (!data.success) {
    document.getElementById("symGrid").innerHTML =
      `<div style="color:var(--rose);font-size:.85rem;padding:20px;background:#fff1f2;border-radius:9px;border:1px solid #fca5a5">
        ⚠ Could not load symptoms from server.<br/>
        <span style="font-size:.78rem;color:#92400e">Make sure Flask is running: <code>python app.py</code></span>
       </div>`;
    return;
  }
  ALL_SYMPTOMS = data.symptoms;
  renderSymptomGrid();
}

function renderSymptomGrid() {
  document.getElementById("symGrid").innerHTML = ALL_SYMPTOMS.map(s =>
    `<label class="sc">
      <input type="checkbox" value="${s}"/>
      <span class="sc-box">
        <svg viewBox="0 0 12 12" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round">
          <polyline points="2,6 5,9 10,3"/>
        </svg>
      </span>
      <span class="sc-text">${fmtSym(s)}</span>
    </label>`
  ).join("");

  document.getElementById("symGrid").addEventListener("change", function (e) {
    if (e.target.type === "checkbox") {
      e.target.closest(".sc").classList.toggle("on", e.target.checked);
      updSymCount();
    }
  });
}

function updSymCount() {
  const n = document.querySelectorAll("#symGrid input:checked").length;
  document.getElementById("symCount").textContent = n + " selected";
  document.getElementById("symNote").textContent =
    n > 0 ? n + " symptom" + (n > 1 ? "s" : "") + " selected · Step 2 of 4" : "Step 2 of 4";
}

function clearSym() {
  document.querySelectorAll("#symGrid .sc").forEach(c => {
    c.classList.remove("on");
    c.querySelector("input").checked = false;
  });
  updSymCount();
}

/* ─── STEPPER ─── */
async function goStep(n) {
  if (n > 1) {
    const age    = parseInt(document.getElementById("age").value);
    const gender = document.getElementById("gender").value;
    if (!age || age < 1 || age > 120) { alert("Please enter a valid age (1–120)."); return; }
    if (!gender) { alert("Please select your gender."); return; }
  }
  if (n > 2) {
    const sel = document.querySelectorAll("#symGrid input:checked");
    if (sel.length === 0) { alert("Please select at least one symptom."); return; }
    await buildResults();
  }
  document.querySelectorAll(".step-panel").forEach(p => p.classList.remove("active"));
  document.getElementById("step" + n).classList.add("active");
  currentStep = n;
  updateStepper(n);
  document.getElementById("diagnose").scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateStepper(active) {
  for (let i = 1; i <= 4; i++) {
    const circle = document.getElementById("sc" + i);
    const label  = document.getElementById("sl" + i);
    circle.classList.remove("done", "active");
    label.classList.remove("done", "active");
    if (i < active) {
      circle.classList.add("done");
      circle.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round"><polyline points="2,6 5,9 10,3"/></svg>`;
      label.classList.add("done");
    } else if (i === active) {
      circle.classList.add("active");
      circle.textContent = i;
      label.classList.add("active");
    } else {
      circle.textContent = i;
    }
    if (i < 4) document.getElementById("line" + i).classList.toggle("done", i < active);
  }
}

/* ─── PREDICT (calls Flask API) ─── */
async function buildResults() {
  const age    = parseInt(document.getElementById("age").value);
  const gender = document.getElementById("gender").value;
  const sel    = [...document.querySelectorAll("#symGrid input:checked")].map(i => i.value);
  const gLabel = gender === "other" ? "Not specified" : gender.charAt(0).toUpperCase() + gender.slice(1);

  const list = document.getElementById("medList");
  list.innerHTML = `
    <li style="padding:32px;text-align:center">
      <div class="loading-spinner"></div>
      <div style="font-size:.85rem;color:var(--sub);margin-top:14px">Analysing symptoms with ML models…</div>
    </li>`;
  document.getElementById("resultSummary").innerHTML = "";
  document.getElementById("resultMeta").textContent  = "Processing…";

  const data = await apiFetch("/api/predict", {
    method: "POST",
    body: JSON.stringify({ symptoms: sel, age, gender, model: "random_forest" }),
  });

  if (!data.success) {
    list.innerHTML = `<li><div class="empty-state">
      <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
      <div class="empty-title">Server Error</div>
      <div class="empty-sub">${data.error}</div>
    </div></li>`;
    return;
  }

  const top  = data.predictions;
  const meta = data.meta;

  document.getElementById("resultMeta").textContent =
    `${top.length} condition${top.length !== 1 ? "s" : ""} identified · Age ${age} · ${gLabel} · ${meta.symptoms_received} symptom${meta.symptoms_received !== 1 ? "s" : ""} analysed`;

  document.getElementById("resultSummary").innerHTML =
    `<div class="rs-chip"><svg viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2" stroke-linecap="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>Age <strong>${age}</strong> · <strong>${gLabel}</strong></div>
     <div class="rs-chip"><svg viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2" stroke-linecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg><strong>${meta.symptoms_received}</strong> symptom${meta.symptoms_received !== 1 ? "s" : ""} selected</div>
     <div class="rs-chip"><svg viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2" stroke-linecap="round"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg><strong>${top.length}</strong> condition${top.length !== 1 ? "s" : ""} matched</div>
     <div class="rs-chip" style="background:var(--teal-lt);border-color:var(--teal-md)"><svg viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2" stroke-linecap="round"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>Model: <strong style="color:var(--teal-dk)">${meta.model_used.replace("_", " ")}</strong></div>`;

  if (top.length === 0) {
    list.innerHTML = `<li><div class="empty-state">
      <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg></div>
      <div class="empty-title">No matching conditions found</div>
      <div class="empty-sub">Try selecting more symptoms or consult a physician directly.</div>
    </div></li>`;
  } else {
    list.innerHTML = top.map((entry, i) => {
      const isTop = i === 0;
      return `
      <li class="med-item disease-card${isTop ? " disease-card--top" : ""}" style="animation-delay:${i * 0.09}s;display:block;padding:0">
        <div class="dc-main">
          <div class="dc-header">
            <div class="dc-title-row">
              ${isTop ? '<span class="dc-top-badge">Best Match</span>' : ""}
              <div class="dc-name">${entry.disease}</div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:5px">
              <span class="sev-badge ${entry.severity_cls}">${entry.severity_label} Severity</span>
              <span style="font-size:.7rem;color:var(--teal-dk);font-weight:700;background:var(--teal-lt);border:1px solid var(--teal-md);border-radius:var(--r-pill);padding:2px 8px">${entry.confidence_pct}% confidence</span>
            </div>
          </div>

          ${entry.description ? `<div class="dc-desc">${entry.description}</div>` : ""}

          <div class="dc-matched-syms">
            ${entry.matched_symptoms.map(s => `<span class="dc-sym-chip">${fmtSym(s)}</span>`).join("")}
          </div>

          <div class="dc-sections">
            ${entry.medications && entry.medications.length > 0 ? `
            <div class="dc-section">
              <div class="dc-section-title"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2" stroke-linecap="round"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>Medications</div>
              <div class="dc-meds">${entry.medications.map(m => `<span class="dc-med-pill">${m}</span>`).join("")}</div>
            </div>` : ""}
            ${entry.precautions && entry.precautions.length > 0 ? `
            <div class="dc-section">
              <div class="dc-section-title"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2" stroke-linecap="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>Precautions</div>
              <div class="dc-precs">${entry.precautions.map((p, pi) => `<div class="dc-prec-item"><span class="dc-prec-num">${pi + 1}</span><span>${p}</span></div>`).join("")}</div>
            </div>` : ""}
          </div>
        </div>
      </li>`;
    }).join("");
  }

  document.getElementById("doneSummary").innerHTML =
    `<strong>${top.length}</strong> condition${top.length !== 1 ? "s" : ""} identified for Age ${age} (${gLabel}) using <strong>${meta.model_used.replace("_", " ")}</strong>. Top match: <strong>${top[0]?.disease || "—"}</strong>. Please consult a licensed physician before taking any medication.`;
}

/* ─── NAV ─── */
function startWizard() { document.getElementById("diagnose").scrollIntoView({ behavior: "smooth" }); }
function restartWizard() {
  document.getElementById("age").value    = "";
  document.getElementById("gender").value = "";
  clearSym();
  goStep(1);
  document.getElementById("home").scrollIntoView({ behavior: "smooth" });
}

/* ─── ADMIN TABLE ─── */
async function loadAdminTable() {
  const tbody = document.getElementById("dtBody");
  if (!tbody) return;
  tbody.innerHTML = `<tr><td colspan="3" style="text-align:center;color:var(--sub);padding:20px;font-size:.85rem">Loading from API…</td></tr>`;
  const data = await apiFetch("/api/diseases");
  if (!data.success) {
    tbody.innerHTML = `<tr><td colspan="3" style="color:var(--rose);padding:16px;font-size:.82rem">⚠ ${data.error}</td></tr>`;
    return;
  }
  tbody.innerHTML = data.diseases.map(entry =>
    `<tr>
      <td style="font-weight:600">${entry.disease}</td>
      <td style="font-size:.75rem;color:var(--sub)">${entry.medications.slice(0, 2).join(", ")}${entry.medications.length > 2 ? " +" + (entry.medications.length - 2) + " more" : ""}</td>
      <td style="font-size:.75rem;color:var(--sub)">${entry.precautions.slice(0, 2).join("; ")}${entry.precautions.length > 2 ? "…" : ""}</td>
    </tr>`
  ).join("");
}

/* ─── ADMIN STATS ─── */
async function loadModelStats() {
  const data = await apiFetch("/api/models/stats");
  if (!data.success) return;
  const dtEl = document.getElementById("dtAccuracy");
  const rfEl = document.getElementById("rfAccuracy");
  const dsEl = document.getElementById("dsCount");
  const syEl = document.getElementById("symCountStat");
  if (dtEl) dtEl.textContent = data.models.decision_tree.accuracy + "%";
  if (rfEl) rfEl.textContent = data.models.random_forest.accuracy + "%";
  if (dsEl) dsEl.textContent = data.dataset.total_diseases;
  if (syEl) syEl.textContent = data.dataset.total_symptoms;
}

function aTab(id, btn) {
  document.querySelectorAll(".atab").forEach(t => t.classList.remove("on"));
  document.querySelectorAll(".apanel").forEach(p => p.classList.remove("on"));
  btn.classList.add("on");
  document.getElementById("ap-" + id).classList.add("on");
  if (id === "data")  loadAdminTable();
  if (id === "stats") loadModelStats();
}

/* ─── NAV ACTIVE ─── */
window.addEventListener("scroll", () => {
  const ids = ["home", "diagnose", "how", "algorithms", "admin"];
  let cur = "home";
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el && window.scrollY >= el.offsetTop - 80) cur = id;
  });
  document.querySelectorAll(".nav-links a").forEach(a => {
    const href = a.getAttribute("href").replace("#", "");
    a.classList.toggle("active", href === cur || (href === "diagnose" && cur === "home"));
  });
}, { passive: true });

/* ─── SPINNER CSS ─── */
const s = document.createElement("style");
s.textContent = `.loading-spinner{width:32px;height:32px;border-radius:50%;border:3px solid var(--teal-md);border-top-color:var(--teal);animation:spin .7s linear infinite;margin:0 auto}@keyframes spin{to{transform:rotate(360deg)}}`;
document.head.appendChild(s);

/* ─── INIT ─── */
initSymptoms();
loadAdminTable();
loadModelStats();

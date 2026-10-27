(() => {
  "use strict";

  // This page is intentionally bound to the project's Student Performance model.
  // The dataset and model are selected by the backend registry; no dataset picker
  // or model picker is added to the student-facing page.
  const DATASET_ID = "student_performance";
  let reportRequest = 0;

  const $ = (selector) => document.querySelector(selector);
  const escHtml = (value) => esc(value == null ? "" : String(value));
  const showNumber = (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 });

  function modelCompatibilityNote(report) {
    const selected = report?.selected?.algorithm;
    const model = selected ? report?.models?.[selected] : null;
    const limitations = model?.limitations || [];
    return `<div class="warn"><b>Important model limitation:</b> the configured Student Performance model predicts a historical pass/fail category from 31 background/behavior inputs. It excludes semester marks and does not accept attendance percentage or fee status. This project has no student-ID lookup, roster, or stored academic-record endpoint. To avoid inventing features, this page does not send partial data to the model or display a fabricated ML prediction.${model ? `<br><b>Backend-selected model:</b> ${escHtml(selected.replaceAll("_", " "))}.` : ""}${limitations.length ? `<br><b>Other model limitations:</b> ${limitations.filter((x) => !/parent|alcohol|romantic/i.test(x)).map(escHtml).join(" ")}` : ""}</div>`;
  }

  V.predict = () => `<div class="wrap">${crumb("predict")}<h1 style="font-size:2.4rem">Student performance</h1><p style="color:var(--mu);font-size:1.05rem">Review the student's reported marks and attendance, see the actual two-point trend, and get supportive next steps.</p><div id="student-model-note"><div class="sk"></div></div><div class="grid g2"><form class="glass" id="student-snapshot-form" novalidate><h2>Student snapshot</h2><p style="color:var(--mu)">Enter only information you have. Student ID is for this screen only; it is not looked up or saved.</p><label for="sp-id">Student ID / roll number <span style="color:var(--mu);font-weight:400">(optional)</span></label><input id="sp-id" name="student_id" maxlength="80" autocomplete="off" placeholder="Optional reference"><label for="sp-age">Student age <span style="color:var(--mu);font-weight:400">(optional)</span></label><input id="sp-age" name="age" type="number" min="3" max="100" step="1" inputmode="numeric" placeholder="Optional"><small style="color:var(--mu)">Age is shown for context only; it is not used to generate a prediction.</small><label for="sp-first">First available semester marks (out of 20)</label><input id="sp-first" name="first_marks" type="number" min="0" max="20" step="0.1" required inputmode="decimal"><label for="sp-latest">Latest available semester marks (out of 20)</label><input id="sp-latest" name="latest_marks" type="number" min="0" max="20" step="0.1" required inputmode="decimal"><label for="sp-attendance">Attendance percentage</label><input id="sp-attendance" name="attendance" type="number" min="0" max="100" step="0.1" required inputmode="decimal"><label for="sp-fees">Fee-payment status <span style="color:var(--mu);font-weight:400">(optional; not used to assess academic ability)</span></label><select id="sp-fees" name="fee_status"><option value="">Not provided</option><option>Paid</option><option>Partially paid</option><option>Unpaid</option></select><div class="warn">Marks are shown on the entered 0–20 scale. This page does not apply a pass/fail or risk threshold because the relevant school grading criteria are not available.</div><div class="err" id="sp-error" role="alert"></div><div class="flex"><button class="btn pri" type="submit">Review snapshot</button><button class="btn" type="button" id="sp-clear">Clear</button></div></form><section class="glass" id="sp-result" aria-live="polite"><h2>Results</h2><p style="color:var(--mu)">Submit two actual marks and attendance to see a concise performance snapshot. No class-average or model prediction will be invented.</p></section></div><div class="glass" style="margin-top:18px"><h2>Class comparison</h2><p style="color:var(--mu)">Unavailable: this project does not include a student roster or class-level records. A reliable class average cannot be calculated from a single student's inputs.</p></div><div class="glass" style="margin-top:18px"><h2>What this page can—and cannot—tell you</h2><p>This view summarizes the two actual marks and attendance value you enter. It does not retrieve a student record, predict future marks, or assign a Low/Medium/High risk label. The bundled model's target is pass/fail on historical final grades; its features do not match this short form. A compatible prediction requires a validated model trained on the available inputs or an authorized student record source; neither is included here.</p><p style="color:var(--mu);font-size:.9rem">Information entered here stays in this page's current browser session and is not sent to the prediction API or saved by this enhancement. Fee status is displayed only for reference and never changes academic status or recommendations.</p></div></div>`;

  async function loadModelContext() {
    const note = $("#student-model-note");
    if (!note || route() !== "predict") return;
    const requestId = ++reportRequest;
    try {
      const response = await fetch(`/api/ml/datasets/${encodeURIComponent(DATASET_ID)}/report`, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`Model report unavailable (${response.status})`);
      const report = await response.json();
      if (requestId !== reportRequest || route() !== "predict" || !$("#student-model-note")) return;
      note.innerHTML = modelCompatibilityNote(report);
    } catch (error) {
      if (requestId !== reportRequest || route() !== "predict" || !$("#student-model-note")) return;
      note.innerHTML = `<div class="warn"><b>Model details could not be loaded.</b> ${escHtml(error.message)} The student snapshot remains available, but no prediction will be fabricated.</div>`;
    }
  }

  function barChart(first, latest) {
    const width = 620, height = 250, left = 56, right = 24, top = 24, bottom = 52;
    const y = (v) => top + (20 - v) / 20 * (height - top - bottom);
    const x1 = left + 72, x2 = width - right - 72;
    const grid = [0, 5, 10, 15, 20].map((v) => `<line x1="${left}" y1="${y(v)}" x2="${width-right}" y2="${y(v)}" stroke="rgba(154,166,214,.22)"/><text x="${left-12}" y="${y(v)+4}" text-anchor="end" fill="#9aa6d6" font-size="12">${v}</text>`).join("");
    return `<div style="overflow-x:auto"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Actual marks: first available ${first} out of 20 and latest available ${latest} out of 20" style="width:100%;min-width:320px;height:auto"><text x="${left}" y="15" fill="#9aa6d6" font-size="12">Marks out of 20 · actual values only</text>${grid}<line x1="${x1}" y1="${y(first)}" x2="${x2}" y2="${y(latest)}" stroke="#22e4ff" stroke-width="3"/><circle cx="${x1}" cy="${y(first)}" r="7" fill="#22e4ff"><title>First available: ${first} / 20 (actual)</title></circle><circle cx="${x2}" cy="${y(latest)}" r="7" fill="#8b5cf6"><title>Latest available: ${latest} / 20 (actual)</title></circle><text x="${x1}" y="${height-24}" text-anchor="middle" fill="#e8ecff" font-size="12">First available</text><text x="${x2}" y="${height-24}" text-anchor="middle" fill="#e8ecff" font-size="12">Latest available</text><text x="${x1}" y="${y(first)-13}" text-anchor="middle" fill="#e8ecff" font-size="12">${showNumber(first)}</text><text x="${x2}" y="${y(latest)-13}" text-anchor="middle" fill="#e8ecff" font-size="12">${showNumber(latest)}</text></svg></div>`;
  }

  function marksDonut(mark) {
    const radius = 42, circumference = 2 * Math.PI * radius, earned = circumference * mark / 20;
    return `<div class="flex" style="gap:18px;align-items:center"><svg viewBox="0 0 120 120" role="img" aria-label="${showNumber(mark)} of 20 marks earned; ${showNumber(20-mark)} of 20 remaining" style="width:112px;height:112px;flex:none;transform:rotate(-90deg)"><circle cx="60" cy="60" r="${radius}" fill="none" stroke="rgba(154,166,214,.25)" stroke-width="16"/><circle cx="60" cy="60" r="${radius}" fill="none" stroke="#22e4ff" stroke-width="16" stroke-dasharray="${earned} ${circumference-earned}" stroke-linecap="round"/></svg><div><b>${showNumber(mark)} / 20</b><br><small style="color:var(--mu)">Latest actual mark<br>${showNumber(20-mark)} points remain on this scale</small><p style="font-size:.8rem;color:var(--mu)">Mark breakdown only; not a class distribution or probability.</p></div></div>`;
  }

  function recommendations(data) {
    const out = [];
    if (data.latest < data.first) {
      out.push(`<b>Marks declined by ${showNumber(data.first-data.latest)} points.</b> Review the topics covered between the two recorded assessments, identify questions that were difficult, and agree a short follow-up check with a teacher.`);
    } else if (data.latest > data.first) {
      out.push(`<b>Marks improved by ${showNumber(data.latest-data.first)} points.</b> Keep the study habits that supported this progress and continue checking understanding with regular practice.`);
    } else {
      out.push(`<b>Marks are unchanged across the two entered points.</b> Try a different revision approach for topics that remain difficult and ask a teacher for a focused progress check.`);
    }
    if (data.attendance < 100) {
      out.push(`<b>Attendance entered: ${showNumber(data.attendance)}%.</b> Check whether missed lessons left gaps, and offer a practical catch-up plan. No attendance risk cutoff is applied here.`);
    } else {
      out.push(`<b>Attendance entered: 100%.</b> Maintain regular participation and keep checking academic understanding as well as attendance.`);
    }
    return out;
  }

  function renderSnapshot(data) {
    const trend = data.latest > data.first ? "Improving" : data.latest < data.first ? "Declining" : "Unchanged";
    const student = data.id ? `<b>${escHtml(data.id)}</b>` : "Not provided";
    const age = data.age === null ? "Not provided" : `${data.age} years (context only)`;
    const fee = data.fee || "Not provided";
    const advice = recommendations(data);
    return `<h2>Student performance snapshot</h2><div class="grid g3"><div class="msg"><small>Student reference</small><br><b>${student}</b></div><div class="msg"><small>Recorded mark trend</small><br><b>${trend}</b></div><div class="msg"><small>Attendance entered</small><br><b>${showNumber(data.attendance)}%</b></div></div><div class="warn"><b>No ML prediction or risk level is available for these inputs.</b> This result summarizes only the actual values entered; it is not a predicted grade or diagnosis.</div><div class="grid g2"><section class="glass"><h3>Actual marks across available points</h3><p style="color:var(--mu)">Only the two provided observations are plotted; no intermediate or future marks are inferred.</p>${barChart(data.first,data.latest)}<div class="flex"><span class="pill">First available: ${showNumber(data.first)} / 20 · actual</span><span class="pill">Latest available: ${showNumber(data.latest)} / 20 · actual</span></div></section><section class="glass"><h3>Latest mark breakdown</h3>${marksDonut(data.latest)}<p><small style="color:var(--mu)">Age: ${escHtml(age)} · Fee status: ${escHtml(fee)} (display only; not used to infer ability)</small></p></section></div><section class="glass" style="margin-top:18px"><h3>Academic results</h3><div style="overflow-x:auto"><table><thead><tr><th>Record</th><th>Marks</th><th>Status</th><th>Type</th></tr></thead><tbody><tr><td>First available result</td><td>${showNumber(data.first)} / 20</td><td>Actual result; school criteria not provided</td><td>Actual, user-entered</td></tr><tr><td>Latest available result</td><td>${showNumber(data.latest)} / 20</td><td>Actual result; school criteria not provided</td><td>Actual, user-entered</td></tr></tbody></table></div></section><section class="glass" style="margin-top:18px"><h3>Suggested next steps</h3><p style="color:var(--mu)">Suggestions are based only on the marks and attendance entered. They do not identify weak subjects or personal circumstances.</p><ol style="padding-left:22px">${advice.map((line) => `<li style="margin:10px 0">${line}</li>`).join("")}</ol></section>`;
  }

  document.addEventListener("submit", (event) => {
    if (event.target.id !== "student-snapshot-form") return;
    event.preventDefault();
    const form = event.target;
    const error = $("#sp-error");
    const firstRaw = $("#sp-first").value.trim();
    const latestRaw = $("#sp-latest").value.trim();
    const attendanceRaw = $("#sp-attendance").value.trim();
    const values = [["First available marks", firstRaw, 0, 20], ["Latest available marks", latestRaw, 0, 20], ["Attendance", attendanceRaw, 0, 100]];
    for (const [name, raw, min, max] of values) {
      const n = Number(raw);
      if (raw === "" || !Number.isFinite(n) || n < min || n > max) {
        error.textContent = `${name} is required and must be between ${min} and ${max}${name.includes("marks") ? " out of 20" : "%"}.`;
        form.querySelector(`[name="${name.startsWith("First") ? "first_marks" : name.startsWith("Latest") ? "latest_marks" : "attendance"}"]`).focus();
        return;
      }
    }
    const ageRaw = $("#sp-age").value.trim();
    const age = ageRaw === "" ? null : Number(ageRaw);
    if (ageRaw !== "" && (!Number.isInteger(age) || age < 3 || age > 100)) {
      error.textContent = "Age must be a whole number from 3 to 100, or left blank.";
      $("#sp-age").focus();
      return;
    }
    error.textContent = "";
    const snapshot = {
      id: $("#sp-id").value.trim(), age,
      first: Number(firstRaw), latest: Number(latestRaw), attendance: Number(attendanceRaw),
      fee: $("#sp-fees").value,
    };
    $("#sp-result").innerHTML = renderSnapshot(snapshot);
    toast("Actual student snapshot ready. No ML prediction was generated.");
    $("#sp-result").scrollIntoView({ behavior: "smooth", block: "start" });
  }, true);

  document.addEventListener("click", (event) => {
    if (event.target.id !== "sp-clear") return;
    $("#student-snapshot-form")?.reset();
    $("#sp-error").textContent = "";
    $("#sp-result").innerHTML = '<h2>Results</h2><p style="color:var(--mu)">Submit two actual marks and attendance to see a concise performance snapshot. No class-average or model prediction will be invented.</p>';
  });

  window.addEventListener("hashchange", () => {
    if (route() === "predict") setTimeout(loadModelContext, 250);
    else reportRequest++;
  });
  if (route() === "predict") {
    render();
    setTimeout(loadModelContext, 500);
  }
})();

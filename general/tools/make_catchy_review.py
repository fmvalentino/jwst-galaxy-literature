"""Write general/outputs/catchy_titles_review.html: a local page to review and change the hand labels of
titles (catchy / borderline / not) in data/<freeze>/catchy_title_labels.csv.

The labels are embedded when the page is built. Clicks are kept in the browser while you work; "Save
labels" downloads catchy_title_labels_edits_<date>.json (every title and its current label), which is then
merged into the CSV. The page opens on the titles new in this freeze (column `reviewed` naming a
labeller other than the author and not yet reviewed by the author). Rebuild after the CSV changes:
    python general/tools/make_catchy_review.py
"""
import json
import os
import sys
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from common.config import FREEZE, GEN_OUT, data   # noqa: E402

labels = pd.read_csv(data("catchy_title_labels.csv"))
new = labels["reviewed"].astype(str).str.contains(FREEZE) & ~labels["reviewed_by_user"].astype(str).eq("True")
# Compact records for the page: b = bibcode, y = year, s = who labelled it, l = label, p = the part of the
# title before the separator (shown in blue), t = full title, n = new in this freeze.
rows = [{"b": r.bibcode, "y": int(r.year) if pd.notna(r.year) else "",
         "s": "Claude, " + FREEZE if n else ("author" if str(r.reviewed_by_user) == "True" else "Claude, 2026-10-07"),
         "l": r.label, "p": r.prefix if isinstance(r.prefix, str) else "", "t": r.title, "n": bool(n)}
        for r, n in zip(labels.itertuples(), new)]
data_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")

page = """<!doctype html>
<html><head><meta charset="utf-8"><title>Catchy title labels</title>
<style>
body { font: 14px/1.4 -apple-system, Helvetica, Arial, sans-serif; margin: 16px; }
table { border-collapse: collapse; width: 100%; }
td, th { border-bottom: 1px solid #ddd; padding: 4px 6px; vertical-align: top; text-align: left; }
th { position: sticky; top: 0; background: #f4f4f4; }
td.lab { white-space: nowrap; }
tr.changed { background: #fff6d6; }
b { color: #1c5cab; }
.bar { position: sticky; top: 0; background: #fff; padding: 6px 0; z-index: 2; }
</style></head><body>
<h3>Catchy title labels</h3>
<p>Choose a label per title (rules: <code>general/03_topics_sample_titles.ipynb</code>, section 6). The page opens on the
titles new in this freeze, labelled by Claude and still to be reviewed. Labels are kept in this browser while you work.
When done, click <b>Save labels</b>: it downloads <code>catchy_title_labels_edits___DATE__.json</code>. Put it in
<code>data/<freeze>/</code> (or leave it in Downloads) and tell Claude, who applies it to <code>catchy_title_labels.csv</code>.</p>
<div class="bar">
Show: <select id="show"><option value="new" selected>new in this freeze (to review)</option><option value="">all</option><option value="catchy">catchy</option>
<option value="borderline">borderline</option><option value="not">not</option>
<option value="changed">changed only</option><option value="newcb">new: catchy or borderline</option></select>
Search: <input id="q" size="30">
<button id="save">Save labels</button> <span id="status"></span>
</div>
<table><thead><tr><th>Label</th><th>Year</th><th>Labelled by</th><th>Title (prefix in blue)</th><th>ADS</th></tr></thead>
<tbody id="tb"></tbody></table>
<script>
const ROWS = __DATA__;
const ORIG = Object.fromEntries(ROWS.map(r => [r.b, r.l]));
let cur = {};
try { cur = JSON.parse(localStorage.getItem("catchy_labels___DATE__") || "{}"); } catch (e) {}
const lab = b => cur[b] || ORIG[b];
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
function render() {
  const show = document.getElementById("show").value, q = document.getElementById("q").value.toLowerCase();
  let n = 0;
  document.getElementById("tb").innerHTML = ROWS.filter(r =>
      (!show || (show === "changed" ? lab(r.b) !== ORIG[r.b] : show === "new" ? r.n
                : show === "newcb" ? r.n && lab(r.b) !== "not" : lab(r.b) === show)) &&
      (!q || r.t.toLowerCase().includes(q))).map(r => {
    n++;
    const i = r.t.indexOf(r.p), title = i >= 0 ? esc(r.t.slice(0, i)) + "<b>" + esc(r.p) + "</b>" + esc(r.t.slice(i + r.p.length)) : esc(r.t);
    const radios = ["catchy", "borderline", "not"].map(v =>
      `<label><input type="radio" name="${esc(r.b)}" value="${v}" ${lab(r.b) === v ? "checked" : ""}> ${v}</label>`).join(" ");
    return `<tr class="${lab(r.b) !== ORIG[r.b] ? "changed" : ""}"><td class="lab">${radios}</td><td>${r.y}</td><td>${esc(r.s)}</td><td>${title}</td><td><a href="https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(r.b)}/abstract" target="_blank">ADS</a></td></tr>`;
  }).join("");
  const ch = ROWS.filter(r => lab(r.b) !== ORIG[r.b]).length;
  document.getElementById("status").textContent = `${n} shown, ${ch} changed`;
}
document.getElementById("tb").addEventListener("change", e => {
  if (e.target.type !== "radio") return;
  cur[e.target.name] = e.target.value;
  try { localStorage.setItem("catchy_labels___DATE__", JSON.stringify(cur)); } catch (err) {}
  e.target.closest("tr").className = e.target.value !== ORIG[e.target.name] ? "changed" : "";
  const ch = ROWS.filter(r => lab(r.b) !== ORIG[r.b]).length;
  document.getElementById("status").textContent = `${ch} changed`;
});
document.getElementById("show").onchange = render;
document.getElementById("q").oninput = render;
document.getElementById("save").onclick = () => {
  const out = Object.fromEntries(ROWS.map(r => [r.b, lab(r.b)]));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "catchy_title_labels_edits___DATE__.json";
  a.click();
};
render();
</script></body></html>
""".replace("__DATA__", data_json).replace("__DATE__", str(date.today()))

out = os.path.join(GEN_OUT, "catchy_titles_review.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(page)
print(f"wrote {os.path.relpath(out, ROOT)} ({len(rows)} titles, {int(new.sum())} new in this freeze)")

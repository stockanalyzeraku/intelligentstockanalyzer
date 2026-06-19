import os
import re
import json

ROOT = "."
IGNORE = {"__pycache__", ".git", "uploads", "logs", ".venv", "node_modules"}

def find_py_files():
    files = []
    for dirpath, dirs, filenames in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE]
        for f in filenames:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
                files.append(rel.replace("\\", "/"))
    return sorted(files)

def parse_file(path):
    imports, functions, classes = [], [], []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip()
                # imports
                m = re.match(r'^(?:from|import)\s+([\w.]+)', line)
                if m:
                    mod = m.group(1)
                    # only project-local imports
                    if not mod.startswith(("os","sys","re","json","typing","pathlib",
                                           "dataclasses","datetime","enum","uuid",
                                           "hashlib","sqlite3","contextlib","logging",
                                           "concurrent","functools","random","threading",
                                           "time","subprocess","shutil","tempfile",
                                           "numpy","chromadb","mistralai","fitz",
                                           "langchain","langgraph","sentence_transformers",
                                           "__future__")):
                        imports.append(mod.replace(".", "/"))
                # functions
                m2 = re.match(r'^    def (\w+)\(', line)
                if m2: functions.append(m2.group(1) + "()")
                m3 = re.match(r'^def (\w+)\(', line)
                if m3: functions.append(m3.group(1) + "()")
                # classes
                m4 = re.match(r'^class (\w+)', line)
                if m4: classes.append(m4.group(1))
    except Exception as e:
        print(f"  [skip] {path}: {e}")
    return list(dict.fromkeys(imports)), list(dict.fromkeys(functions)), list(dict.fromkeys(classes))

def build_map():
    files = find_py_files()
    module_map = {}

    for f in files:
        imports, functions, classes = parse_file(f)
        key = f.replace(".py", "").lstrip("./")
        module_map[key] = {
            "file": f,
            "imports": imports,
            "functions": functions,
            "classes": classes,
            "used_by": []
        }

    # build reverse: who imports whom
    for key, data in module_map.items():
        for imp in data["imports"]:
            # fuzzy match: imp might be "config" matching "config" or "codebase/ragrun/config"
            for other_key in module_map:
                if other_key == imp or other_key.endswith("/" + imp) or other_key.endswith("/" + imp.split("/")[-1]):
                    if key not in module_map[other_key]["used_by"]:
                        module_map[other_key]["used_by"].append(key)

    return module_map

def write_html(module_map):
    nodes_js = json.dumps(module_map, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Project Map — Sample-1</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0 }}
  body {{ font-family: system-ui, sans-serif; background: #f5f5f5; color: #222; }}
  header {{ background: #fff; border-bottom: 1px solid #e0e0e0; padding: 14px 24px; display:flex; align-items:center; gap:16px }}
  header h1 {{ font-size: 16px; font-weight: 600 }}
  input {{ flex:1; padding: 6px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px }}
  #stats {{ font-size: 12px; color: #888 }}
  main {{ max-width: 900px; margin: 24px auto; padding: 0 16px }}
  .group-label {{ font-size: 11px; font-weight: 600; color: #888; text-transform: uppercase;
                  letter-spacing: .06em; margin: 18px 0 6px }}
  .card {{ background: #fff; border: 1px solid #e8e8e8; border-radius: 8px;
           padding: 12px 14px; margin-bottom: 6px; cursor: pointer; transition: border-color .15s }}
  .card:hover {{ border-color: #aaa }}
  .card.active {{ border-color: #5563DE; background: #f0f2ff }}
  .card-header {{ display:flex; align-items:center; gap:8px }}
  .card-name {{ font-size: 13px; font-weight: 600; font-family: monospace }}
  .badge {{ font-size: 10px; padding: 2px 7px; border-radius: 20px;
            background: #eee; color: #555; font-weight: 500 }}
  .card-detail {{ margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px; display:none }}
  .card-detail.open {{ display: block }}
  .section-label {{ font-size: 11px; font-weight: 600; color: #aaa; text-transform: uppercase;
                    letter-spacing:.05em; margin-bottom: 4px; margin-top: 8px }}
  .chip-row {{ display:flex; flex-wrap:wrap; gap:4px; margin-bottom:4px }}
  .chip {{ font-size: 11px; padding: 2px 8px; border-radius: 20px; border: 1px solid #e0e0e0;
           color: #555; cursor: pointer; background: #f8f8f8 }}
  .chip:hover {{ border-color: #5563DE; color: #5563DE }}
  .fn-row {{ display:flex; flex-wrap:wrap; gap:4px }}
  .fn {{ font-size: 11px; font-family: monospace; padding: 2px 7px; border-radius: 4px;
         background: #f3f3f3; color: #444 }}
  .cls {{ font-size: 11px; font-family: monospace; padding: 2px 7px; border-radius: 4px;
          background: #e8f0ff; color: #3344bb }}
  .empty {{ text-align:center; color:#aaa; padding:40px; font-size:14px }}
</style>
</head>
<body>
<header>
  <h1>📦 Project Map</h1>
  <input type="text" id="search" placeholder="Search files, functions, classes…" oninput="filter(this.value)">
  <span id="stats"></span>
</header>
<main id="main"></main>

<script>
const DATA = {nodes_js};

// group by top-level folder
function groupKey(key) {{
  const parts = key.split("/");
  if (parts.length === 1) return "root";
  if (parts[0] === "codebase") return parts[1] || "codebase";
  return parts[0];
}}

let active = null;

function render(q) {{
  const main = document.getElementById("main");
  main.innerHTML = "";
  q = (q || "").toLowerCase();

  const groups = {{}};
  let total = 0;
  for (const [key, data] of Object.entries(DATA)) {{
    const match = !q ||
      key.toLowerCase().includes(q) ||
      data.functions.some(f => f.toLowerCase().includes(q)) ||
      data.classes.some(c => c.toLowerCase().includes(q));
    if (!match) continue;
    const gk = groupKey(key);
    if (!groups[gk]) groups[gk] = [];
    groups[gk].push([key, data]);
    total++;
  }}

  document.getElementById("stats").textContent = total + " files";

  if (!total) {{ main.innerHTML = '<div class="empty">No results</div>'; return; }}

  for (const [grp, items] of Object.entries(groups)) {{
    const gl = document.createElement("div");
    gl.className = "group-label"; gl.textContent = grp;
    main.appendChild(gl);

    for (const [key, data] of items) {{
      const card = document.createElement("div");
      card.className = "card" + (key === active ? " active" : "");
      card.dataset.key = key;

      const fnHtml = data.functions.map(f => `<span class="fn">${{f}}</span>`).join("");
      const clsHtml = data.classes.map(c => `<span class="cls">${{c}}</span>`).join("");
      const impHtml = data.imports.map(i => `<span class="chip" onclick="jump('${{i}}');event.stopPropagation()">${{i.split("/").pop()}}</span>`).join("") || "<span style='color:#bbb;font-size:11px'>none</span>";
      const usedHtml = data.used_by.map(u => `<span class="chip" onclick="jump('${{u}}');event.stopPropagation()">${{u.split("/").pop()}}</span>`).join("") || "<span style='color:#bbb;font-size:11px'>none</span>";

      card.innerHTML = `
        <div class="card-header">
          <span class="badge">${{grp}}</span>
          <span class="card-name">${{data.file}}</span>
        </div>
        <div class="card-detail${{key === active ? " open" : ""}}">
          ${{clsHtml || fnHtml ? `<div class="section-label">Classes & Functions</div><div class="fn-row">${{clsHtml}}${{fnHtml}}</div>` : ""}}
          <div class="section-label">Imports</div><div class="chip-row">${{impHtml}}</div>
          <div class="section-label">Used by</div><div class="chip-row">${{usedHtml}}</div>
        </div>`;

      card.addEventListener("click", () => toggle(key));
      main.appendChild(card);
    }}
  }}
}}

function toggle(key) {{
  active = active === key ? null : key;
  render(document.getElementById("search").value);
  if (active) {{
    setTimeout(() => {{
      const el = document.querySelector(".card.active");
      if (el) el.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
    }}, 50);
  }}
}}

function jump(key) {{
  // try exact, then suffix match
  const exact = DATA[key];
  if (exact) {{ active = key; }}
  else {{
    const match = Object.keys(DATA).find(k => k.endsWith("/" + key) || k === key);
    if (match) active = match;
  }}
  document.getElementById("search").value = "";
  render("");
  setTimeout(() => {{
    const el = document.querySelector(".card.active");
    if (el) el.scrollIntoView({{ behavior: "smooth", block: "center" }});
  }}, 60);
}}

function filter(q) {{ render(q); }}

render("");
</script>
</body>
</html>"""

    with open("project_map.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅  project_map.html written!")

if __name__ == "__main__":
    m = build_map()
    print(f"Found {len(m)} Python files")
    write_html(m)
"""Extract neg-id encoding output from the executed 02c notebook."""
import json
nb = json.load(open(r"C:\Users\Alexander\Documents\CFB\CFBapp\research\notebooks\02c_explosive_vs_sustained.executed.ipynb", encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    txt = ""
    for o in cell.get("outputs", []):
        if "text" in o:
            t = o["text"]
            if isinstance(t, list):
                txt += "".join(t)
            else:
                txt += str(t)
    if "negative-id" in txt or "CFBD negative" in txt or "cache-hit assertion" in txt or "loaded from cache" in txt:
        print(f"--- cell {i} (id={cell.get('id')}) ---")
        print(txt)
        print()

"""Extract key prints from the executed 02b notebook."""
import json

nb = json.load(
    open(
        r"C:\Users\Alexander\Documents\CFB\CFBapp\research\notebooks\02b_opening_drive_shock.executed.ipynb",
        encoding="utf-8",
    )
)
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    txt = ""
    for o in cell.get("outputs", []):
        if "text" in o:
            t = o["text"]
            txt += "".join(t) if isinstance(t, list) else str(t)
        elif o.get("output_type") == "error":
            txt += f"\n[ERROR] {o.get('ename')}: {o.get('evalue')}\n"
    if txt.strip():
        print(f"=== cell {i} (id={cell.get('id')}) ===")
        print(txt.strip())
        print()

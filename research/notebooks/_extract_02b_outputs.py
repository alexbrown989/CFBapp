"""Extract execution report payload from 02b_opening_drive_shock.executed.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("research/notebooks/02b_opening_drive_shock.executed.ipynb")
SRC_PATH = Path("research/notebooks/02b_opening_drive_shock.ipynb")
CSV_PATH = Path("research/results/feature_validation.csv")
SCHEMA_PATH = Path("research/results/feature_validation.schema.md")


def cell_outputs(cell: dict) -> str:
    parts: list[str] = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            txt = out.get("text", "")
            if isinstance(txt, list):
                txt = "".join(txt)
            parts.append(txt)
        elif out.get("output_type") in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                txt = data["text/plain"]
                if isinstance(txt, list):
                    txt = "".join(txt)
                parts.append(txt)
        elif out.get("output_type") == "error":
            parts.append(
                "[ERROR " + out.get("ename", "?") + "] " + out.get("evalue", "")
            )
    return "".join(parts)


def has_warning(cell: dict) -> bool:
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream" and out.get("name") == "stderr":
            return True
    return False


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    print(f"=== executed notebook: {NB_PATH} ===")
    print(f"code cells: {len(cells)}")
    print()
    warnings: list[tuple[int, str]] = []
    for idx, c in enumerate(cells, start=1):
        meta_id = c.get("id", "?")
        src_first = "".join(c.get("source", []))[:80].replace("\n", " ")
        out_text = cell_outputs(c)
        marker = " [STDERR]" if has_warning(c) else ""
        if has_warning(c):
            warnings.append((idx, meta_id))
        print(f"--- CELL {idx} id={meta_id}{marker} ---")
        print(f"  src head: {src_first}")
        if out_text.strip():
            print("  outputs:")
            for line in out_text.rstrip().splitlines():
                print(f"    {line}")
        else:
            print("  outputs: (none)")
        print()
    print("=== summary ===")
    print(f"cells with stderr output: {len(warnings)}")
    for idx, mid in warnings:
        print(f"  cell {idx} id={mid}")
    print()
    print(f"file sizes (bytes):")
    for p in (SRC_PATH, NB_PATH, CSV_PATH, SCHEMA_PATH):
        if p.exists():
            print(f"  {p}: {p.stat().st_size}")
        else:
            print(f"  {p}: MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

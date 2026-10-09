"""Audit PrimGate-SC histories: val F1-only vs composite (F1 - beta*G1) selection."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "experiments" / "primgate_sc"
BETA = 0.35  # 0.1 G1 drop ≈ +0.035 F1 worth


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def pick(hist, key_fn):
    best_i, best_v = 0, float("-inf")
    for i, r in enumerate(hist):
        v = key_fn(r)
        if v > best_v:
            best_v, best_i = v, i
    return best_i, hist[best_i]


def main():
    rows = []
    for hist_path in sorted(SC.glob("*/history.json")):
        run = hist_path.parent.name
        hist = load(hist_path)
        if not hist or hist[0].get("val_G1") is None:
            continue
        res_path = hist_path.parent / "result.json"
        test = load(res_path) if res_path.exists() else {}
        i_f1, r_f1 = pick(hist, lambda r: float(r["val_macro_f1"]))
        i_c, r_c = pick(
            hist,
            lambda r: float(r["val_macro_f1"]) - BETA * float(r["val_G1"]),
        )
        rows.append(
            {
                "run": run,
                "f1_pick_epoch": r_f1["epoch"],
                "f1_pick_val_f1": r_f1["val_macro_f1"],
                "f1_pick_val_G1": r_f1["val_G1"],
                "comp_pick_epoch": r_c["epoch"],
                "comp_pick_val_f1": r_c["val_macro_f1"],
                "comp_pick_val_G1": r_c["val_G1"],
                "epoch_delta": r_c["epoch"] - r_f1["epoch"],
                "test_macro_f1": test.get("test_macro_f1"),
                "test_G1": test.get("test_G1"),
            }
        )

    out = {"beta": BETA, "runs": rows}
    out_dir = SC / "selection_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# SC checkpoint selection audit",
        "",
        f"Composite score = val_macro_f1 - {BETA} * val_G1",
        "",
        "| Run | F1-pick ep | val F1 | val G1 | Comp-pick ep | val F1 | val G1 | d_ep | test F1 | test G1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run'][:40]} | {r['f1_pick_epoch']} | {r['f1_pick_val_f1']:.3f} | "
            f"{r['f1_pick_val_G1']:.3f} | {r['comp_pick_epoch']} | {r['comp_pick_val_f1']:.3f} | "
            f"{r['comp_pick_val_G1']:.3f} | {r['epoch_delta']:+d} | "
            f"{(r['test_macro_f1'] or float('nan')):.3f} | {(r['test_G1'] or float('nan')):.3f} |"
        )
    lines += [
        "",
        "## Takeaway",
        "- If Comp-pick epoch differs from F1-pick, F1-only selection ignored better G1 tradeoffs.",
        "- Retrain with composite selection + stronger regularization for Gate C v3.2.",
        "",
    ]
    (out_dir / "selection_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

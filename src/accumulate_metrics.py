#!/usr/bin/env python3
"""Merge the current (per-segment, overwritten-on-resume) experiment_log.jsonl into a
persistent metrics_master.jsonl, deduped by (step, is_val). Newer rows win. Idempotent.
Call at each chain-segment START (before verl truncates the log) so NO segment's metrics
(esp. val reward) are lost across the self-resuming chain. For paper training curves."""
import json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/grpo_longdoc"
master = os.path.join(OUT, "metrics_master.jsonl")
cur = os.path.join(OUT, "experiment_log.jsonl")

def is_val(r):
    return "val" in r and isinstance(r.get("val"), dict)

seen = {}
for src in (master, cur):
    if os.path.exists(src):
        for line in open(src):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "step" not in r:
                continue
            seen[(r["step"], is_val(r))] = r  # newer source (cur) wins

if not seen:
    print("[accumulate] nothing to merge")
    sys.exit(0)

with open(master, "w") as f:
    for k in sorted(seen):
        f.write(json.dumps(seen[k]) + "\n")

tr = [r for k, r in seen.items() if not k[1]]
vr = [r for k, r in seen.items() if k[1]]
vsteps = sorted(r["step"] for r in vr)
print(f"[accumulate] master: {len(tr)} train + {len(vr)} val rows; "
      f"val steps {vsteps}")


import json
import statistics
from pathlib import Path
 
EVAL_DIR = Path(__file__).parent
CONDITIONS = {
    "detection_only": EVAL_DIR / "debug_predictions.jsonl",
}
 
 
def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
 
 
def summarize(rows):
    if not rows:
        return None
    # cpu_percent/ram_percent live inside the "final" result dict
    cpu = [r["final"]["cpu_percent"] for r in rows if r.get("final", {}).get("cpu_percent") is not None]
    ram = [r["final"]["ram_percent"] for r in rows if r.get("final", {}).get("ram_percent") is not None]
    return {
        "n_samples": len(rows),
        "cpu_mean": round(statistics.mean(cpu), 1) if cpu else None,
        "cpu_max": round(max(cpu), 1) if cpu else None,
        "ram_mean": round(statistics.mean(ram), 1) if ram else None,
        "ram_max": round(max(ram), 1) if ram else None,
    }
 
 
def run_all():
    out = {}
    for name, path in CONDITIONS.items():
        rows = load_jsonl(path)
        out[name] = summarize(rows) or {"note": f"{path.name} not found run this condition and copy debug_predictions.jsonl here first"}
    return out
 
 
if __name__ == "__main__":
    result = run_all()
    out_path = EVAL_DIR / "resource_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")
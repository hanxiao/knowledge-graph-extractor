#!/usr/bin/env python3
"""
Orchestrator for the autoresearch loop.

Usage:
  python run.py baseline                 # establish + persist baseline.json
  python run.py run configs/batchN.json  # run a batch of candidate configs
  python run.py board                    # regenerate leaderboard.md from log

Server reuse: consecutive configs that share identical server_args reuse the
running container (saves the ~60-120s model reload). Configs are therefore
sorted by server_args before running.
"""
import os, sys, json, time
import harness as H

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "experiments.jsonl")
BASELINE = os.path.join(HERE, "baseline.json")

def load_baseline():
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            return json.load(f)
    return None

def append_log(rec):
    rec = dict(rec)
    rec.pop("unique_facts_list", None)  # keep log compact
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

def do_baseline():
    cfg = {"id": "baseline", "desc": "repo defaults (docker-compose.yml)",
           "server_args": H.BASELINE_SERVER_ARGS, "sampling": H.BASELINE_SAMPLING}
    print("[baseline] launching + running 3 rounds...")
    res = H.run_config(cfg, baseline=None)
    H.stop_server()
    # baseline coverage of itself = 1.0 by definition; store facts for guard
    res["metrics"]["coverage_of_baseline"] = 1.0
    with open(BASELINE, "w") as f:
        json.dump(res, f, indent=2)
    rec = dict(res);
    append_log({**{k: v for k, v in res.items() if k != "unique_facts_list"},
                "verdict": "baseline", "ts": int(time.time())})
    print(json.dumps(res["metrics"], indent=2))
    return res

def do_run(configs_path):
    baseline = load_baseline()
    if not baseline:
        print("No baseline.json -- run `python run.py baseline` first."); sys.exit(1)
    with open(configs_path) as f:
        configs = json.load(f)
    # sort so identical server_args are adjacent -> reuse server
    configs.sort(key=lambda c: json.dumps(c["server_args"]))
    prev_args = None
    for i, cfg in enumerate(configs):
        same = (cfg["server_args"] == prev_args)
        print(f"\n=== [{i+1}/{len(configs)}] {cfg['id']} :: {cfg.get('desc','')} "
              f"(reuse_server={same}) ===")
        try:
            if not same:
                H.start_server(cfg["server_args"])
            res = H.run_config(cfg, baseline=baseline, reuse_server=True)
            ok, reasons = H.quality_pass(res, baseline)
            res["metrics"]["quality_pass"] = ok
            m = res["metrics"]; bm = baseline["metrics"]
            base_task = bm.get("task_tps") or bm["decode_tps"]
            verdict = "KEEP" if ok and m["task_tps"] > base_task * 1.01 else \
                      ("QUALITY_OK_NO_SPEEDUP" if ok else "REJECT")
            print(f"  task_tps={m['task_tps']} decode_tps={m['decode_tps']} "
                  f"(base task {base_task})  unique={m['unique_facts']}/{bm['unique_facts']}  "
                  f"cov={m['coverage_of_baseline']}  ground={m['groundedness']}  "
                  f"-> {verdict} {reasons if reasons else ''}")
            append_log({**{k: v for k, v in res.items() if k != "unique_facts_list"},
                        "verdict": verdict, "reasons": reasons, "ts": int(time.time())})
            prev_args = cfg["server_args"]
        except Exception as e:
            print(f"  ERROR: {e}")
            append_log({"id": cfg["id"], "desc": cfg.get("desc", ""),
                        "server_args": cfg["server_args"], "sampling": cfg.get("sampling"),
                        "verdict": "ERROR", "error": str(e)[:500], "ts": int(time.time())})
            prev_args = None  # force reload next
            H.stop_server()
    H.stop_server()

def do_board():
    base = load_baseline()
    rows = []
    if os.path.exists(LOG):
        with open(LOG) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # dedup by id, keep last
    by_id = {}
    for r in rows:
        if "metrics" in r:
            by_id[r["id"]] = r
    def task_of(r):
        return r["metrics"].get("task_tps") or r["metrics"].get("decode_tps", 0)
    items = sorted(by_id.values(), key=lambda r: -task_of(r))
    base_task = (base["metrics"].get("task_tps") or base["metrics"]["decode_tps"]) if base else None
    lines = ["# Leaderboard", "",
             f"Baseline task_tps: **{base_task}** tok/s  |  decode_tps: **{base['metrics']['decode_tps'] if base else '?'}**  |  unique facts: **{base['metrics']['unique_facts'] if base else '?'}**",
             "", "Ranked by task throughput (tokens/sec for the 3-round task). decode_tps = per-stream rate.",
             "", "| rank | id | task tok/s | vs base | decode tok/s | par | unique | cov | ground | verdict | desc |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rank = 0
    for r in items:
        m = r["metrics"]
        rank += 1
        tt = task_of(r)
        delta = f"+{round((tt/base_task-1)*100,1)}%" if base_task else "-"
        lines.append(f"| {rank} | {r['id']} | {round(tt,2)} | {delta} | {m.get('decode_tps')} | "
                     f"{'Y' if m.get('parallel_rounds') else '-'} | {m.get('unique_facts')} | "
                     f"{m.get('coverage_of_baseline')} | {m.get('groundedness')} | "
                     f"{r.get('verdict')} | {r.get('desc','')[:36]} |")
    with open(os.path.join(HERE, "leaderboard.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if cmd == "baseline":
        do_baseline()
    elif cmd == "run":
        do_run(sys.argv[2])
    elif cmd == "board":
        do_board()
    else:
        print(__doc__)

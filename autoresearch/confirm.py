#!/usr/bin/env python3
"""
Rigorous multi-seed confirmation. Single 3-round runs have ~+-1 t/s (~2%) noise
on decode_tps, comparable to the candidate gains -- so a one-shot comparison is
not trustworthy. This runs each config R times with a DIFFERENT seed-triple each
repeat (server reused across repeats), and reports mean +- std so we can tell
signal from noise and confirm the no-quality-loss claim across seeds.

Usage: python confirm.py <id1> <id2> ... [--repeats N]
  ids resolve against candidates.BATCHES (any batch) plus the literal 'baseline'.
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
import candidates as C

def all_configs():
    m = {"baseline": {"id": "baseline", "desc": "repo defaults",
                      "server_args": H.BASELINE_SERVER_ARGS, "sampling": H.BASELINE_SAMPLING}}
    for batch in C.BATCHES.values():
        for c in batch:
            m[c["id"]] = c
    return m

def confirm(cfg, baseline, repeats):
    H.start_server(cfg["server_args"])
    decs, tasks, uniqs, covs, grounds = [], [], [], [], []
    base_seeds = list(H.SEEDS)
    for r in range(repeats):
        H.SEEDS = [s + r * 1000 + 7 for s in base_seeds]  # distinct triple per repeat
        res = H.run_config(cfg, baseline=baseline, reuse_server=True)
        m = res["metrics"]
        decs.append(m["decode_tps"]); tasks.append(m["task_tps"])
        uniqs.append(m["unique_facts"]); covs.append(m["coverage_of_baseline"])
        grounds.append(m["groundedness"])
        print(f"    repeat {r+1}/{repeats}: decode={m['decode_tps']} task={m['task_tps']} "
              f"uniq={m['unique_facts']} cov={m['coverage_of_baseline']}", flush=True)
    H.SEEDS = base_seeds
    def ms(x): return (round(statistics.mean(x), 2),
                       round(statistics.pstdev(x), 2) if len(x) > 1 else 0.0)
    return {"id": cfg["id"], "repeats": repeats,
            "decode_mean": ms(decs)[0], "decode_std": ms(decs)[1], "decode_all": decs,
            "task_mean": ms(tasks)[0], "task_std": ms(tasks)[1],
            "uniq_mean": ms(uniqs)[0], "uniq_all": uniqs,
            "cov_mean": ms(covs)[0], "cov_min": min(c for c in covs if c is not None) if any(covs) else None,
            "ground_mean": ms(grounds)[0]}

if __name__ == "__main__":
    args = sys.argv[1:]
    repeats = 5
    if "--repeats" in args:
        i = args.index("--repeats"); repeats = int(args[i+1]); del args[i:i+2]
    cfgs = all_configs()
    baseline = json.load(open(os.path.join(os.path.dirname(__file__), "baseline.json")))
    out = []
    for cid in args:
        if cid not in cfgs:
            print(f"unknown id {cid}; have {list(cfgs)}"); continue
        print(f"\n=== confirming {cid} ({repeats} repeats) ===", flush=True)
        out.append(confirm(cfgs[cid], baseline, repeats))
    H.stop_server()
    print("\n=== SUMMARY (mean +- std over repeats) ===")
    bd = next((o for o in out if o["id"] == "baseline"), None)
    for o in out:
        delta = f" ({round((o['decode_mean']/bd['decode_mean']-1)*100,1):+}% vs baseline)" if bd and bd["decode_mean"] else ""
        print(f"{o['id']:16s} decode {o['decode_mean']}+-{o['decode_std']} t/s{delta} | "
              f"task {o['task_mean']}+-{o['task_std']} | uniq {o['uniq_mean']} {o['uniq_all']} | "
              f"cov_min {o['cov_min']} | ground {o['ground_mean']}")
    with open(os.path.join(os.path.dirname(__file__), "confirmation.json"), "w") as f:
        json.dump(out, f, indent=2)

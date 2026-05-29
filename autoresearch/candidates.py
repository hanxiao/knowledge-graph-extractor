#!/usr/bin/env python3
"""
Candidate config generator. Emits batch JSON files consumed by run.py.

A config = {id, desc, server_args, sampling, prompt?}.
server_args is the exact CLI token list passed to llama-server.

Central hypothesis for the L4 24GB bottleneck:
  22GB weights + KV cache + compute buffers > 24GB, so -fitt offloads tensors
  to CPU and the offloaded MoE experts make decode CPU-bound. Anything that
  frees VRAM (KV quantization, smaller ctx) should reduce offload and raise
  decode tok/s. MTP draft tuning changes speed without changing the output
  distribution (speculative sampling is distribution-preserving), so it is a
  safe speed-only lever.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

CANON = {
    "--ctx-size": "16384", "--parallel": "1", "--flash-attn": "1", "--no-mmap": "",
    "--threads": "8", "--spec-type": "draft-mtp", "--spec-draft-n-max": "2",
    "--n-predict": "8192", "--jinja": "", "--chat-template-file": "/templates/chat_template.jinja",
    "-fitt": "512", "--cache-reuse": "256",
}

def args(**ov):
    d = dict(CANON); d.update(ov)
    out = ["--model", H.MODEL_PATH, "--host", "0.0.0.0", "--port", "8080"]
    for k, v in d.items():
        if v is None:
            continue
        out.append(k)
        if v != "":
            out.append(str(v))
    return out

SAMP = dict(H.BASELINE_SAMPLING)

def cfg(id, desc, **ov):
    return {"id": id, "desc": desc, "server_args": args(**ov), "sampling": SAMP}

# ----- Batch 1: single-variable probes -----
BATCH1 = [
    cfg("kv_q8",       "KV cache q8_0 k+v (frees VRAM, tiny numeric change)",
        **{"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"}),
    cfg("kv_q8k_f16v", "KV k=q8_0 v=f16 (flash-attn likes f16 v)",
        **{"--cache-type-k": "q8_0"}),
    cfg("kv_q4",       "KV cache q4_0 k+v (max VRAM saving, higher risk)",
        **{"--cache-type-k": "q4_0", "--cache-type-v": "q4_0"}),
    cfg("mtp_n3",      "MTP draft n-max 3", **{"--spec-draft-n-max": "3"}),
    cfg("mtp_n4",      "MTP draft n-max 4", **{"--spec-draft-n-max": "4"}),
    cfg("mtp_n3_min1", "MTP n-max 3 n-min 1", **{"--spec-draft-n-max": "3", "--spec-draft-n-min": "1"}),
    cfg("ubatch1024",  "batch 2048 ubatch 1024", **{"--batch-size": "2048", "--ubatch-size": "1024"}),
    cfg("ctx12k",      "ctx 12288 (smaller KV -> less offload)", **{"--ctx-size": "12288"}),
]

def pcfg(id, desc, parallel_rounds=True, **ov):
    c = cfg(id, desc, **ov)
    c["parallel_rounds"] = parallel_rounds
    return c

# ----- Batch 2: parallelism (the 3 rounds are independent -> run concurrently).
# GPU util is only ~56% on single-stream decode, so concurrent rounds should
# raise aggregate task throughput. --parallel N splits ctx across N slots, so
# ctx-size is raised to keep >=8k tokens/slot (doc ~4.5k prompt + ~2.7k output).
BATCH2 = [
    pcfg("par2",       "parallel 2 slots, ctx16384 (8192/slot)", **{"--parallel": "2"}),
    pcfg("par3",       "parallel 3 slots, ctx24576 (8192/slot)",
         **{"--parallel": "3", "--ctx-size": "24576"}),
    pcfg("par3_q8",    "parallel 3 + KV q8 (VRAM-safe), ctx24576",
         **{"--parallel": "3", "--ctx-size": "24576", "--cache-type-k": "q8_0", "--cache-type-v": "q8_0"}),
    pcfg("par3_mtp3",  "parallel 3 + MTP n-max 3 (stack batch1 win), ctx24576",
         **{"--parallel": "3", "--ctx-size": "24576", "--spec-draft-n-max": "3"}),
]

# ----- Batch 3: MTP/spec-decode acceptance sweep (the only positive lever).
# Build default --spec-draft-n-max is 3 (repo overrode to 2). --spec-draft-p-min
# (default 0.0) gates which positions to draft: higher = draft only confident
# positions -> fewer rollbacks but fewer drafts. Sweep the trade-off. These are
# distribution-preserving (speed-only); coverage guard catches any RNG drift.
BATCH3 = [
    cfg("mtp_n2_ctrl",   "MTP n-max 2 (repo's override, control)", **{"--spec-draft-n-max": "2"}),
    cfg("mtp_n3_rc",     "MTP n-max 3 (build default; reconfirm +2%)", **{"--spec-draft-n-max": "3"}),
    cfg("mtp_n3_pmin01", "n3 + p-min 0.1", **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.1"}),
    cfg("mtp_n3_pmin03", "n3 + p-min 0.3", **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.3"}),
    cfg("mtp_n3_pmin05", "n3 + p-min 0.5", **{"--spec-draft-n-max": "3", "--spec-draft-p-min": "0.5"}),
    cfg("mtp_n4_pmin03", "n4 + p-min 0.3", **{"--spec-draft-n-max": "4", "--spec-draft-p-min": "0.3"}),
    cfg("mtp_n5_pmin05", "n5 + p-min 0.5", **{"--spec-draft-n-max": "5", "--spec-draft-p-min": "0.5"}),
    pcfg("par3_clean",   "parallel 3, ctx36864 (12288/slot, no truncation) - clean close-out",
         **{"--parallel": "3", "--ctx-size": "36864"}),
]

BATCHES = {"batch1": BATCH1, "batch2": BATCH2, "batch3": BATCH3}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "batch1"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{name}.json")
    with open(out, "w") as f:
        json.dump(BATCHES[name], f, indent=2)
    print(f"wrote {out} ({len(BATCHES[name])} configs)")

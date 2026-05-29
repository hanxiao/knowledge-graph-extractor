# program.md — autoresearch loop for KI-extractor decode throughput

Analog of karpathy/autoresearch `program.md`: the instructions that drive the
autonomous research org. The human edits this file; the agent runs the loop.

## Goal

Push the **decode throughput (tokens/sec)** of `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL`
on the KI-extraction task to its limit on a single **NVIDIA L4 24GB**, with
**zero quality loss**.

## Hard constraints (never violate)

1. Same model file. No requantization, no different checkpoint.
2. Same GPU type (single L4 24GB).
3. No quality regression vs baseline (see guard).
4. Task fixed: 3-round KI extraction on the cached article (`doc_cache.md`),
   prompt + JSON schema from the repo `app.py`.

## Metric

- **Primary:** `decode_tps` = `sum(predicted_n)/sum(predicted_ms)*1000` over the
  3 rounds, read from llama-server's own `timings` (ground truth, not the SSE
  chunk count the UI shows).
- **Secondary:** `wall_3rounds_s` (end-to-end), `prefill_tps`.

## Quality guard (a config is only KEPT if all hold vs baseline)

- `schema_valid == True`
- `unique_facts >= baseline.unique_facts` (dedup: jina-v5-nano, triple, 0.90)
- `groundedness >= baseline.groundedness - 0.02` (evidence_span is verbatim)
- `coverage_of_baseline >= 0.90` (baseline facts recalled by candidate set,
  title+desc embedding, cosine >= 0.80) — guards against "fewer tokens, lost info"

Comparisons use **fixed seeds** `[101,202,303]` so configs are apples-to-apples
(the repo uses random seeds, which adds noise).

## Loop

1. `python run.py baseline` -> writes `baseline.json`.
2. `python candidates.py <batch>` -> writes `configs/<batch>.json`.
3. `python run.py run configs/<batch>.json` -> appends to `experiments.jsonl`,
   prints KEEP / QUALITY_OK_NO_SPEEDUP / REJECT per config.
4. `python run.py board` -> regenerates `leaderboard.md`.
5. Read results, record learnings in `strategies.md`, hill-climb: the next batch
   combines the KEPT levers and probes around the new best.

Server reuse: configs with identical `server_args` reuse the running container
(saves model reload). run.py sorts batches to exploit this.

## Lever map (search space)

- VRAM-freeing (reduce CPU offload, the main bottleneck): `--cache-type-k/v`
  (q8_0, q4_0), `--ctx-size`. Numerics-changing -> guard applies.
- Speed-only (distribution-preserving): `--spec-draft-n-max/min`, draft
  acceptance, `--batch-size/--ubatch-size`, `--threads`, `--cache-reuse`.
- Output-shortening (changes tokens -> guard critical): prompt/schema edits.
  Lower priority; only if serving levers are exhausted.

## Stop / convergence

Stop a search branch after 2 consecutive batches yield no quality-passing
speedup over the current best. Record the final best config + its diff vs the
repo defaults in `strategies.md` and `leaderboard.md`.

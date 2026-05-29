# autoresearch — dataroom

Autonomous optimization of `Qwen3.6-35B-A3B` decode throughput on the KI
extraction task, single L4 24GB, zero quality loss. Structure follows
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): a fixed
measurement rig, an iterated config, a metric, and an append-only experiment log.

| file | role | karpathy analog |
|---|---|---|
| `program.md` | loop instructions, goal, constraints, guard | `program.md` |
| `harness.py` | measurement rig (server lifecycle, timings, quality) — not edited | `prepare.py` |
| `candidates.py` | generates config batches (the iterated thing) | `train.py` |
| `run.py` | orchestrator: baseline / run batch / leaderboard | runner |
| `doc_cache.md` | fixed input article (Jina Reader markdown) | dataset |
| `baseline.json` | reference metrics + facts | — |
| `experiments.jsonl` | append-only log of every experiment | experiment log |
| `leaderboard.md` | configs ranked by decode tok/s | results |
| `strategies.md` | GOOD/BAD strategy ledger with reasoning | — |

## Quickstart (on the L4 box, in venv)

```bash
cd ~/ki-extractor/autoresearch
uv run run.py baseline
uv run candidates.py batch1
uv run run.py run configs/batch1.json
uv run run.py board
```

Metric = ground-truth decode tok/s from llama-server `timings`. Quality guard:
unique facts >= baseline, groundedness preserved, baseline facts recalled
(semantic coverage >= 0.90). Fixed seeds [101,202,303] for comparability.

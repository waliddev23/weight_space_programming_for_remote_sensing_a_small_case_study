# Evidence for the Fable 5.1 run

Everything needed to check how Claude Fable 5.1 produced the READ, EDIT and WRITE results, in a single Claude Code session from 2026-11-10 19:21:02 UTC to 2026-11-11 10:08:36 UTC.

## What to read first

1. `fable_reasoning_transcript.md` shows how Fable reasoned, step by step, with the analyses it ran and the outputs it saw.
2. `blindness_audit.md` lists every command that came near a blinded or forbidden file, and when.
3. The `raw_response.txt` file in each run folder is the answer Fable committed to disk before running the evaluator.

## What this record does and does not contain

- **Contains** every message, command and output of the Fable portion of the session (in the raw log).
- **Does not contain** the model's full internal thinking, which Claude Code does not store. The short reasoning notes between steps are what was recorded.
- **Curated, not edited.** The readable transcript leaves out harness bugs, retries and memory notes, each marked in place. `fable_session_raw.jsonl` has every message of the Fable portion. Only system-injected context was removed, and personal identifiers (home path, account names, email) were redacted.
- **Scope.** Only work by Fable 5.1 is here. Later analysis in this repository, such as the repair baselines, the multi-seed training and the figures, was done afterwards with a different model and with the results already known.

## Timeline

- Directive received: 2026-11-10 19:21:02 UTC
- READ predictions written to disk: 2026-11-10 19:26:55 UTC
- Last Fable message: 2026-11-11 10:08:36 UTC

## Files and SHA-256 hashes

Recompute with `sha256sum <file>` (paths relative to `eurosat_wsp/runs/`). The hashes of this folder's own files change if the build script is rerun.

**Fable's own records, written during the run**

| File | SHA-256 |
|---|---|
| `read_run_001/raw_response.txt` | `191ebc04ad0576ba…` |
| `read_run_001/analysis_features.py` | `8badc095b427e482…` |
| `read_run_001/analysis_features.json` | `3e8789d2c0cf1758…` |
| `read_run_001/eval_output.txt` | `11e6edc916ad36f2…` |
| `edit_run_001/raw_response.txt` | `eaf9f63c43c17033…` |
| `edit_run_001/parsed_output/repair_script.py` | `7030e8d879bc3ff3…` |
| `edit_run_001/eval_output.txt` | `98d37a8177d5d8dd…` |
| `write_run_001/raw_response.txt` | `c6fae15d79744e6c…` |
| `write_run_001/make_filters.py` | `aba169e5bec61857…` |
| `write_run_001/conv1_design.json` | `ba6610154489bd53…` |
| `write_run_001/parsed_output/first_conv_fable.pt` | `e8eb7e70377b5c47…` |
| `write_run_001/parse_output.txt` | `e4ee3bef86760fc2…` |
| `write_run_001/eval_zero_shot_output.txt` | `d92fbf24ba2daef7…` |
| `write_run_001/eval_train_output.txt` | `f4cfc3fd6931fabf…` |
| `write_run_001/trained_seed0_original_eval.json` | `b000ef2132d506c9…` |
| `final_wsp_report.md` | `e43d517d679c0d80…` |

**Evaluator outputs written by the benchmark harness**

| File | SHA-256 |
|---|---|
| `../experiment_1_read/metrics/llm_read_9models_result.json` | `b508f1e32b73ad8f…` |
| `../experiment_2_edit/metrics/llm_repair_a_result.json` | `458d560e7ab56e4f…` |
| `../experiment_3_write/baseline/llm_zero_shot_metrics.json` | `53029605f7f1b053…` |

**Session evidence (this folder)**

| File | SHA-256 |
|---|---|
| `evidence/fable_reasoning_transcript.md` | `181cc2c9ea2a8a11…` |
| `evidence/fable_session_raw.jsonl` | `f2c2af7f401742ee…` |
| `evidence/blindness_audit.md` | `63ba1fda6a49b219…` |
| `evidence/build_evidence.py` | `b274f83dc47d0b32…` |

## Rebuild

```bash
python eurosat_wsp/runs/evidence/build_evidence.py \
  --session ~/.claude/projects/<project>/<session-id>.jsonl --redact <names>
```

# lost-in-transcription

Code-switched ASR for the Mozilla Data Collective **"Lost in Transcription"** competition
(DrivenData). Three tracks: Spanish–English (Bangor Miami), Spanish–Nahuatl (W. Sierra Puebla),
Indonesian–Javanese (Jember). Metric: **WER**. Submission is a containerized, offline inference
package. See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for strategy.

## Layout

```
configs/tracks.yaml        per-track config (corpus, langs, model, split policy)
src/lit/
  config.py                config loader
  cli.py                   `python -m lit.cli train ...`
  data/    prepare.py splits.py    corpus -> unified manifest; speaker-disjoint & k-fold splits
  text/    normalize.py nahuatl_ortho.py   scorer-matching normalization (the edge)
  scoring/ wer.py          local WER scorer replicating the competition metric
  models/  whisper_infer.py mms_infer.py   inference wrappers (zero-shot + fine-tuned)
  training/ aft_whisper.py aft_mms.py aft_xlsr.py   fine-tuning
  lm/      kenlm_build.py  bilingual n-gram LMs for CTC decoding
  scoring/ wer.py official.py    local WER via the OFFICIAL runtime/score.py (single source)
scripts/   create_splits.py run_baseline.py make_gold_csv.py score_local.py download_data.sh + slurm/
submission_src/ main.py pack_submission.sh assets/   our submission source (packed to submission.zip)
submission/     build output (submission.zip, gitignored)
runtime/        official DrivenData runtime as a git SUBMODULE (score.py, Dockerfile, deps)
results/   eval logs   logs/ slurm job logs   data/ corpora (gitignored)
```

## Runtime (fixed by the competition)
Python 3.12, uv ≥0.9.24, base `nvidia/cuda:13.0.3`, `transformers<5`, `vllm==0.23.0`, `qwen-asr`.
Inference: 1×A100 80GB, ≤2h, no network. `omnilingual-asr` is NOT in the runtime (local research
only). Init the submodule after clone: `git submodule update --init runtime`.

## Setup

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

On the cluster: `bash scripts/slurm/setup_env.sh`.

## Workflow

```bash
python scripts/create_splits.py --track es_en --out data/manifests/es_en.parquet
python scripts/run_baseline.py  --track es_en --split test --model whisper
sbatch --export=MODEL=whisper scripts/slurm/tillicum_baseline.slurm   # all 3 tracks
```

## Status
Scaffold in place; `prepare_*`, model inference, and training are stubs pending the
login-gated data layout, official scorer, and submission spec (see EXPERIMENT_PLAN.md).

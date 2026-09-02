# Lost in Transcription — Experiment Plan

Living strategy doc. Goal: **win** the Mozilla Data Collective "Lost in Transcription"
code-switched ASR competition (DrivenData). $20k. Three independent tracks + best-average bonus.
Pursuing **all three tracks, balanced**.

## The three tracks

| Track | Corpus | Data | Lang codes | Character |
|---|---|---|---|---|
| `es_en` | Bangor Miami | ~35h, ~240k words, conversational | es, en | Data-rich; intra-sentential code-switching |
| `es_nah` | W. Sierra Puebla Nahuatl | ~3.4h, 2,681 pre-seg utts | es, nhi | Extreme low-resource; orthography (u/k/s) sensitive |
| `id_jv` | Jember Javanese | ~10h, 6,679 utts | id, jv, mad, en | 4-way mixing |

## Rules that shape the design (confirmed from the official runtime)
- **Code-execution submission** (`runtime/` submodule = drivendataorg/lost-in-transcription-runtime):
  `submission.zip` with `main.py` at root → `uv run src/main.py`. Reads
  `/code_execution/data/submission_format.csv` + `clips/`, writes `submission/submission.csv`.
  Source in `submission_src/`, pack with `submission_src/pack_submission.sh` (`uvx rpzip`).
- **Env is fixed**: Python **3.12**, uv **≥0.9.24**, base `nvidia/cuda:13.0.3`, torch on **cu130**,
  `transformers<5`, and notably **`vllm==0.23.0` + `qwen-asr`** are in the runtime.
  **`omnilingual-asr`/`fairseq2` are NOT** → Omni ASR is local-research only unless a dep-add is
  requested/accepted. Whisper, MMS, and **Qwen3-ASR (via vLLM)** run natively.
- **Inference hardware/limits**: 1×A100 80GB, 24 vCPU, 220GB RAM, **≤2h** (≤1 min smoke), **no
  network** (weights baked into the zip). 7B models fit.
- **Metric**: official `runtime/score.py` used AS-IS. `normalize_text` strips `[...]`/`(?)`, unwraps
  `(word)`, removes `¿¡";:,!?` and periods (keeps `...`), and **lowercases only sentence-initial
  letters** (preserves acronyms/mid-word case). Our `lit.scoring` imports this exact function.
- **External data with rights** allowed. **Open-weight models** allowed (public weights, permissive
  license, no hosted API). Competition data must NOT be sent to third-party services.
- Winners open-source under **MPL**, reproducible with/without retraining.

## The #1 edge: local scorer parity
No public test labels. Every submission is precious (daily cap TBD). We replicate the competition's
normalization + WER **exactly** in `src/lit/scoring/` so local WER ≈ leaderboard WER. Until we have
the official scorer, `src/lit/text/normalize.py` encodes our best-guess normalization; revise the
moment we obtain the real one.

## Phase 0 — Scaffold + data (current)
- [ ] Repo skeleton (this scaffold)
- [ ] Obtain login-gated details: exact splits, test segmented vs long-form, sample rates, metadata,
      official scorer, submission Docker/GPU/runtime/daily-cap.
- [ ] `prepare_*` per corpus → unified manifest (audio_path, text, speaker, session, duration).
- [ ] **Speaker/session-disjoint** splits (`src/lit/data/splits.py`); k-fold for tiny `es_nah`.

## Phase 1 — Baselines
Zero-shot floor, all tracks (**runtime-compatible models first**, since only these can be submitted):
- **Qwen3-ASR (via vLLM)** — in the runtime deps; strong multilingual; likely intended baseline.
- Whisper large-v3 (+ turbo for the runtime budget), MMS-1B-all (CTC).
- **Omni ASR** — LOCAL RESEARCH ONLY (fairseq2 not in runtime). Still worth benchmarking on the
  cluster: best zero-shot for jav/nahuatl, and useful for pseudo-labeling / distillation into a
  submittable model. Not directly submittable unless a dep-add request is accepted.
Fine-tuned starting line:
- `es_en`: Whisper large-v3 + LoRA (enough data; likely track winner).
- `es_nah`: MMS/XLS-R + CTC + bilingual KenLM (low-resource specialist).
- `id_jv`: MMS-1B FT or Whisper FT, multilingual char output.
Log everything to `results/` with the local scorer.

## Phase 2 — Improve over baselines (levers)
- **External data with rights**: Common Voice (es, id, jv), extra Nahuatl corpora (Pugh et al.).
- **LM rescoring**: bilingual KenLM + beam search (reuse n-gram ablation harness).
- **Whisper decoding hygiene**: `condition_on_previous_text=False`, temperature fallback,
  `initial_prompt`, token suppression.
- **Nahuatl orthography normalizer** applied identically to targets + predictions.
- **Newer backbones**: w2v-BERT 2.0 / Seamless, MMS-zeroshot vs Whisper/XLS-R.
- **Segmentation/VAD** if test is long-form.

## Phase 3 — Submission hardening
- Fit within runtime budget (ensemble only if it fits). Pin deps. Reproducible container.
- Dry-run the container on held-out local test; confirm CSV schema.

## Compute
UW HPC: Tillicum + Hyak (Slurm). Scripts in `scripts/slurm/`. Eval logs in `results/`, job logs in `logs/`.

## Open questions (resolve from platform)
1. Test audio: pre-segmented utterances or long-form recordings? (drives VAD need + split length dist)
2. Official normalization/scorer script.
3. Submission spec: base image, GPU type, wall-clock limit, daily submission cap, one vs three submissions.
4. Provided splits / dev set?

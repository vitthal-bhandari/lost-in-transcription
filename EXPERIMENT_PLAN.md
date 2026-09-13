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

### id_jv Whisper ablation (current — run on Tillicum H200)
Submittable baseline = `whisper-large-v3`. Trainer: `src/lit/training/aft_whisper.py` (segment-slicing
loader, official-scorer val WER). Cells decided on **dev** WER:
```
# zero-shot ceiling reference
sbatch --export=ALL,TRACK=id_jv,SPLIT=dev,MODEL=openai/whisper-large-v3,RUN=whisper_zeroshot \
       scripts/slurm/tillicum_eval.slurm
# 3 fine-tune cells (own GPU each)
for A in whisper_full whisper_lora whisper_freeze_enc; do
  sbatch --export=ALL,TRACK=id_jv,APPROACH=$A scripts/slurm/tillicum_train.slurm
done
# then eval each checkpoint on dev
sbatch --export=ALL,TRACK=id_jv,SPLIT=dev,MODEL=checkpoints/id_jv/whisper_full,RUN=whisper_full \
       scripts/slurm/tillicum_eval.slurm
```
Defaults (env-overridable): lang=id, lr=1e-5, epochs=8 + early-stop(patience 3) on val WER,
bf16, targets=official-normalized. No n-gram LM (seq2seq). Trains on split=train, early-stops on
split=val; dev is the honest proxy.

### Omni (ceiling/teacher; runtime PR pending)
Isolated env: `bash scripts/slurm/setup_omni_env.sh` (builds `.venv-omni`, dumps the resolved dep
tree to `runtime_pr/omni_resolved_deps.txt` for the runtime PR). API: `ASRInferencePipeline.transcribe(inp, lang=[...])`
— **CTC ignores `lang`; LLM needs one code/clip**. Cards are v2 (`omniASR_LLM_7B_v2`, `omniASR_CTC_7B_v2`);
40s cap (use `_Unlimited_` cards for longer). Zero-shot benchmark on dev (**sweep the LLM lang**):
```
# LLM 7B — try Indonesian vs Javanese conditioning
sbatch --export=ALL,TRACK=id_jv,CARD=omniASR_LLM_7B_v2,LANG=ind_Latn,RUN=omni_llm7b_ind scripts/slurm/tillicum_omni.slurm
sbatch --export=ALL,TRACK=id_jv,CARD=omniASR_LLM_7B_v2,LANG=jav_Latn,RUN=omni_llm7b_jav scripts/slurm/tillicum_omni.slurm
# CTC 7B — lang ignored (single run)
sbatch --export=ALL,TRACK=id_jv,CARD=omniASR_CTC_7B_v2,RUN=omni_ctc7b scripts/slurm/tillicum_omni.slurm
```
Then open the PR adding `omnilingual-asr`/`fairseq2` to the runtime. FT (teacher, later): `CTC_3B`
full + `CTC_7B` LoRA + KenLM. Not submittable until the PR lands.

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

---

## Phase 4 — id_jv parallel experiment grid (2026-09-12)

**Standings**: public leaderboard #1 = 0.2196 (Univ. Hawaii Manoa), we're #43 at 0.2898 (1
submission used, 2 left). Best dev so far: Qwen3-ASR-1.7B zero-shot forced Indonesian = 0.2528
(still beats every fine-tune attempted). Diagnosed cause of 3/3 fine-tune regressions: train
segments (Jember, median ~5-8s) are much shorter than dev clips, plus a register/diacritic
convention gap — see [[lost-in-transcription-approach]] memory.

**Data now available**: Jember (10h, gitignored `data/jember_javanese/`) + Homostoria (11.3h,
5163 segments, `data/homostoria/`, linguist-reviewed Ind-Eng-Jav code-switch, same TSV schema as
Jember — see `prepare_homostoria` in `src/lit/data/prepare.py`). Pooled manifest built:
`data/manifests/id_jv_plus_homostoria.parquet` (train 10,737 / val 1,105 / dev 372).
Hari Minggoean (10h, single-speaker) identified but not yet downloaded.

**Runtime now supports** (pyproject.toml updated 2026-09-11): `faster-whisper`+`ctranslate2`
(CTranslate2 Whisper inference, real beam search/VAD/hallucination guards) and `kenlm`+
`pyctcdecode` (CTC+n-gram-LM decoding — MMS-1B-all pipeline built, see `scripts/mms_zeroshot.py`).
Omni (`fairseq2`) still NOT in the runtime.

### Step 0 — prerequisite, not yet built
**Segment-concatenation**: join consecutive same-recording short segments (Jember + Homostoria)
into ~20-35s windows matching dev's length distribution. Required before any fine-tuning cell
below — without it we'd likely reproduce the same length-collapse regression seen in the Qwen
full-FT/LoRA attempts. Build this next.

### Cells that can run NOW (no prerequisite)
```bash
# MMS-1B-all zero-shot, greedy CTC (no LM) — pure acoustic baseline
sbatch --export=ALL,TRACK=id_jv,RUN=mms_greedy scripts/slurm/tillicum_mms.slurm

# Build a KenLM on Jember+Homostoria text (our own convention), then MMS+KenLM
sbatch --export=ALL,MANIFEST=data/manifests/id_jv_plus_homostoria.parquet,\
OUT=data/lm/id_jv_plus_homostoria,ORDER=4 scripts/slurm/tillicum_build_kenlm.slurm
# (after the LM build completes:)
sbatch --export=ALL,TRACK=id_jv,MANIFEST=data/manifests/id_jv_plus_homostoria.parquet,\
KENLM=data/lm/id_jv_plus_homostoria/lm_4gram.bin,RUN=mms_kenlm scripts/slurm/tillicum_mms.slurm

# faster-whisper conversion of the existing whisper_full checkpoint + beam search decode
# (not yet built — do after MMS cells if promising)
```

### Cells gated on Step 0 (segment-concatenation)
| Run name | Model | Data | Purpose |
|---|---|---|---|
| `qwen_lora_concat_jember` | Qwen3-ASR LoRA | Jember only, length-fixed | isolate: does fixing length alone beat 0.2528? |
| `qwen_lora_concat_plus_homo` | Qwen3-ASR LoRA | Jember + Homostoria, length-fixed | isolate: does more matched data help on top? |
| `whisper_full_concat` | Whisper full-FT | Jember + Homostoria, length-fixed | parity check against the currently-submitted model |

Each writes to a distinct `results/id_jv/<run_name>/` and `checkpoints/id_jv/<run_name>/` — safe
to `sbatch` all of them at once once Step 0 exists; they run as independent H200 jobs.

### Discipline
- All of the above stay on **dev only**. A submission is spent only once a cell clearly beats
  0.2528 dev by a real margin (submission #1's dev→public gap was +0.031).
- Compare everything via `python3 scripts/leaderboard.py --track id_jv` once jobs land.

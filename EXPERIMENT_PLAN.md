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

## Rules that shape the design
- **Code-execution submission**: containerized, offline (no proprietary API, likely no internet),
  GPU + runtime limits. Reads test MP3/WAV → CSV `audio_filename,transcript`. See `submission/`.
- **Metric**: `WER = (S+D+I)/N`. Scorer strips meta-linguistic tags, unintelligible markers,
  punctuation; whole-word match; Nahuatl orthography normalization.
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
Zero-shot floor, all tracks:
- Whisper large-v3 (+ turbo, for runtime budget), MMS-1B-all (CTC).
- **Omni ASR (Meta Omnilingual ASR)** — Apache-2.0, open weights, offline (fairseq2). Confirmed
  native support for spa/eng/ind/**jav** + many Nahuatl varieties (nhi/nhw/azz/…) that Whisper &
  MMS lack → rules-compliant and likely the strongest zero-shot for jav & nahuatl. Caveats: fairseq2
  (not transformers, install offline in the container); **40s audio limit** on standard variants
  (segment or use *_Unlimited_*); 7B ~17 GiB VRAM likely too heavy for the submission budget → prefer
  CTC / 300M / 1B. Confirm the corpus's exact Nahuatl ISO variety for the `lang` code.
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

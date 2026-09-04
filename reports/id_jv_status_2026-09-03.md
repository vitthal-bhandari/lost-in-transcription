# id_jv (Indonesian-Javanese) — status report, 2026-09-03

## Task
Code-switched ASR, Mozilla "Lost in Transcription" competition. Metric: WER via the official
scorer (`runtime/score.py`) — normalizes text, does NOT fold vowel diacritics. Submission =
containerized `main.py`, offline, no fairseq2 (Omni not submittable as-is).

## Data
- **Train**: Jember corpus, 200 long-form recordings, 6,678 hand-segmented utterances → split
  5,969 train / 710 val (recording-disjoint), ~10h total. Segments are **short** (median 5s, max
  29s). Diacritic-heavy (69% of utterances carry pepet/taling marks: ê, è).
- **Dev**: official `indonesian_dev`, 372 pre-segmented clips, different collection (48kHz vs
  train's 44.1kHz). Longer clips (up to ~40s+). Diacritics rare (8.9% of utterances). This is the
  eval proxy — closest available match to hidden test.
- **Test**: hidden, 2,118 clips, `submission_format.csv` only.

## Results (dev WER, official scorer; lower is better)

| Run | Approach | Dev WER | Verified |
|---|---|---|---|
| **qwen17b_ind** | Qwen3-ASR-1.7B, zero-shot, forced `language=Indonesian` | **0.2528** | ✅ local metrics.json |
| omni_llm7b_ind | Omni `omniASR_LLM_7B_v2`, zero-shot, `lang=ind_Latn` | 0.2524 | ✅ local metrics.json |
| whisper_full | Whisper-large-v3, full fine-tune, 8 epochs, lr 1e-5, + diacritic fold | 0.2592 | ✅ local metrics.json |
| whisper_lora | Whisper-large-v3, LoRA fine-tune, + diacritic fold | 0.2634 | ✅ local metrics.json |
| qwen17b_auto | Qwen3-ASR-1.7B, zero-shot, `language=auto`-detect | 0.2674 | ✅ local metrics.json |
| whisper_freeze_enc | Whisper-large-v3, encoder-frozen fine-tune, + diacritic fold | 0.2900 | ✅ local metrics.json |
| whisper_zeroshot | Whisper-large-v3, zero-shot, no fine-tune | 0.2921 | ✅ local metrics.json |
| qwen_lora | Qwen3-ASR-1.7B, LoRA fine-tune (r=16, lr 3e-5, 3 epochs) | 0.2930 | job log only |
| omni_ctc7b | Omni `omniASR_CTC_7B_v2`, zero-shot, lang ignored (CTC) | 0.3226 | ✅ local metrics.json |
| qwen_ft | Qwen3-ASR-1.7B, **full** fine-tune (lr 2e-5, 3 epochs) | 0.3913 | job log only |
| omni_ctc300m_test | Omni `omniASR_CTC_300M_v2` (plumbing/env validation only) | 0.4888 | ✅ local metrics.json |

**Public leaderboard**: one scored submission so far — `whisper_full` + diacritic fold →
**0.2898 public WER, rank #20/65**. Dev→public gap on that run: +0.031 (dev underestimates public).
Field is dense (#2 = 0.2555 … #20 = 0.2898); #1 = 0.2290, a clear outlier.

## What's been tried and the verdict

- **Zero-shot base comparison**: Qwen3-ASR-1.7B ≈ Omni-7B-LLM (both ~0.252) > Whisper-large-v3
  (0.292). Qwen is submittable (in the competition runtime); Omni is not (fairseq2 conflicts with
  the runtime's torch/vllm pins — confirmed via a dependency probe, PR path abandoned).
- **Fine-tuning has made every model WORSE than its own zero-shot**, at every setting tried
  (Whisper full/LoRA/frozen-encoder; Qwen full FT and LoRA at lr 1e-4 and 3e-5). Whisper's
  regression was partly a diacritic-convention artifact (fixed via `fold_diacritics`, dev
  0.31→0.26); Qwen's regression is NOT a normalization artifact — inspected predictions are
  fluent but semantically wrong, and get worse the longer the dev clip is.
- **Diagnosed root cause**: train (Jember) and dev/test (indonesian_dev) are meaningfully
  different distributions — segment length (train ~5s median vs dev up to 40s+), diacritic
  convention (69% vs 9%), and register (train skews more Javanese-heavy). Fine-tuning pulls the
  model toward Jember and away from dev/test. Symptom on Qwen: on long dev clips the fine-tuned
  model truncates output early (e.g. a multi-clause reference → a 4-word hypothesis), consistent
  with having only ever seen short training targets.
- **Omni distillation (teacher→student) considered and dropped**: zero-shot Omni-7B ≈ zero-shot
  Qwen on dev, so there is no meaningful quality gap for a teacher to transfer; pseudo-labeling
  would only inject Omni's own errors/convention at Qwen's own error rate.
- **Current best submittable model**: Qwen3-ASR-1.7B, zero-shot, forced Indonesian, no
  fine-tuning (0.2528 dev) — beats every fine-tuned attempt.
- **Submission budget**: 3 scored submissions per rolling 7 days; 1 used, 2 remaining. Smoke-test
  submissions are separate/unlimited.

## Untried / open directions
- Segment-concatenation of Jember training data (join consecutive same-recording utterances into
  ~25–35s windows) to directly address the length mismatch before re-attempting fine-tuning.
- Monolingual external data (Common Voice Indonesian, Javanese TTS corpora found on Mozilla Data
  Collective) — assessed as low-confidence: likely reinforces the short-segment bias (those
  corpora are short, scripted/synthetic clips) and doesn't teach code-switching; possible minor
  value only as an anti-forgetting regularizer mixed alongside in-domain data.
- Any unlabeled, spontaneous, code-switched Indonesian-Javanese audio (if sourceable with rights)
  as genuine distillation fuel for a teacher model — not yet identified.

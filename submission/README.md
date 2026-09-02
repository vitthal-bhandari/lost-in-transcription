# submission/

Build output directory. `submission_src/pack_submission.sh` writes `submission.zip` here.
The zip and any packed weights are gitignored. Source lives in `../submission_src/`.

## Build & test flow (uses the official runtime, `../runtime/` submodule)

```bash
# 1. Pack our source (+ baked weights under submission_src/assets/) into submission/submission.zip
bash submission_src/pack_submission.sh

# 2. Smoke/full test in the official container (needs Docker + the runtime image)
cd runtime
just test-submission      # see runtime/justfile / README for the exact targets
```

The container runs `uv run src/main.py` with Python 3.12, no network, on 1×A100 80GB, ≤2h
(≤1 min for smoke). Weights must be baked into the zip — nothing downloads at runtime.

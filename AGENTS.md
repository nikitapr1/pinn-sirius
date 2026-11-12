# Working with this repository

- GitHub is the source of truth; commit and push requested code changes before submitting a Kaggle run.
- Every pushed commit is automatically checked in a private offline Kaggle CPU session by .github/workflows/kaggle.yml, as requested by the user. GPU training remains an explicit experiment; never enable GPU in automatic checks by default.
- Minimize Kaggle compute: use CPU for smoke tests; reserve GPU for actual training. Never leave idle GPU sessions running.
- Use `scripts/kaggle_run.py`, which downloads a specific GitHub commit outside Kaggle and stages it in an offline, private kernel. Do not make GPU sessions clone repositories or download large weights when they can be attached as Kaggle inputs.
- Prefer Kaggle's built-in packages. Prepare missing wheels, model weights and data as Kaggle inputs when practical. Do not reinstall torch/CUDA by default.
- Training code should support short smoke runs, write metrics and regular resumable checkpoints under `OUTPUT_DIR`, and read attached files under `INPUT_DIR`. Resume must restore optimizer/scheduler/step and any required scaler/RNG state.
- Never commit secrets, data, or model weights. This repository is public. Keep Kaggle outputs private unless the user requests publication.
- Full pipeline documentation and commands are in README.md. Loading future user training code still requires checking its dependencies, input paths, resume support, and accelerator compatibility; the bundled smoke test does not prove those.

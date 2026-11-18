# Reproducing the pipeline

The released inference package is distinct from a training checkpoint. Training
checkpoints contain optimizer/configuration state; the Hub package contains
only `model.safetensors` and inference `config.json`.

## Environment

Use Python 3.10+, a CUDA-compatible PyTorch install, the project requirements,
and a Chinese font. The shell launchers require Bash (Linux is recommended for
training). Data preparation is multiprocessing-heavy; start with a small
`--limit` when checking your own dataset. The original training dataset is not
distributed, so this repository alone cannot reproduce the reported scores.

The VAE must be obtained separately under its upstream terms. Pass its local
directory using `--vae` for inference/cache creation and `--vae-path` for training.
That directory must contain the VAE configuration and weights. A full FLUX
pipeline directory is not the same as its `vae` subdirectory.

## Prepare your own data

The existing preprocessing scripts expect the paired directory/naming layout
implemented in `scripts/pair_utils.py`. Adapt that input boundary to your own
authorized dataset before running:

```text
paired-pages/
  2JA/1.webp   # Japanese source
  1CH/1.webp   # matching Chinese target
```

Names must match between the two folders and use numeric stems; the current
loader sorts them numerically. This is an input convention, not an included dataset.

```bash
python scripts/preprocess_pairs.py --root /path/to/paired-pages --out data/proc
python scripts/ocr_pages.py --proc data/proc --out data/ocr
PYTHONPATH=. python scripts/cache_latents.py --proc data/proc --out data/latents --vae /path/to/vae
```

`preprocess_pairs.py` registers page pairs, builds change masks, groups works,
and assigns splits. `ocr_pages.py` produces OCR JSON for Chinese target pages.
Check their reports and rejected samples before training. Keep train/validation
works separate; do not interpret OCR read-back as a human quality score.

## Training stages

1. `build_v2.py` forms composite targets and glyph conditions from real pairs.
2. `cache_latents.py --manifest manifest_v2.jsonl --reuse-src data/latents`
   reuses source latents and encodes the new targets.
3. `train.py` trains v2 on real pairs using the v2 manifest.
4. `build_synth.py` generates random text/font variants from training pages only.
5. Synthetic targets are cached and their manifest is combined with v2.
6. `train.py --init-from runs/v2/latest.pt` starts v3 from v2 EMA weights with
   a fresh optimizer; real and synthetic rows are sampled with equal probability.

The `run_v2_pipeline.sh` and `run_v3_pipeline.sh` launchers automate these stages.
Read their environment settings first and point `MTDIFF_VAE` at your local VAE.
They assume preprocessing, OCR and the initial source-latent cache already exist.
They do not download data or provision compute.

```bash
export MTDIFF_VAE=/path/to/local/vae
bash scripts/run_v2_pipeline.sh 60000
bash scripts/run_v3_pipeline.sh 40000 4
```

Optional path overrides are `MTDIFF_DATA_ROOT`, `MTDIFF_PROC`, `MTDIFF_OCR`,
`MTDIFF_REUSE_CACHE_DIR`, `MTDIFF_CACHE_DIR`, and `MTDIFF_RUNS_DIR`.
Evaluation output can be set with `MTDIFF_EVAL_DIR`.

## Evaluate and export

```bash
bash scripts/run_eval.sh runs/v3/latest.pt v3 pred
python scripts/export_weights.py --checkpoint runs/v3/latest.pt --out weights/mangatrans-v3
```

Evaluation reads the manifest, cache and VAE paths recorded in the training
checkpoint. Run it where those paths are available. `pred`, `source` and `target`
select generated pages, source reconstructions and target reconstructions.
`scripts/ocr_eval.py` scores generated line crops against reference text using
RapidOCR recognition. `scripts/text_metrics.py` defines normalization and scoring.

The existing 72.1% / 46.0% results are historical project-reported numbers. Raw
evaluation artifacts are not included in this release; release-time unit tests
and model-loading checks do not independently reproduce those scores.

# Release verification · 2026-11-18

This record distinguishes packaging checks from research evaluation.

## Verified locally

- 84 CPU regression tests pass on Python 3.10 / PyTorch 2.10.0.
- Python source compiles; the three Bash launchers pass `bash -n`.
- The released safetensors load strictly into the configured DiT.
- Actual parameter count: **119,291,712**; export metadata: **step 40,000**.
- A small CPU forward pass returns the expected `[1, 16, 8, 8]` shape with finite values.
- Code publication uses an explicit Git file list excluding datasets, manga images,
  latent caches, checkpoints, credentials and local agent state.
- A credential-pattern scan of that publication list found no matches.

Model file SHA-256:

```text
dc62cbd1fd7d5700e1df430022089747658044772c66fc901791c60121b6aa84
```

## Not rerun

Full training, full-page GPU inference through the VAE, multi-GPU execution,
and OCR evaluation on the original data were not rerun for this release.
The README's 72.1% / 46.0% results remain historical project-reported metrics.

The local environment emits a pre-existing `pynvml` deprecation warning.
Formatting across the historical research code is not normalized; this release
avoids a large unrelated reformat.

---
title: MangaTrans Diffusion
emoji: 🖋️
colorFrom: red
colorTo: gray
sdk: static
app_file: index.html
fullWidth: true
license: mit
short_description: A diffusion transformer for Chinese manga lettering
tags:
  - diffusion
  - manga
  - image-editing
  - research
---

# MangaTrans-Diffusion · Project showcase

A bilingual visual explanation of a 119.3M-parameter diffusion transformer,
with the repository's all-ages before/after sample and implementation diagram.

[Source code](https://github.com/jomify/MangaTrans-Diffusion) ·
[Model weights](https://huggingface.co/Jomify/MangaTrans-Diffusion-v3)

The page shows the Japanese `おはよう` source beside the v3 `早上好` render,
links to the full-resolution files and `example/regions.json`, and then explains
the inference architecture. The paired images are a repository sample and the
Chinese string is supplied by the caller. Run the Python CLI on a CUDA machine
for additional inference. No GPU service or paid hosting is required for this page.

The source code and released DiT weights use MIT. Third-party VAE and font terms
remain separate. See the model card for provenance, evaluation limitations,
and details of the FLUX.1-dev VAE dependency.

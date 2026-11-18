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

A bilingual interactive explanation of a 119.3M-parameter diffusion transformer
that renders supplied Chinese text inside supplied manga text regions.

[Source code](https://github.com/jomify/MangaTrans-Diffusion) ·
[Model weights](https://huggingface.co/Jomify/MangaTrans-Diffusion-v3)

This static Space explains the method. It does **not** run inference, upload
images, perform OCR, or translate text. ImageGen supplied the cover illustration and lettering. The interactive diagram illustrates local compositing. Run the Python CLI on a CUDA machine for
actual inference. No GPU service or paid hosting is required for this page.

The source code and released DiT weights use MIT. Third-party VAE and font terms
remain separate. See the model card for provenance, evaluation limitations,
and details of the FLUX.1-dev VAE dependency.

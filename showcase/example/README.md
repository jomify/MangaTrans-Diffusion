# Reproduce the lettering example

The Japanese panel shows a gardener greeting a cat with **おはよう**. ImageGen created the source artwork and Japanese lettering. The distributed [input PNG](source.png) is a 768 × 512 resize of the generated panel. The [region file](regions.json) supplies the Chinese translation **早上好** and its vertical layout box.

The [output PNG](output.png) was written directly by the repository CLI. The [comparison image](comparison.webp) places the two complete panels side by side for display. Use the PNGs for pixel comparisons.

From the repository root, after downloading the [v3 inference weights](https://github.com/jomify/MangaTrans-Diffusion/releases/tag/v0.1.0) and obtaining a compatible VAE, run:

```bash
python translate_page.py showcase/example/source.png showcase/example/regions.json reproduced.png --weights weights/mangatrans-v3 --vae /path/to/vae --width 0 --steps 30 --seed 0 --margin 12
```

This run used an RTX 4060 Laptop GPU and Microsoft YaHei (msyh.ttc, face 0). The working width remained 768 px throughout. A different font, hardware, or library version may change the resulting pixels.

| Component | Artifact used for this render |
| --- | --- |
| DiT | v3 EMA, step 40,000; SHA-256 `dc62cbd1fd7d5700e1df430022089747658044772c66fc901791c60121b6aa84` |
| VAE | [Flex.1-alpha VAE, fixed revision](https://huggingface.co/ostris/Flex.1-alpha/tree/5dd0b1bc2a9421b891abcf7c9218993f696dd550/vae); weight SHA-256 `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`, matching the configured FLUX.1-dev VAE |
| Input PNG | SHA-256 `94a19c7c5cfde849595f89abcfa51b3e0ca71a91a1932ef815f26bd2552ffde3` |
| Output PNG | SHA-256 `8e245cd97fb49e5ada0945c62ace356e59c04134f39981291f56f7c2ce4e3c29` |

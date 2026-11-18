# Contributing

Small, focused contributions are welcome: glyph layout, text-region handling,
tiling, evaluation, portability, and documentation.

1. Use Python 3.10 or later and install `requirements.txt` in a virtual environment.
2. Run `python -m pytest tests -q` before and after your changes. Unit tests use
   small CPU models; they do not need the released weights or training data.
3. Add a regression test for a bug fix. Keep changes to model math separate from
   interface and formatting changes.
4. Describe what changed, the reason, and exactly what you tested in your pull request.

Do not include credentials, private paths, datasets, training logs, checkpoints,
or manga pages in issues or pull requests. Use an original synthetic example to
reproduce a visual problem. The project does not redistribute its training data.

The showcase is a self-contained static page in `showcase/index.html`. Preview it
with `python -m http.server 8765 --bind 127.0.0.1 --directory showcase` and check desktop,
mobile, keyboard navigation, and both languages. Credit source artwork and link actual renders to the input, region file,
and inference settings that produced them.

Commit messages should explain the motivation. Optional Git trailers such as
`Tested:`, `Not-tested:`, and `Constraint:` preserve useful context.

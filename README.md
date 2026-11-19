# Jianwen Zhu — public research website

- [Active Research](https://jianwen-zhu.github.io/uoft-public/research-page/)
- [Selected Work](https://jianwen-zhu.github.io/uoft-public/past-research/)

## Update and preview

Edit `research-page/content/` and `research-page/build.py` for active research.
Edit `past-research/build.py` for selected publications; the builder generates both HTML and publication metadata.

```sh
python3 research-page/build.py
python3 past-research/build.py
python3 -m http.server 8766 --bind 127.0.0.1
```

The YTL/TTL reference HTML pages are local snapshots of the language manuals. Citation counts are dated snapshots, not live counts.

GitHub Pages serves the repository root from the `main` branch. The root opens Active Research. Both pages retain their reviewed styles and content.

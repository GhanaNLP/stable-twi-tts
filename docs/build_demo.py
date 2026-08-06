"""Build the GitHub Pages sample page from samples/index.json.

GitHub strips <audio> from README markdown, so inline players need a real page. Generated rather
than hand-written so the page cannot drift from the audio files actually in the repo.
"""
from __future__ import annotations

import json
from pathlib import Path

HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Stable Twi TTS — samples</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {{ color-scheme: light dark; --fg:#111; --dim:#666; --line:#e3e3e3; --bg:#fff;
           --accent:#0a7d5a; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e8e8e8; --dim:#9a9a9a; --line:#2c2c2c; --bg:#141414; --accent:#4fd1a5; }}
  }}
  body {{ max-width: 46rem; margin: 0 auto; padding: 2rem 1.1rem 5rem;
          font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
          color: var(--fg); background: var(--bg); }}
  h1 {{ font-size: 1.6rem; margin: 0 0 .3rem; }}
  h2 {{ font-size: 1.15rem; margin: 2.4rem 0 .4rem; }}
  .sub {{ color: var(--dim); margin: 0 0 1.6rem; }}
  .quote {{ border-left: 3px solid var(--accent); padding: .5rem 0 .5rem .9rem;
            margin: 1rem 0 1.2rem; font-size: 1.02rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; }}
  th, td {{ text-align: left; padding: .45rem .5rem; border-bottom: 1px solid var(--line);
            vertical-align: middle; }}
  th {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; color: var(--dim);
        font-weight: 600; }}
  td.v {{ font-weight: 600; white-space: nowrap; }}
  td.m {{ color: var(--dim); font-variant-numeric: tabular-nums; font-size: .88rem;
          white-space: nowrap; }}
  audio {{ width: 100%; min-width: 15rem; height: 34px; }}
  .note {{ color: var(--dim); font-size: .9rem; }}
  code {{ font-size: .88em; background: rgba(128,128,128,.14); padding: .1em .35em;
          border-radius: 3px; }}
  .wrap {{ overflow-x: auto; }}
  a {{ color: var(--accent); }}
</style>
<h1>Stable Twi TTS — samples</h1>
<p class="sub">Twi and Ghanaian English speech synthesis, driven by IPA phonemes.
<a href="https://github.com/GhanaNLP/stable-twi-tts">Code</a> ·
<a href="https://huggingface.co/ghanaopendata/stable-twi-tts">Model</a></p>
"""

FOOT = """
<h2>How to read these</h2>
<p class="note">
Voices are <strong>pseudo-speakers</strong> — derived by clustering x-vectors over unlabelled
broadcast audio, because neither source corpus had speaker labels. One real person may appear as
two voices, and no voice is a consented identity.
</p>
<p class="note">
They are ranked by <strong>measured intelligibility, not training hours</strong>, which turned out
to predict almost nothing: <code>twi-1</code> is the best code-switch voice yet 21st of 30 on pure
Twi, and two of the three best Twi voices have under 3.3&nbsp;h of audio each. Use
<code>tiers.codeswitch</code> for text mixing English into Twi and <code>tiers.twi_only</code> for
pure Twi — the two rankings disagree sharply.
</p>
<p class="note">
English is <strong>audibly weaker than Twi</strong> and band-limited to 8&nbsp;kHz, because the
English training audio was 16&nbsp;kHz where the Twi was 24&nbsp;kHz. Round-trip phoneme error is
33.5% for Twi against a 25.9% floor, and 59.5% for English against 32.2%.
</p>
"""


def player(rel: str) -> str:
    return f'<audio controls preload="none" src="{rel}"></audio>'


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data = json.loads((root / "samples" / "index.json").read_text(encoding="utf-8"))
    voices = json.loads((root / "samples" / "voices.json").read_text(encoding="utf-8")) \
        if (root / "samples" / "voices.json").exists() else None

    meta = {}
    if voices:
        for v in voices["voices"]:
            meta[v["name"]] = v

    out = [HEAD]

    # ---- same text, every voice ----
    labels = {"twi": "Every voice, same Twi sentence",
              "cs": "Every voice, same code-switched sentence"}
    for tag in ("twi", "cs"):
        rows = [r for r in data["compare"] if r["tag"] == tag]
        if not rows:
            continue
        out.append(f"<h2>{labels[tag]}</h2>")
        out.append(f'<p class="quote">{data["compare_texts"][tag]}</p>')
        out.append('<div class="wrap"><table><tr><th>Voice</th><th>Sample</th>'
                   '<th>Twi&nbsp;/&nbsp;code-switch error</th></tr>')
        rows.sort(key=lambda r: int(r["voice"].split("-")[1]))
        for r in rows:
            m = meta.get(r["voice"], {})
            cs = m.get("codeswitch_uer_avg")
            tw = m.get("twi_only_uer")
            score = (f"{tw:.0%} / {cs:.0%}" if tw is not None and cs is not None else "—")
            out.append(f'<tr><td class="v">{r["voice"]}</td>'
                       f'<td>{player("../samples/" + r["file"])}</td>'
                       f'<td class="m">{score}</td></tr>')
        out.append("</table></div>")

    # ---- diverse text ----
    out.append("<h2>Range of text</h2>")
    out.append('<p class="note">Different sentence types, using the best voice for each mode.</p>')
    out.append('<div class="wrap"><table><tr><th>Kind</th><th>Text</th><th>Sample</th></tr>')
    for r in data["showcase"]:
        text = data["showcase_texts"][r["name"]]
        out.append(f'<tr><td class="v">{r["name"]}</td><td>{text}</td>'
                   f'<td>{player("../samples/" + r["file"])}</td></tr>')
    out.append("</table></div>")

    out.append(FOOT)
    dest = root / "docs" / "index.html"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(data['compare'])} comparison players, {len(data['showcase'])} showcase players")


if __name__ == "__main__":
    main()

"""Build standalone HTML versions of the main Markdown documents into docs/html/.

Why: some Markdown previewers fail to show images or render tables differently. Each HTML file is
self-contained: images are embedded as data URIs, so a page renders in any browser, even when the
file is moved or sent on its own. The Markdown files stay the source; rerun this after editing them.

* Markdown: markdown-it-py (CommonMark + GFM tables, strikethrough, task lists), raw HTML allowed
  (docs/figure_comparison.md uses <table>/<img> for side-by-side panels).
* Heading ids use GitHub-style slugs, so "#section" links written for GitHub keep working.
* Links: to another converted document -> its .html (anchor kept); to any other repo file ->
  a relative path from docs/html/; external links unchanged. Image sources -> embedded.

Run: pixi run docs-html
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import unquote

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "html"

# (source, short label for the navigation bar), in reading order
DOCS = [
    ("README.md", "README"),
    ("docs/report.md", "Report"),
    ("WORKING_WITH_AI.md", "Working with AI"),
    ("docs/figure_comparison.md", "Figure comparison"),
    ("REPLICATION.md", "Results & register"),
    ("docs/ext_sbp_notes.md", "SBP notes"),
    ("docs/paper_check.md", "Paper check"),
    ("docs/methods_notes.md", "Methods notes"),
    ("docs/briefings.md", "Briefings"),
]
OUT_NAME = {(ROOT / src).resolve(): Path(src).stem + ".html" for src, _ in DOCS}

CSS = """
:root {
  --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0; --soft: #f6f8fa;
  --link: #0969da; --accent: #0969da; --code-bg: #eff1f3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d; --soft: #151b23;
    --link: #4493f8; --accent: #4493f8; --code-bg: #262c36;
  }
}
:root[data-theme="dark"] {
  --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d; --soft: #151b23;
  --link: #4493f8; --accent: #4493f8; --code-bg: #262c36;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
}
nav {
  position: sticky; top: 0; z-index: 1; background: var(--soft); border-bottom: 1px solid var(--border);
  padding: 8px 16px; display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 14px;
}
nav a { color: var(--muted); text-decoration: none; }
nav a:hover { color: var(--link); }
nav a.current { color: var(--fg); font-weight: 600; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
.source-note { color: var(--muted); font-size: 13px; border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 16px; }
h1, h2, h3, h4 { line-height: 1.25; margin: 1.6em 0 0.6em; }
h1 { font-size: 2em; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; margin-top: 0.4em; }
h2 { font-size: 1.5em; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
h3 { font-size: 1.25em; }
a { color: var(--link); }
p, ul, ol, table, pre, blockquote { margin: 0 0 1em; }
ul, ol { padding-left: 2em; }
li + li { margin-top: 0.25em; }
li.task-list-item { list-style: none; }
li.task-list-item input { margin: 0 0.4em 0 -1.4em; vertical-align: middle; }
code { background: var(--code-bg); border-radius: 4px; padding: 0.15em 0.35em; font-size: 0.88em;
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }
pre { background: var(--code-bg); border-radius: 6px; padding: 12px 16px; overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote { color: var(--muted); border-left: 4px solid var(--border); margin-left: 0; padding: 0 1em; }
hr { border: 0; border-top: 1px solid var(--border); margin: 2em 0; }
img { max-width: 100%; height: auto; background: #fff; }
.table-wrap { overflow-x: auto; margin: 0 0 1em; }
table { border-collapse: collapse; font-size: 14px; }
th, td { border: 1px solid var(--border); padding: 6px 10px; vertical-align: top; text-align: left; }
th { background: var(--soft); }
tr:nth-child(2n) td { background: color-mix(in srgb, var(--soft) 60%, transparent); }
/* side-by-side figure tables (raw HTML <table> with <img>) span the page in two equal columns */
table:has(img) { width: 100%; table-layout: fixed; }
table:has(img) td { background: transparent; }
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<nav>{nav}</nav>
<main>
<p class="source-note">HTML copy of <code>{src}</code>, built {date} by <code>pixi run docs-html</code>.
The Markdown file is the source: edit it, then rebuild.</p>
{body}
</main>
</body>
</html>
"""


def make_md() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable(["table", "strikethrough"])
    md.use(tasklists_plugin)
    # GitHub-style ids on every heading level so "#anchor" links written for GitHub resolve
    md.use(anchors_plugin, min_level=1, max_level=6, permalink=False)
    return md


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def rewrite_urls(body: str, src_dir: Path) -> tuple[str, list[str]]:
    """Embed local images and retarget relative links; returns the new HTML and any missing targets."""
    missing: list[str] = []

    def fix(m: re.Match) -> str:
        attr, url = m.group(1), html.unescape(m.group(2))
        if re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|//)", url, re.I):  # external, data:, mailto:, same-page anchor
            return m.group(0)
        path_part, _, anchor = url.partition("#")
        target = (src_dir / unquote(path_part)).resolve()
        if not target.exists():
            missing.append(url)
            return m.group(0)
        if attr == "src" and target.is_file():
            new = data_uri(target)
        elif target in OUT_NAME:
            new = OUT_NAME[target] + (f"#{anchor}" if anchor else "")
        else:
            new = Path(os.path.relpath(target, OUT)).as_posix() + (f"#{anchor}" if anchor else "")
        return f'{attr}="{html.escape(new, quote=True)}"'

    return re.sub(r'\b(src|href)="([^"]*)"', fix, body), missing


def check_anchors(pages: dict[str, str]) -> list[str]:
    """Every '#anchor' link (same page or another built page) must match an id."""
    ids = {name: set(re.findall(r'\bid="([^"]+)"', page)) for name, page in pages.items()}
    bad = []
    for name, page in pages.items():
        for href in re.findall(r'href="([^"]*#[^"]*)"', page):
            if href.startswith(("http", "data:")):
                continue
            file, _, anchor = href.partition("#")
            file = file or name
            if file in ids and html.unescape(anchor) not in ids[file]:
                bad.append(f"{name}: {href}")
    return bad


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = make_md()
    date = dt.date.today().isoformat()
    pages: dict[str, str] = {}
    for src, label in DOCS:
        path = ROOT / src
        text = path.read_text(encoding="utf-8")
        body = md.render(text)
        body, missing = rewrite_urls(body, path.parent)
        for u in missing:
            print(f"  WARNING {src}: link target not found: {u}")
        body = body.replace("<table", '<div class="table-wrap"><table').replace("</table>", "</table></div>")
        h1 = re.search(r"^#\s+(.+)$", text, re.M)
        title = re.sub(r"[*_`]", "", h1.group(1)).strip() if h1 else label
        out_name = OUT_NAME[path.resolve()]
        nav = " ".join(
            f'<a href="{OUT_NAME[(ROOT / s).resolve()]}"{" class=current" if s == src else ""}>{html.escape(lab)}</a>'
            for s, lab in DOCS)
        pages[out_name] = PAGE.format(title=html.escape(title), css=CSS, nav=nav, src=src, date=date, body=body)
    for name, page in pages.items():
        (OUT / name).write_text(page, encoding="utf-8")
        print(f"wrote docs/html/{name} ({len(page.encode()) / 1e6:.2f} MB)")
    bad = check_anchors(pages)
    for b in bad:
        print(f"  WARNING broken anchor: {b}")
    print("anchors OK" if not bad else f"{len(bad)} broken anchors")


if __name__ == "__main__":
    main()

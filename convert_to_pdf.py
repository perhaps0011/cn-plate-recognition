"""Convert 项目报告.md to PDF with GitHub-style rendering via Playwright."""
import asyncio
import tempfile
from pathlib import Path

import markdown
from playwright.async_api import async_playwright

md_path = Path(__file__).resolve().parent / "项目报告.md"
pdf_path = Path(__file__).resolve().parent / "项目报告.pdf"

md_content = md_path.read_text(encoding="utf-8")

# Convert markdown → HTML with GitHub-flavored extensions
html_body = markdown.markdown(
    md_content,
    extensions=[
        "markdown.extensions.extra",
        "markdown.extensions.codehilite",
        "markdown.extensions.toc",
        "markdown.extensions.nl2br",
    ],
)

github_css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.7;
  color: #24292f;
  background: #fff;
  padding: 48px 56px;
  max-width: 960px;
  margin: 0 auto;
}

h1 { font-size: 28px; font-weight: 700; margin: 0 0 8px 0; padding-bottom: 8px; border-bottom: 1px solid #d0d7de; text-align: center; }
h2 { font-size: 22px; font-weight: 600; margin: 28px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid #d0d7de; }
h3 { font-size: 18px; font-weight: 600; margin: 22px 0 10px 0; }
h4 { font-size: 15px; font-weight: 600; margin: 18px 0 8px 0; }

p { margin: 0 0 12px 0; }

table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0 18px 0;
  font-size: 13px;
  page-break-inside: avoid;
}
th, td {
  padding: 8px 12px;
  border: 1px solid #d0d7de;
  text-align: left;
}
th { background: #f6f8fa; font-weight: 600; }
tr:nth-child(even) { background: #f8f9fa; }

pre, code {
  font-family: 'SF Mono', Consolas, 'Courier New', monospace;
}
code {
  font-size: 12.5px;
  background: rgba(175,184,193,0.2);
  padding: 2px 6px;
  border-radius: 4px;
}
pre {
  background: #f6f8fa;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: 14px 16px;
  margin: 12px 0 18px 0;
  overflow-x: auto;
  line-height: 1.5;
  page-break-inside: avoid;
}
pre code {
  background: none;
  padding: 0;
  border-radius: 0;
}

blockquote {
  border-left: 4px solid #d0d7de;
  padding: 0 16px;
  color: #57606a;
  margin: 12px 0;
}

ul, ol { padding-left: 24px; margin: 8px 0 12px 0; }
li { margin: 4px 0; }

hr { border: none; border-top: 1px solid #d0d7de; margin: 24px 0; }

strong { font-weight: 700; }
a { color: #0969da; text-decoration: none; }

img { max-width: 100%; border-radius: 6px; }

@media print {
  body { padding: 32px 40px; }
  h1, h2, h3, h4 { page-break-after: avoid; }
}
"""

html_full = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<base href="{Path(__file__).resolve().parent.as_posix()}/">
<style>{github_css}</style>
</head>
<body>
{html_body}
</body>
</html>
"""


async def main():
    # Write HTML to a temp file so local images load via file:// protocol
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    )
    tmp.write(html_full)
    tmp_path = tmp.name
    tmp.close()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file://{tmp_path}", wait_until="networkidle")
        await page.pdf(
            path=str(pdf_path),
            format="A4",
            margin={"top": "24mm", "bottom": "24mm", "left": "20mm", "right": "20mm"},
            print_background=True,
        )
        await browser.close()

    Path(tmp_path).unlink()
    print(f"PDF saved to {pdf_path}")


asyncio.run(main())

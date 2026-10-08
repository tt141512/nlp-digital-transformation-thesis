# -*- coding: utf-8 -*-
"""将论文 Markdown 渲染为仓库根目录 index.html，并列出全部仓库文件。"""
import os, re, html

ROOT = os.path.dirname(os.path.abspath(__file__))

def md_to_html(md: str) -> str:
    out, lines = [], md.split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        # 表格
        if line.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i+1]):
            tbl = [line]
            j = i + 1
            while j < n and lines[j].strip().startswith("|"):
                tbl.append(lines[j]); j += 1
            out.append(render_table(tbl))
            i = j; continue
        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            txt = inline(m.group(2))
            cls = {1:"h1",2:"h2",3:"h3",4:"h4"}[lvl]
            out.append(f'<{cls}>{txt}</{cls}>')
            i += 1; continue
        # 引用块
        if line.startswith(">"):
            out.append(f'<blockquote>{inline(line[1:].strip())}</blockquote>')
            i += 1; continue
        # 无序列表
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(f"<li>{inline(re.sub(r'^\s*[-*]\s+','',lines[i]))}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue
        if line.strip() == "":
            i += 1; continue
        out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out)

def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    return t

def render_table(rows):
    head = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = []
    for r in rows[2:]:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        body.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
    th = "<tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
    return f'<table class="dt"><thead>{th}</thead><tbody>{"".join(body)}</tbody></table>'

def scan_files():
    groups = {
        "论文与说明": ["论文正文.md", "README.md", "LICENSE"],
        "测度构建代码 (code/)": ["00_generate_demo_data.py","01_collect_reports.py","02_clean_and_segment.py",
                                  "03_build_lexicon.py","04_compute_index.py","05_regression.py","utils.py","run_all.py","build_docx.py"],
        "词典与标注样本 (dict/ & data/)": ["seed_lexicon.txt","segmentation_dict.txt","stopwords_zh.txt","expanded_lexicon.csv",
                                            "manual_coding_sample.csv","heldout_terms.txt"],
        "结果表与图表 (results/)": ["T1_descriptive.md","T2_validity.md","T3_baseline.md","T4_mechanism.md",
                                     "T5_heterogeneity.md","T6_robustness.md","T7_endogeneity.md",
                                     "fig1_index_trend.png","fig2_industry_and_scatter.png","fig3_validity_scatter.png",
                                     "fig4_dimensions.png","fig5_coefplot.png"],
        "数据说明 (docs/)": ["data_source_statement.md","measurement_construction.md","variable_dictionary.md"],
    }
    html_parts = []
    for title, files in groups.items():
        items = []
        for f in files:
            # 定位文件（可能在子目录）
            cand = [os.path.join(ROOT, f), os.path.join(ROOT, "paper", f),
                    os.path.join(ROOT, "data", f), os.path.join(ROOT, "data","lexicon",f),
                    os.path.join(ROOT, "data","raw",f), os.path.join(ROOT, "results","tables",f),
                    os.path.join(ROOT, "results","figures",f), os.path.join(ROOT, "docs", f)]
            path = next((c for c in cand if os.path.exists(c)), None)
            if path:
                rel = os.path.relpath(path, ROOT).replace("\\", "/")
                items.append(f'<li><a href="{rel}">{f}</a> <span class="muted">{rel}</span></li>')
        if items:
            html_parts.append(f'<div class="fgroup"><h3>{title}</h3><ul class="files">{"".join(items)}</ul></div>')
    return "\n".join(html_parts)

md = open(os.path.join(ROOT, "paper", "论文正文.md"), encoding="utf-8").read()
body = md_to_html(md)
files_html = scan_files()

doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>企业数字化转型的文本测度构建及其对全要素生产率的影响</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.85;color:#1f2933;
  max-width:920px;margin:0 auto;padding:32px 20px 80px;background:#fff}}
header{{border-bottom:3px solid #2b6cb0;padding-bottom:14px;margin-bottom:8px}}
h1{{font-size:1.7em;color:#1a365d;margin:.2em 0}}
.sub{{color:#52606d;font-size:.95em}}
h2{{font-size:1.3em;color:#1a365d;margin-top:1.8em;border-left:4px solid #2b6cb0;padding-left:10px}}
h3{{font-size:1.08em;color:#2b6cb0;margin-top:1.4em}}
h4{{color:#3e4c59}}
p{{margin:.7em 0;text-align:justify}}
blockquote{{background:#ebf8ff;border-left:4px solid #4299e1;margin:1em 0;padding:.6em 1em;color:#2c5282}}
table.dt{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.86em}}
table.dt th,table.dt td{{border:1px solid #cbd5e0;padding:6px 9px;text-align:left;vertical-align:top}}
table.dt thead{{background:#2b6cb0;color:#fff}}
table.dt tbody tr:nth-child(even){{background:#f7fafc}}
ul.files{{list-style:none;padding-left:0}}
ul.files li{{padding:3px 0;border-bottom:1px dashed #e4e7eb}}
ul.files a{{color:#2b6cb0;text-decoration:none;font-weight:600}}
ul.files a:hover{{text-decoration:underline}}
.muted{{color:#9aa5b1;font-size:.82em;margin-left:6px}}
.fgroup{{background:#f7fafc;border:1px solid #e4e7eb;border-radius:8px;padding:12px 16px;margin:14px 0}}
.fgroup h3{{margin-top:.2em;color:#1a365d}}
.toc{{background:#f0f4f8;border-radius:8px;padding:12px 18px;margin:16px 0;font-size:.92em}}
.toc a{{color:#2b6cb0}}
footer{{margin-top:40px;border-top:1px solid #cbd5e0;padding-top:14px;color:#52606d;font-size:.85em}}
code{{background:#edf2f7;padding:1px 5px;border-radius:4px;font-size:.9em}}
</style></head>
<body>
<header>
<h1>企业数字化转型的文本测度构建及其对全要素生产率的影响</h1>
<div class="sub">基于 A 股上市公司年报文本的 NLP 分析（2013—2023） · 模拟毕业论文（预演论文）配套材料</div>
</header>
<div class="toc"><strong>本页内容</strong>：① 论文全文（中英文摘要、五章正文、参考文献） ② <a href="#repo">仓库文件索引</a>（代码 / 词典 / 数据 / 结果 / 说明，均可点击下载）</div>
<h2>论文全文</h2>
{body}
<h2 id="repo">仓库文件索引（可点击下载 / 查看）</h2>
<p class="sub">本仓库为端到端可复现的 NLP 测度流水线：语料 → 分词 → 词典构建 → 面板指标 → 实证检验。所有脚本以固定随机种子生成仿真数据，可在本地一键复现。</p>
{files_html}
<footer>
<p><strong>诚实披露：</strong>本仓库语料与财务面板为固定随机种子生成的仿真数据，用于演示"NLP 测度构建 + 实证检验"方法的完整性与正确性；实证数字应理解为方法验证而非真实经济发现。真实数据替换方式见 <code>docs/data_source_statement.md</code>。</p>
<p>复现：<code>pip install -r requirements.txt &amp;&amp; python code/run_all.py</code> ｜ 导出 Word：<code>python code/run_all.py --build-docx</code></p>
</footer>
</body></html>"""

open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(doc)
print("index.html generated:", len(doc), "bytes")

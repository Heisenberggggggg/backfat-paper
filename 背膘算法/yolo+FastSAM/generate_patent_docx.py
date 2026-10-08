"""Convert PATENT_DRAFT_algorithm_v2.md into a Word docx that mimics the reference patent template format.

Template format observed:
- Page: A4 portrait, margins ~ 1 inch
- Single 1x1 table wrapping all content
- Font: KaiTi (Chinese 楷体)
- Title: 16pt bold centered
- Section heading: 14pt bold
- Body: 14pt regular
- Caption: 12pt
- Bold prefix in body paragraphs
"""
import sys
import io
import re
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

INPUT_MD = r'F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM\PATENT_DRAFT_algorithm_v2.md'
OUTPUT_DOCX = r'F:\Postgraduate\Lab\专利软著\专利\新建文件夹\一种基于P2点自动定位与多模态特征融合的母猪背膘非接触测量方法_v2_生成.docx'

FONT_NAME = '楷体'  # KaiTi
SIZE_TITLE = Pt(16)
SIZE_BODY = Pt(14)
SIZE_CAPTION = Pt(12)


def set_run_font(run, size=SIZE_BODY, bold=False):
    run.font.name = FONT_NAME
    run.font.size = size
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), FONT_NAME)
    rfonts.set(qn('w:ascii'), FONT_NAME)
    rfonts.set(qn('w:hAnsi'), FONT_NAME)


def add_para_in_cell(cell, text, size=SIZE_BODY, bold=False, align=None, first_line_indent=True):
    p = cell.add_paragraph()
    if align is not None:
        p.alignment = align
    if first_line_indent and text and not bold:
        p.paragraph_format.first_line_indent = Pt(28)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)

    if text:
        parts = re.split(r'(\*\*[^*]+\*\*)', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                run = p.add_run(part[2:-2])
                set_run_font(run, size=size, bold=True)
            else:
                run = p.add_run(part)
                set_run_font(run, size=size, bold=bold)
    return p


def add_heading_in_cell(cell, text, size=SIZE_BODY):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text.replace('*', ''))
    set_run_font(run, size=size, bold=True)
    return p


def add_title_in_cell(cell, text):
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    set_run_font(run, size=SIZE_TITLE, bold=True)
    return p


def add_empty_para(cell):
    p = cell.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run('')
    set_run_font(run, size=SIZE_BODY)


def parse_md_table(lines, start_idx):
    rows = []
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith('|'):
            break
        if re.match(r'^\|[\s\-:|]+\|\s*$', line):
            i += 1
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        rows.append(cells)
        i += 1
    return rows, i


def add_md_table_to_cell(cell, rows):
    if not rows:
        return
    n_cols = len(rows[0])
    table = cell.add_table(rows=len(rows), cols=n_cols)
    table.style = 'Table Grid'
    table.autofit = True
    for ri, row_data in enumerate(rows):
        for ci, cdata in enumerate(row_data):
            if ci >= n_cols:
                break
            tc = table.rows[ri].cells[ci]
            tc.text = ''
            p = tc.paragraphs[0]
            p.paragraph_format.line_spacing = 1.2
            parts = re.split(r'(\*\*[^*]+\*\*)', cdata)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    set_run_font(run, size=Pt(11), bold=True)
                else:
                    run = p.add_run(part)
                    set_run_font(run, size=Pt(11), bold=(ri == 0))


def main():
    with open(INPUT_MD, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)

    big_table = doc.add_table(rows=1, cols=1)
    big_table.style = 'Table Grid'
    big_cell = big_table.rows[0].cells[0]

    p0 = big_cell.paragraphs[0]
    p0._element.getparent().remove(p0._element)

    i = 0
    n = len(lines)
    title_done = False
    while i < n:
        line = lines[i].rstrip('\n')
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == '---':
            i += 1
            continue

        if stripped.startswith('# ') and not title_done:
            add_title_in_cell(big_cell, stripped[2:].strip())
            title_done = True
            i += 1
            continue

        if stripped.startswith('## '):
            add_heading_in_cell(big_cell, stripped[3:].strip(), size=SIZE_BODY)
            i += 1
            continue

        if stripped.startswith('### '):
            add_heading_in_cell(big_cell, stripped[4:].strip(), size=SIZE_BODY)
            i += 1
            continue

        if stripped.startswith('#### '):
            add_heading_in_cell(big_cell, stripped[5:].strip(), size=SIZE_BODY)
            i += 1
            continue

        if stripped.startswith('>'):
            text = stripped.lstrip('> ').strip()
            add_para_in_cell(big_cell, text, size=SIZE_BODY, bold=False, first_line_indent=True)
            i += 1
            continue

        if stripped.startswith('|'):
            rows, end_i = parse_md_table(lines, i)
            if rows:
                first_content = ' '.join(rows[0]) if rows else ''
                if len(rows) <= 4 and len(rows[0]) == 2 and (
                    '专利名称' in first_content or '字段' in first_content
                ):
                    for r in rows:
                        if r[0].strip() == '字段' or '---' in r[0]:
                            continue
                        label = r[0].strip().replace('**', '')
                        value = r[1].strip().replace('**', '')
                        if not label and not value:
                            continue
                        line_text = f"{label}：{value}" if label else value
                        add_para_in_cell(big_cell, line_text, size=SIZE_BODY,
                                         bold=False, first_line_indent=False)
                else:
                    add_md_table_to_cell(big_cell, rows)
                    add_empty_para(big_cell)
            i = end_i
            continue

        if stripped.startswith('- '):
            text = stripped[2:].strip()
            add_para_in_cell(big_cell, '• ' + text, size=SIZE_BODY,
                             bold=False, first_line_indent=False)
            i += 1
            continue

        add_para_in_cell(big_cell, stripped, size=SIZE_BODY, bold=False,
                         first_line_indent=True)
        i += 1

    os.makedirs(os.path.dirname(OUTPUT_DOCX), exist_ok=True)
    doc.save(OUTPUT_DOCX)
    print(f"Done. Output: {OUTPUT_DOCX}")
    print(f"File size: {os.path.getsize(OUTPUT_DOCX) / 1024:.1f} KB")


if __name__ == '__main__':
    main()

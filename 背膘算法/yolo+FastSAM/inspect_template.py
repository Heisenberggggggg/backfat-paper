"""Inspect the patent template docx to understand its formatting structure."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from docx import Document

template_path = r'F:\Postgraduate\Lab\专利软著\专利\新建文件夹\一种基于真实尺度掩膜深度图与多模态特征融合的母猪背膘非接触测量方法.docx'

doc = Document(template_path)

print("=" * 80)
print("段落结构（前 80 段）")
print("=" * 80)
for i, p in enumerate(doc.paragraphs[:80]):
    style = p.style.name if p.style else 'None'
    text = p.text[:100] if p.text else '[empty]'
    print(f"[{i:03d}] style={style} | {text}")

# inspect inside cell
print()
print("=" * 80)
print("Cell 内段落（前 80 段）含字体")
print("=" * 80)
t = doc.tables[0]
cell = t.rows[0].cells[0]
for i, p in enumerate(cell.paragraphs[:80]):
    style = p.style.name if p.style else 'None'
    text = p.text[:80] if p.text else '[empty]'
    runs_info = []
    for r in p.runs[:2]:
        runs_info.append(f"f={r.font.name}|sz={r.font.size}|b={r.font.bold}")
    print(f"[{i:03d}] {style} | {text}")
    if runs_info:
        print(f"      L {' '.join(runs_info)}")
print(f"Cell paragraphs total: {len(cell.paragraphs)}")
sys.exit(0)

print()
print("=" * 80)
print(f"表格数: {len(doc.tables)}")
print("=" * 80)
for ti, table in enumerate(doc.tables):
    print(f"\n--- Table {ti+1} ({len(table.rows)} rows x {len(table.columns)} cols) ---")
    for ri, row in enumerate(table.rows[:3]):
        for ci, cell in enumerate(row.cells):
            print(f"  R{ri}C{ci}: {cell.text[:80]}")

print()
print("=" * 80)
print("默认字体 & 标题样式")
print("=" * 80)
from docx.shared import Pt
styles = doc.styles
for s in ['Normal', 'Heading 1', 'Heading 2', 'Heading 3']:
    try:
        st = styles[s]
        font = st.font
        print(f"{s}: font.name={font.name}, font.size={font.size}, bold={font.bold}")
    except KeyError:
        pass

print()
print("=" * 80)
print("页面设置")
print("=" * 80)
for si, sec in enumerate(doc.sections):
    print(f"Section {si}: page_w={sec.page_width}, page_h={sec.page_height}, "
          f"left={sec.left_margin}, right={sec.right_margin}, top={sec.top_margin}, bottom={sec.bottom_margin}")

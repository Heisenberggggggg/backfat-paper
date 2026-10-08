"""Read reference patent docx and dump text to file."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from docx import Document

src = r'D:\Tencent\WeChat\file\xwechat_files\wxid_gplw0674a21q22_f2fb\msg\file\2025-08\一种基于3D视觉与边缘计算的妊娠母猪闭环精准饲喂系统及方法(2).docx'
out = r'F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM\reference_patent.txt'

doc = Document(src)

with open(out, 'w', encoding='utf-8') as f:
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip():
            f.write(f"{p.text}\n")
    f.write("\n\n===== TABLES =====\n\n")
    for ti, table in enumerate(doc.tables):
        f.write(f"\n--- Table {ti+1} ---\n")
        for row in table.rows:
            cells = [c.text.strip().replace('\n', ' / ') for c in row.cells]
            f.write(" | ".join(cells) + "\n")

print(f"Done. Wrote {out}")

import io
from pypdf import PdfReader

src = r'C:/Users/mayan/Downloads/whole audit by fable.pdf'
r = PdfReader(src)
print('pages', len(r.pages))
out = io.open(r'C:/Users/mayan/Downloads/audit_extracted.txt', 'w', encoding='utf-8')
for i, p in enumerate(r.pages):
    out.write(f'\n===== PAGE {i + 1} =====\n\n')
    out.write(p.extract_text() or '')
out.close()
print('done')
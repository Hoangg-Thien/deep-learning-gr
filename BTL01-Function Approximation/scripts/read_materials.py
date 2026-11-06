"""Extract all supplied reference material without modifying originals."""
from pathlib import Path
import hashlib
import json

from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'source_materials'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(ROOT.glob('*.pdf')):
        reader = PdfReader(path)
        pages = [f'\n--- PAGE {i + 1} ---\n{page.extract_text(extraction_mode="layout")}'
                 for i, page in enumerate(reader.pages)]
        (OUT / f'{path.stem}.txt').write_text('\n'.join(pages), encoding='utf-8')
        manifest.append({'file': path.name, 'pages': len(pages),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    path = ROOT / 'Sample_Code.html'
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    for element in soup(['script', 'style']):
        element.decompose()
    (OUT / 'Sample_Code.txt').write_text(soup.get_text('\n', strip=True), encoding='utf-8')
    manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

"""Render supplied PDFs and final PDFs to contact sheets for visual inspection."""
from pathlib import Path
import argparse
import math

from PIL import Image, ImageDraw
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]


def render(path, output):
    document = pdfium.PdfDocument(path)
    thumbnails = []
    for index in range(len(document)):
        page = document[index]
        bitmap = page.render(scale=0.7)
        picture = bitmap.to_pil().convert('RGB')
        panel = Image.new('RGB', (picture.width + 12, picture.height + 32), 'white')
        panel.paste(picture, (6, 26))
        ImageDraw.Draw(panel).text((6, 6), f'Page {index + 1}', fill='black')
        thumbnails.append(panel)
        if path.name == 'MLP_Exercise - Yeu Cau.pdf' and index < 2:
            page.render(scale=1.5).to_pil().save(output / f'official_page_{index + 1}.png')
        if path.name == 'ACADEMIC_REPORT.pdf' and index in [3, 9, 13]:
            page.render(scale=1.3).to_pil().save(output / f'report_page_{index + 1}.png')
    columns = min(4, len(thumbnails))
    width, height = max(p.width for p in thumbnails), max(p.height for p in thumbnails)
    sheet = Image.new('RGB', (columns * width, math.ceil(len(thumbnails)/columns) * height), '#bbc3c9')
    for index, picture in enumerate(thumbnails):
        sheet.paste(picture, ((index % columns)*width, (index // columns)*height))
    result = output / (path.stem.replace(' ', '_') + '_contact.png')
    sheet.save(result)
    print(f'{path.name}: {len(document)} pages -> {result.relative_to(ROOT)}')
    document.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'results/verification/pdf_previews'
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(ROOT.glob('*.pdf'))
    if not args.sources_only:
        paths += [ROOT / 'reports/ACADEMIC_REPORT.pdf', ROOT / 'slides/BTL01_PRESENTATION.pdf']
    for path in paths:
        render(path, output)


if __name__ == '__main__':
    main()

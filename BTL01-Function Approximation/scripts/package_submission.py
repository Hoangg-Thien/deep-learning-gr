"""Create an audited submission ZIP; omit environments, caches and smoke runs."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit import audit


def submission_files():
    root_files = ['main.py', 'README.md', 'requirements.txt', 'pyproject.toml', '.gitignore',
                  'MLP_Exercise - Yeu Cau.pdf', 'MLP_Exercise.pdf', 'Sample_Code.html', 'Sample_Report.pdf']
    paths = [ROOT / name for name in root_files]
    for directory in ['src', 'tests', 'configs', 'scripts', 'data', 'reports', 'slides']:
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ['.pyc', '.pyo'])
    for directory in ['logs', 'tables', 'figures', 'checkpoints', 'verification']:
        paths.extend(p for p in (ROOT / 'results' / directory).rglob('*') if p.is_file())
    for name in ['config.json', 'manifest.json', 'selection.json', 'validation_records.json', 'official_execution.txt']:
        paths.append(ROOT / 'results' / name)
    return sorted(set(paths))


def main():
    result = audit(ROOT / 'results')
    assert result['status'] == 'PASS' and result['runs_verified'] == 72
    manifest_path = ROOT / 'reports/DELIVERABLE_MANIFEST.json'
    paths = [p for p in submission_files() if p != manifest_path]
    manifest = {p.relative_to(ROOT).as_posix(): dict(bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    paths.append(manifest_path)
    output = ROOT / 'submission/BTL01_Nhom_HoTen_MSSV.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert not any(any(part in ['.venv', '.venv-run', '__pycache__', '.pytest_cache', '.mplconfig', 'smoke']
                           for part in Path(name).parts) for name in names)
        for name, entry in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256']
    print(json.dumps(dict(archive=output.relative_to(ROOT).as_posix(), files=len(paths),
                          bytes=output.stat().st_size,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                          excludes=['venv', 'caches', 'smoke', 'demo']), indent=2))


if __name__ == '__main__':
    main()

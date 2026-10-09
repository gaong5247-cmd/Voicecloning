"""Stream a ZIP64 portable archive without PowerShell MemoryStream size limits."""
import sys, zipfile
from pathlib import Path
root=Path(sys.argv[1]); output=Path(sys.argv[2])
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file(): archive.write(path,Path(root.name)/path.relative_to(root))
with zipfile.ZipFile(output) as archive:
    if archive.testzip() is not None: raise RuntimeError('Archive integrity verification failed')
print(f'Created {output} ({output.stat().st_size} bytes), ZIP64 verification passed')

"""Build portable ZIP64 archives, with a fast mode for XPU's large native DLLs."""
import argparse
import time
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--fast", action="store_true",
                    help="Use fast deflate and check ZIP structure rather than re-decompressing every DLL")
args = parser.parse_args()
root, output = args.root, args.output
if not root.is_dir():
    raise NotADirectoryError(root)
files = sorted(path for path in root.rglob("*") if path.is_file())
if not files:
    raise RuntimeError("Portable directory is empty")
output.parent.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
level = 1 if args.fast else 3
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                     compresslevel=level, allowZip64=True) as archive:
    for path in files:
        archive.write(path, Path(root.name) / path.relative_to(root))

# Full ZIP CRC validation reads and decompresses every member a second time.
# For huge XPU archives, verify the complete central directory/file manifest;
# zipfile already computed per-entry CRCs during writing.
with zipfile.ZipFile(output) as archive:
    expected = {str(Path(root.name) / path.relative_to(root)).replace("\\", "/") for path in files}
    actual = set(archive.namelist())
    if expected != actual:
        raise RuntimeError(f"ZIP entry mismatch: missing={len(expected-actual)} extra={len(actual-expected)}")
    if not args.fast:
        damaged = archive.testzip()
        if damaged is not None:
            raise RuntimeError(f"ZIP CRC verification failed for {damaged}")
print(f"Created {output} ({output.stat().st_size} bytes), ZIP64 manifest passed, "
      f"full_crc_checked={not args.fast}, seconds={time.perf_counter()-start:.1f}", flush=True)

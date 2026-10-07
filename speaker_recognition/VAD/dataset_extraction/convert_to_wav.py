import sys
import subprocess
from pathlib import Path

INPUT_ROOT = (__import__("paths").DATA_DIR / "dataset")
OUTPUT_ROOT = (__import__("paths").DATA_DIR / "wav_dataset")
TARGET_SR = 16000
TARGET_CHANNELS = 1
EXTS = {".m4a"}
SKIP_IF_EXISTS = True

def have_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except Exception:
        return False

def convert_one(in_path: Path, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(in_path),
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SR),
        "-vn",
        str(out_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0

def main():
    if not have_ffmpeg():
        print("ERROR: ffmpeg not found. Install it and make sure it's in PATH.")
        sys.exit(1)

    if not INPUT_ROOT.exists():
        print(f"ERROR: INPUT_ROOT not found: {INPUT_ROOT.resolve()}")
        sys.exit(1)

    m4a_files = [p for p in INPUT_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]

    if not m4a_files:
        print(f"No .m4a files found under: {INPUT_ROOT.resolve()}")
        return

    print(f"Found {len(m4a_files)} .m4a files under {INPUT_ROOT}")

    ok = fail = skipped = 0

    for in_path in m4a_files:
        rel = in_path.relative_to(INPUT_ROOT)              
        out_path = OUTPUT_ROOT / rel.with_suffix(".wav")   

        if SKIP_IF_EXISTS and out_path.exists():
            skipped += 1
            print(f"[SKIP] {out_path} already exists")
            continue

        print(f"[CONVERT] {in_path} -> {out_path}")
        if convert_one(in_path, out_path):
            ok += 1
        else:
            fail += 1
            print(f"  [FAIL] ffmpeg failed for: {in_path}")

    print("\n=== DONE ===")
    print(f"Converted: {ok}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {fail}")
    print(f"Output:    {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()

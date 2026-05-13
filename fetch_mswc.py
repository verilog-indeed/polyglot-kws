"""
Download MSWC audio from HuggingFace Hub for language-head training.

Fetches N random clips per language — word labels are irrelevant.
The notebook assigns kw_idx=-1 to every MSWC sample, masking it from the
keyword loss; only the language label matters.

Shards are streamed (not downloaded to disk): we open each tar.gz over HTTP
and extract audio files until the per-language quota is met, then close the
connection. Typically 1–2 shards per language are enough.

Dataset layout on HF:
    data/opus/{lang}/{split}/audio/{n}.tar.gz   (n = 0, 1, 2, …)

Output layout:
    <root>/mswc/{lang}/clip_NNNNN.wav

Usage:
    pip install huggingface_hub soundfile torchaudio
    python fetch_mswc.py --root ./kws_data
    python fetch_mswc.py --root ./kws_data --per-lang 200 --langs en de
"""
import argparse
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
LANGS       = ["en", "de", "tr", "ar", "fr", "fa"]
HF_REPO     = "datasets/MLCommons/ml_spoken_words"
HF_SPLIT    = "train"
AUDIO_EXTS  = {".opus", ".wav", ".flac", ".mp3"}


def _decode(audio_bytes: bytes, suffix: str = ".opus") -> np.ndarray:
    import torchaudio

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        wav, sr = torchaudio.load(tmp_path)
        wav = wav.mean(0)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav.numpy().astype(np.float32)
    finally:
        os.unlink(tmp_path)


def fetch_lang(lang: str, out_dir: Path, per_lang: int, split: str) -> None:
    from huggingface_hub import HfFileSystem
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(out_dir.glob("clip_*.wav")))
    if count >= per_lang:
        print(f"  [{lang}] already complete ({count} clips) — skipping")
        return

    fs = HfFileSystem()
    shard_glob = f"{HF_REPO}/data/opus/{lang}/{split}/audio/*.tar.gz"
    shards = sorted(
        fs.glob(shard_glob),
        key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else 0,
    )
    if not shards:
        print(f"  [{lang}] no shards found at {shard_glob}")
        return

    print(f"  [{lang}] {count}/{per_lang} clips present, "
          f"{len(shards)} shard(s) available")

    for shard_path in shards:
        if count >= per_lang:
            break

        shard_name = Path(shard_path).name
        print(f"    [{lang}] streaming {shard_name} ...", end=" ", flush=True)
        found = 0
        errors = 0
        try:
            # r|gz = streaming (pipe) mode: reads sequentially, no seeking.
            # We never download more bytes than we actually process.
            with fs.open(shard_path, "rb") as fobj:
                with tarfile.open(fileobj=fobj, mode="r|gz") as tar:
                    for member in tar:
                        if count >= per_lang:
                            break
                        if not member.isfile():
                            continue
                        ext = Path(member.name).suffix.lower()
                        if ext not in AUDIO_EXTS:
                            continue
                        try:
                            fobj_member = tar.extractfile(member)
                            if fobj_member is None:
                                continue
                            data = fobj_member.read()
                            arr  = _decode(data, ext)
                        except Exception:
                            errors += 1
                            continue
                        sf.write(
                            str(out_dir / f"clip_{count:05d}.wav"),
                            arr, SAMPLE_RATE, subtype="PCM_16",
                        )
                        count += 1
                        found += 1
        except Exception as e:
            print(f"\n    [{lang}] shard error: {e}")
            continue

        print(f"extracted {found}  errors {errors}  (total {count}/{per_lang})")

    print(f"  [{lang}] done — {count} clips saved to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",     type=Path, default=Path("./kws_data"))
    p.add_argument("--per-lang", type=int,  default=400,
                   help="Audio clips to collect per language (default: 400)")
    p.add_argument("--split",    default=HF_SPLIT,
                   help="HF dataset split: train / dev / test")
    p.add_argument("--langs",    nargs="+", default=LANGS)
    args = p.parse_args()

    try:
        from huggingface_hub import HfFileSystem  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Install:  pip install huggingface_hub soundfile torchaudio"
        ) from e

    mswc_dir = args.root / "mswc"
    mswc_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {mswc_dir}  (split={args.split}, per_lang={args.per_lang})")
    for lang in args.langs:
        fetch_lang(lang, mswc_dir / lang, args.per_lang, args.split)


if __name__ == "__main__":
    main()

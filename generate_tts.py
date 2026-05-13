"""
Stand-alone TTS + silence data generation for the multilingual KWS project.

Run once before opening the notebook. Keeping this out of the notebook
avoids dependency conflicts between edge-tts (aiohttp etc.) and the
Torch / Colab runtime.

Usage:
    pip install edge-tts librosa soundfile numpy
    python generate_tts.py --root /content/kws_data

Output layout:
    <root>/tts/<lang>/<concept>/<wordslug>__<voice>.wav   # keyword + unknown
    <root>/tts/_silence/silence/silence_NNNN.wav          # silence class
"""
import argparse
import asyncio
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import edge_tts
from kws_config import VOICES, KEYWORDS, VARIANTS, UNKNOWN_WORDS

# ---------------------------------------------------------------------------
# Audio config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION_S  = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_S)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


async def _tts_one(text: str, voice: str, out_mp3: Path,
                   rate: str = "+0%", pitch: str = "+0Hz") -> None:
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(out_mp3))


def _mp3_to_wav16k(mp3: Path, wav: Path) -> None:
    y, _ = librosa.load(str(mp3), sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) >= NUM_SAMPLES:
        peak  = int(np.argmax(np.abs(y)))
        start = max(0, peak - NUM_SAMPLES // 2)
        y     = y[start:start + NUM_SAMPLES]
    if len(y) < NUM_SAMPLES:
        pad = NUM_SAMPLES - len(y)
        y   = np.pad(y, (pad // 2, pad - pad // 2))
    sf.write(str(wav), y.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")


async def synthesize(text: str, voice: str, out_wav: Path,
                     rate: str = "+0%", pitch: str = "+0Hz",
                     overwrite: bool = False) -> None:
    if out_wav.exists() and not overwrite:
        return
    mp3 = out_wav.with_suffix(".mp3")
    await _tts_one(text, voice, mp3, rate=rate, pitch=pitch)
    _mp3_to_wav16k(mp3, out_wav)
    mp3.unlink(missing_ok=True)


async def build_tts_corpus(tts_dir: Path, overwrite: bool = False) -> None:
    # jobs: (text, voice, rate, pitch, out_path)
    jobs = []
    for lang in VOICES:
        for concept, words in KEYWORDS[lang].items():
            for word in words:
                for voice in VOICES[lang]:
                    for vi, (rate, pitch) in enumerate(VARIANTS):
                        # Variant 0 keeps the original filename for backward compat.
                        suffix = f"__v{vi}" if vi > 0 else ""
                        out = tts_dir / lang / concept / f"{slug(word)}__{voice}{suffix}.wav"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        jobs.append((word, voice, rate, pitch, out))
        for word in UNKNOWN_WORDS[lang]:
            for voice in VOICES[lang]:
                for vi, (rate, pitch) in enumerate(VARIANTS):
                    suffix = f"__v{vi}" if vi > 0 else ""
                    out = tts_dir / lang / "unknown" / f"{slug(word)}__{voice}{suffix}.wav"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    jobs.append((word, voice, rate, pitch, out))

    print(f"queued {len(jobs)} TTS jobs in {tts_dir}")
    skipped = errors = 0
    for i, (text, voice, rate, pitch, out) in enumerate(jobs):
        try:
            if out.exists() and not overwrite:
                skipped += 1
            else:
                await synthesize(text, voice, out, rate=rate, pitch=pitch, overwrite=overwrite)
        except Exception as e:
            errors += 1
            print(f"  [skip] {text!r} via {voice} (rate={rate} pitch={pitch}): {e}")
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(jobs)} processed  (skipped={skipped}, errors={errors})")
    print(f"TTS generation done. total={len(jobs)} skipped={skipped} errors={errors}")


def make_silence_clips(tts_dir: Path, total: int = 480, sigma: float = 0.005, seed: int = 1234) -> None:
    rng = np.random.default_rng(seed)
    d = tts_dir / "_silence" / "silence"
    d.mkdir(parents=True, exist_ok=True)
    existing = len(list(d.glob("silence_*.wav")))
    if existing >= total:
        print(f"silence: {existing} clips already present in {d} — skipping")
        return
    for i in range(total):
        y = rng.normal(0, sigma, NUM_SAMPLES).astype(np.float32)
        sf.write(str(d / f"silence_{i:04d}.wav"), y, SAMPLE_RATE, subtype="PCM_16")
    print(f"wrote {total} silence clips to {d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/content/kws_data"),
                   help="Dataset root directory.")
    p.add_argument("--silence", type=int, default=480, help="Number of silence clips to write.")
    p.add_argument("--overwrite", action="store_true", help="Re-synthesise existing WAVs.")
    p.add_argument("--skip-tts", action="store_true", help="Skip TTS generation.")
    p.add_argument("--skip-silence", action="store_true", help="Skip silence generation.")
    args = p.parse_args()

    tts_dir = args.root / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_tts:
        asyncio.run(build_tts_corpus(tts_dir, overwrite=args.overwrite))
    if not args.skip_silence:
        make_silence_clips(tts_dir, total=args.silence)


if __name__ == "__main__":
    main()

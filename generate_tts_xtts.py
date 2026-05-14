"""
XTTS-v2 voice-cloned augmentation for Turkish (and optionally other languages).

edge-tts ships only 2 Turkish neural voices, capping the Turkish corpus to
~180 clips. Coqui XTTS-v2 has ~50 built-in studio speakers and supports tr,
so we generate the same keyword/unknown vocabulary in every studio voice.

The resulting wavs land alongside the edge-tts wavs in:
    <root>/tts/tr/<concept>/<wordslug>__xtts_<speaker>.wav

After running this script, re-run the notebook's `pack_tts_bundles(...,
overwrite=True)` so the new files are picked up by the .pt bundle.

Usage:
    pip install TTS soundfile librosa numpy
    python generate_tts_xtts.py --root /content/kws_data
    python generate_tts_xtts.py --root /content/kws_data --langs tr en
"""
import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from kws_config import KEYWORDS, UNKNOWN_WORDS

SAMPLE_RATE = 16000
DURATION_S  = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_S)

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS-v2 supports: en es fr de it pt pl tr ru nl cs ar zh-cn ja hu ko
XTTS_LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
              "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"}


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def speaker_slug(name: str) -> str:
    """XTTS studio speaker names contain spaces — slug them for filesystem use."""
    return name.replace(" ", "_").replace(".", "")


def trim_and_pad(y: np.ndarray) -> np.ndarray:
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) >= NUM_SAMPLES:
        peak  = int(np.argmax(np.abs(y)))
        start = max(0, peak - NUM_SAMPLES // 2)
        y     = y[start:start + NUM_SAMPLES]
    if len(y) < NUM_SAMPLES:
        pad = NUM_SAMPLES - len(y)
        y   = np.pad(y, (pad // 2, pad - pad // 2))
    return y.astype(np.float32)


def synthesize_corpus(tts_dir: Path, langs: list, overwrite: bool = False) -> None:
    from TTS.api import TTS
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading XTTS-v2 on {device} ...")
    tts = TTS(XTTS_MODEL).to(device)

    speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
    print(f"  {len(speakers)} XTTS studio speakers available")

    for lang in langs:
        if lang not in XTTS_LANGS:
            print(f"  [{lang}] not supported by XTTS-v2 — skipping")
            continue
        if lang not in KEYWORDS:
            print(f"  [{lang}] not in kws_config.KEYWORDS — skipping")
            continue

        out_root = tts_dir / lang
        # build (text, concept) list
        items = []
        for concept, words in KEYWORDS[lang].items():
            for w in words:
                items.append((w, concept))
        for w in UNKNOWN_WORDS[lang]:
            items.append((w, "unknown"))

        total = len(items) * len(speakers)
        print(f"  [{lang}] {len(items)} words × {len(speakers)} speakers = {total} clips")
        done = skipped = errors = 0

        for spk in speakers:
            for word, concept in items:
                out_dir = out_root / concept
                out_dir.mkdir(parents=True, exist_ok=True)
                out_wav = out_dir / f"{slug(word)}__xtts_{speaker_slug(spk)}.wav"
                if out_wav.exists() and not overwrite:
                    skipped += 1
                    continue
                try:
                    wav = tts.tts(text=word, speaker=spk, language=lang)
                    y   = np.asarray(wav, dtype=np.float32)
                    # XTTS outputs 24kHz — resample to 16kHz
                    if tts.synthesizer.output_sample_rate != SAMPLE_RATE:
                        y = librosa.resample(
                            y,
                            orig_sr=tts.synthesizer.output_sample_rate,
                            target_sr=SAMPLE_RATE,
                        )
                    y = trim_and_pad(y)
                    sf.write(str(out_wav), y, SAMPLE_RATE, subtype="PCM_16")
                    done += 1
                except Exception as e:
                    errors += 1
                    print(f"    [skip] {word!r} via {spk}: {e}")
                if (done + skipped + errors) % 25 == 0:
                    print(f"    [{lang}] {done + skipped + errors}/{total}  "
                          f"(done={done}, skipped={skipped}, errors={errors})")
        print(f"  [{lang}] complete — done={done}, skipped={skipped}, errors={errors}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/content/kws_data"))
    p.add_argument("--langs", nargs="+", default=["tr"],
                   help="Languages to augment with XTTS-v2 (default: tr).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    tts_dir = args.root / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    synthesize_corpus(tts_dir, args.langs, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

"""
fetch_kws_data.py — Pre-fetch and cache MSWC keyword audio for multilingual_kws_v2.ipynb

This script runs ONCE before the notebook and does all the heavy HuggingFace I/O:
  1. Reads mswc-metadata.json to build the keyword inventory instantly (no shard scan)
  2. Downloads MSWC tar shards via hf_hub_download (bounded HTTP request, no stall risk)
  3. Matches members using metadata-derived filename lookup (O(1) dict, no string parsing)
  4. Decodes audio, computes 49x40 log-mel features, saves float16 .npy to Drive
  5. Saves keyword_inventory.json in the exact format the notebook expects

After this script completes, multilingual_kws_v2.ipynb loads everything from Drive cache
and never touches HuggingFace.

Usage (Colab):
  # Upload mswc-metadata.json to your Drive first, then:
  !python fetch_kws_data.py \\
      --metadata /content/drive/MyDrive/mswc-metadata.json \\
      --root     /content/drive/MyDrive/kws_cache

Usage (local / debug):
  python fetch_kws_data.py --metadata ../mswc-metadata.json --root ./kws_cache --debug
"""
import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from huggingface_hub import HfFileSystem, hf_hub_download

# ── Audio constants — must match the notebook ─────────────────────────────────
SAMPLE_RATE = 16_000
N_MELS      = 40
N_FFT       = 640
WIN_LENGTH  = 640
HOP_LENGTH  = 320
F_MIN       = 20
F_MAX       = 8_000

# ── HF dataset coords ─────────────────────────────────────────────────────────
_HF_REPO_ID = "MLCommons/ml_spoken_words"
_AUDIO_EXTS = {".opus", ".wav", ".flac", ".mp3"}
_CV_MARKER  = "_common_voice_"


# ── Log-mel ───────────────────────────────────────────────────────────────────
_mel_xform: T.MelSpectrogram | None = None
_amp2db:    T.AmplitudeToDB   | None = None

def _init_transforms(device: str) -> None:
    global _mel_xform, _amp2db
    _mel_xform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX,
        power=2.0, center=False,
    ).to(device)
    _amp2db = T.AmplitudeToDB(stype="power", top_db=80).to(device)


def _wav_to_logmel(wav: torch.Tensor, device: str) -> np.ndarray:
    """1-D float32 tensor → (1, 49, 40) float16 ndarray."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.to(device)
    mel = _mel_xform(wav)         # (1, 40, 49)
    mel = _amp2db(mel)
    mel = mel.squeeze(0).T        # (49, 40)
    mel = (mel + 80.0) / 80.0
    return mel.unsqueeze(0).cpu().numpy().astype(np.float16)  # (1, 49, 40)


def _decode_member(tar: tarfile.TarFile, member: tarfile.TarInfo, device: str):
    """Extract one tar member and decode to (1, 49, 40) float16, or None on error."""
    suffix = Path(member.name).suffix.lower()
    raw = tar.extractfile(member)
    if raw is None:
        return None
    audio_bytes = raw.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        wav, sr = torchaudio.load(tmp_path)
        wav = wav.mean(0)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        n = wav.shape[-1]
        if n >= SAMPLE_RATE:
            wav = wav[:SAMPLE_RATE]
        else:
            wav = F.pad(wav, (0, SAMPLE_RATE - n))
        return _wav_to_logmel(wav, device)
    except Exception:
        return None
    finally:
        os.unlink(tmp_path)


# ── HuggingFace helpers ───────────────────────────────────────────────────────
def _list_shard_indices(lang: str, split: str) -> list[int]:
    """Return sorted list of shard indices by querying the HF filesystem (metadata only)."""
    fs = HfFileSystem()
    for prefix in [f"hf://datasets/{_HF_REPO_ID}", f"datasets/{_HF_REPO_ID}"]:
        pattern = f"{prefix}/data/opus/{lang}/{split}/audio/*.tar.gz"
        try:
            paths = fs.glob(pattern)
            if paths:
                return sorted(int(Path(p).stem) for p in paths if Path(p).stem.isdigit())
        except Exception:
            continue
    return []


def _download_shard(lang: str, shard_idx: int, shard_dir: Path, split: str) -> Path:
    """Download one shard to shard_dir and return its local path.
    hf_hub_download caches by content hash — repeat calls return immediately."""
    filename = f"data/opus/{lang}/{split}/audio/{shard_idx}.tar.gz"
    local = hf_hub_download(
        repo_id=_HF_REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(shard_dir),
    )
    return Path(local)


# ── Metadata helpers ──────────────────────────────────────────────────────────
def build_inventory(meta: dict, languages: list, top_k: int,
                    n_heldout: int, min_chars: int = 3) -> dict:
    """
    Build keyword inventory ranked by exact filename counts from metadata.

    Words are ranked by len(filenames[word]) — the actual number of audio
    clips in the dataset — not by wordcounts, which aggregate across all splits
    and can be misleading for low-resource languages.  No hard minimum is
    enforced here; fetch_lang caps each word at min(samples_per_word, available)
    so low-resource languages contribute whatever they have.
    """
    inv = {}
    for lang in languages:
        entry     = meta.get(lang, {})
        filenames = entry.get("filenames", {})

        ranked = sorted(
            [(w, len(fnames))
             for w, fnames in filenames.items()
             if len(w) >= min_chars],
            key=lambda x: -x[1],
        )

        training = [w for w, _ in ranked[:top_k]]
        heldout  = [w for w, _ in ranked[top_k:top_k + n_heldout]]
        inv[lang] = {
            "training": training,
            "heldout":  heldout,
            "counts":   {w: len(filenames.get(w, [])) for w in training + heldout},
        }
        train_min = min((len(filenames.get(w, [])) for w in training), default=0)
        print(f"  [{lang}] {entry.get('language', lang)}: "
              f"{len(training)} training (min available: {train_min}), "
              f"{len(heldout)} heldout")
    return inv


def _build_lookup(meta: dict, lang: str, target_words: set) -> dict:
    """
    Build {base_stem: word} lookup for all target word clips in this language.

    base_stem = 'common_voice_{lang}_{id}'  — the part after the word prefix
    in a shard member name '{word}_common_voice_{lang}_{id}.opus'.

    Lookup is O(1) per member and covers every clip in the dataset for these words.
    """
    filenames = meta.get(lang, {}).get("filenames", {})
    lookup = {}
    for word in target_words:
        for fname in filenames.get(word, []):
            base = Path(fname).stem   # e.g. 'common_voice_de_21910915'
            lookup[base] = word
    return lookup


def _cache_path(feats_dir: Path, lang: str, word: str, kind: str) -> Path:
    safe = word.replace("/", "_")
    return feats_dir / kind / lang / f"{safe}.npy"


# ── Per-language fetch ────────────────────────────────────────────────────────
def fetch_lang(lang: str, training_words: list, heldout_words: list,
               meta: dict, feats_dir: Path, shard_dir: Path,
               samples_per_word: int, shard_indices: list,
               device: str, split: str) -> None:

    for kind, target_words in [("train", set(training_words)),
                                ("heldout", set(heldout_words))]:
        if not target_words:
            continue

        (feats_dir / kind / lang).mkdir(parents=True, exist_ok=True)

        # Resume: skip words that already have a cache file
        to_collect = {w for w in target_words
                      if not _cache_path(feats_dir, lang, w, kind).exists()}
        if not to_collect:
            print(f"  [{lang}/{kind}] all {len(target_words)} words cached — skipping")
            continue

        # Per-word target: stop at min(samples_per_word, clips available in metadata).
        # This prevents fruitless shard scanning for low-resource words.
        filenames_meta = meta.get(lang, {}).get("filenames", {})
        word_targets = {
            w: min(samples_per_word, len(filenames_meta.get(w, [])))
            for w in to_collect
        }

        print(f"  [{lang}/{kind}] {len(to_collect)} words to collect")
        for w, tgt in sorted(word_targets.items(), key=lambda x: -x[1]):
            avail = len(filenames_meta.get(w, []))
            print(f"    {w}: target {tgt}  (metadata has {avail})")

        # Build O(1) filename→word lookup from metadata
        lookup = _build_lookup(meta, lang, to_collect)
        if not lookup:
            print(f"  [{lang}/{kind}] WARNING: no filenames in metadata for target words")

        buckets  = {w: [] for w in to_collect}
        n_filled = 0

        for shard_idx in shard_indices:
            if n_filled >= len(to_collect):
                break

            print(f"    [{lang}] downloading shard {shard_idx} ...", end=" ", flush=True)
            try:
                local = _download_shard(lang, shard_idx, shard_dir, split)
                print("reading ...", end=" ", flush=True)
                found_this_shard = 0

                with tarfile.open(local, "r:gz") as tar:
                    for member in tar:
                        if n_filled >= len(to_collect):
                            break
                        if not member.isfile():
                            continue
                        if Path(member.name).suffix.lower() not in _AUDIO_EXTS:
                            continue

                        stem = Path(member.name).stem
                        idx  = stem.find(_CV_MARKER)
                        if idx < 0:
                            continue
                        base = stem[idx + 1:]   # 'common_voice_{lang}_{id}'
                        word = lookup.get(base)
                        if word is None:
                            continue
                        if len(buckets[word]) >= word_targets[word]:
                            continue

                        spec = _decode_member(tar, member, device)
                        if spec is None:
                            continue

                        buckets[word].append(spec)
                        found_this_shard += 1
                        if len(buckets[word]) >= word_targets[word]:
                            n_filled += 1

                print(f"{found_this_shard} decoded  ({n_filled}/{len(to_collect)} filled)")

            except Exception as exc:
                print(f"ERROR: {exc}")
                continue

        # Save collected buckets — partial buckets (< samples_per_word) are saved too
        for word, frames in buckets.items():
            out = _cache_path(feats_dir, lang, word, kind)
            if frames:
                arr = np.stack(frames, axis=0)  # (N, 1, 49, 40) float16
                np.save(out, arr)
                print(f"    saved {lang}/{kind}/{word}: {arr.shape[0]} samples")
            else:
                print(f"    WARNING: {lang}/{kind}/{word}: 0 samples — word not found in any shard")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path,
                    help="Path to mswc-metadata.json")
    ap.add_argument("--root", type=Path, default=Path("./kws_cache"),
                    help="Output root — must match CACHE_DIR in the notebook")
    ap.add_argument("--langs", nargs="+",
                    default=["en", "de", "fr", "ca", "fa", "es", "it", "nl", "rw"])
    ap.add_argument("--top-k",    type=int, default=50,
                    help="Training keywords per language (default 50)")
    ap.add_argument("--n-heldout", type=int, default=20,
                    help="Heldout keywords per language (default 20)")
    ap.add_argument("--samples",  type=int, default=400,
                    help="Max samples per (lang, word) (default 400)")
    ap.add_argument("--split",    default="train",
                    help="MSWC split: train / dev / test (default train)")
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--debug", action="store_true",
                    help="Small run: 2 langs, top-5 words, 40 samples, 3000-clip scan")
    args = ap.parse_args()

    if args.debug:
        args.langs        = ["en", "de"]
        args.top_k        = 5
        args.n_heldout    = 2
        args.samples      = 40
        print("*** DEBUG MODE — small run, results not meaningful ***")

    print(f"Device  : {args.device}")
    print(f"Output  : {args.root}")
    print(f"Langs   : {args.langs}")
    print(f"top-k   : {args.top_k}  heldout: {args.n_heldout}  samples: {args.samples}")

    _init_transforms(args.device)

    # Shards cached in /tmp — session-ephemeral, no Drive quota used for raw audio
    shard_dir = Path(tempfile.gettempdir()) / "kws_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    feats_dir   = args.root / "feats"
    invent_path = args.root / "keyword_inventory.json"
    args.root.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print(f"\nLoading metadata from {args.metadata} ...")
    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"  {len(meta)} languages in metadata")

    # Build or load inventory
    if invent_path.exists():
        print(f"\nInventory already exists — loading from {invent_path}")
        inventory = json.loads(invent_path.read_text(encoding="utf-8"))
    else:
        print("\nBuilding inventory from metadata wordcounts ...")
        inventory = build_inventory(meta, args.langs, args.top_k, args.n_heldout)
        invent_path.write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Inventory saved → {invent_path}")

    # Fetch audio per language
    for lang in args.langs:
        if lang not in inventory:
            print(f"\n[{lang}] not in inventory — skipping")
            continue
        inv = inventory[lang]
        print(f"\n── [{lang}] ──────────────────────────────────────────────")
        shard_indices = _list_shard_indices(lang, args.split)
        if not shard_indices:
            print(f"  [{lang}] no shards found on HF — skipping")
            continue
        print(f"  {len(shard_indices)} shards available on HF")
        fetch_lang(
            lang=lang,
            training_words=inv["training"],
            heldout_words=inv["heldout"],
            meta=meta,
            feats_dir=feats_dir,
            shard_dir=shard_dir,
            samples_per_word=args.samples,
            shard_indices=shard_indices,
            device=args.device,
            split=args.split,
        )

    # Summary
    print("\n── Summary ──────────────────────────────────────────────────")
    total_train = total_held = 0
    for lang in args.langs:
        for kind, key in [("train", "training"), ("heldout", "heldout")]:
            words = inventory.get(lang, {}).get(key, [])
            cached = [w for w in words if _cache_path(feats_dir, lang, w, kind).exists()]
            n = sum(np.load(_cache_path(feats_dir, lang, w, kind)).shape[0]
                    for w in cached)
            tag = f"{len(cached)}/{len(words)} words"
            if kind == "train":
                total_train += n
            else:
                total_held += n
            print(f"  {lang}/{kind}: {tag}, {n} samples")

    print(f"\nTotal training samples : {total_train:,}")
    print(f"Total heldout samples  : {total_held:,}")
    print(f"\nAll done. Run multilingual_kws_v2.ipynb — it will load from Drive cache.")


if __name__ == "__main__":
    main()

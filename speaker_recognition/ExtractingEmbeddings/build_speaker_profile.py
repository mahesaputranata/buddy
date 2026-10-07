from ExtractingEmbeddings.types.ecapa_extractor import ECAPAEmbeddingExtractor
from pathlib import Path
import numpy as np

def build_speaker_profile(
    OUT_DIR_SILENCE_FILTERED: Path,
    OUT_DIR_EMBEDDINGS: Path,
    holdout_ratio,
):
    """
    Build the enrolled speaker's voice profile from processed chunks.

    - chunks_dir: directory containing silence-filtered .wav segments
    - out_dir: where to save centroid + embeddings
    - holdout_ratio: fraction of segments kept as eval (not used for centroid)
    """
    chunks_dir = OUT_DIR_SILENCE_FILTERED
    out_dir = OUT_DIR_EMBEDDINGS
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(chunks_dir.glob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No .wav files found in {chunks_dir}")

    print(f"Found {len(wav_files)} chunks in {chunks_dir}")

    # Simple shuffle to randomize which are used for enrollment/eval
    rng = np.random.default_rng(seed=42)
    wav_files = np.array(wav_files)
    rng.shuffle(wav_files)

    # Split into enrollment + eval
    n_total = len(wav_files)
    n_holdout = int(n_total * holdout_ratio)
    n_enroll = n_total - n_holdout

    enroll_files = wav_files[:n_enroll]
    eval_files = wav_files[n_enroll:]

    print(f"Using {n_enroll} for enrollment, {n_holdout} for eval.")

    extractor = ECAPAEmbeddingExtractor()

    enroll_embs = []
    for f in enroll_files:
        print(f"[ENROLL] Encoding {f.name}")
        emb = extractor.encode_file(f)
        enroll_embs.append(emb)

    enroll_embs = np.stack(enroll_embs, axis=0)  # (N_enroll, D)
    centroid = enroll_embs.mean(axis=0)          # (D,)

    # Optionally store eval embeddings too
    eval_embs = []
    for f in eval_files:
        print(f"[EVAL] Encoding {f.name}")
        emb = extractor.encode_file(f)
        eval_embs.append(emb)

    if eval_embs:
        eval_embs = np.stack(eval_embs, axis=0)  # (N_eval, D)
    else:
        eval_embs = np.empty((0, enroll_embs.shape[1]))

    # Save outputs
    np.save(out_dir / "speaker_centroid.npy", centroid)
    np.save(out_dir / "speaker_enroll_embeddings.npy", enroll_embs)
    np.save(out_dir / "speaker_eval_embeddings.npy", eval_embs)

    # Also store list of file names (for debugging, analysis)
    with open(out_dir / "speaker_enroll_files.txt", "w", encoding="utf-8") as f_out:
        for fpath in enroll_files:
            f_out.write(str(fpath) + "\n")

    with open(out_dir / "speaker_eval_files.txt", "w", encoding="utf-8") as f_out:
        for fpath in eval_files:
            f_out.write(str(fpath) + "\n")

    print(f"Saved centroid and embeddings to {out_dir}")

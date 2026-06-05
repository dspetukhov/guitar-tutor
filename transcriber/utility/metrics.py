import librosa
import numpy as np
from settings import EPS


def evaluate_harmony(chroma, T, predictions, eps=EPS):
    """
    Lower RMSE and higher cosine similarity mean
    triads better capture the harmonic content of the audio.

    To Do:
    - add more evaluation metrics (no ground-truth)
    """
    reconstructed = np.zeros_like(chroma)
    energy_coverage = []
    total_energy = chroma.sum()
    for p in range(predictions.size):
        indices = np.nonzero(T[predictions[p]])[0]
        reconstructed[indices, p] = 1
        # Energy coverage
        energy_coverage.append(
            chroma[indices, p].sum() / total_energy
        )

    rmse = np.sqrt(
        ((chroma - reconstructed) ** 2).mean()
    )
    chroma_norm = chroma / (np.linalg.norm(chroma) + eps)
    reconstructed_norm = reconstructed / (np.linalg.norm(reconstructed) + eps)
    # Chroma fidelity
    cosine_similarity = np.array([
        np.dot(chroma_norm[:, idx], reconstructed_norm[:, idx])
        for idx in range(predictions.size)
    ]).mean()
    l2_distance = np.linalg.norm(chroma_norm - reconstructed_norm)

    return {
        "RMSE": float(rmse),
        "Cosine similarity": float(cosine_similarity),
        "L2 Distance": float(l2_distance),
        "Energy coverage": float(np.mean(energy_coverage))
    }


def evaluate_melody(waveform, sample_rate, hop_length):
    """
    audio resynthesis + chromagram similarity for overall harmonic match,
    direct chroma/CQT similarity for framewise note activity,
    onset alignment and per-note salience for timing and note presence
    """
    # Extract fundamental frequency (f0) using pYIN
    hop_length = 512
    _, voiced_flag, _ = librosa.pyin(
        waveform, sr=sample_rate,
        hop_length=hop_length,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E7"),
        fill_na=np.nan
    )
    # Onset alignment to compare detected note onsets to energy spikes
    # in the original audio, using precision, recall, and F1 metrics.
    onset_env = librosa.onset.onset_strength(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length
    )
    # Extract reference onsets from waveform
    reference_onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=True,
        units="time"
    )

    frame_time = hop_length / sample_rate  # Duration of one frame in seconds

    onsets = []
    in_note = False
    for i, v in enumerate(voiced_flag):
        if v and not in_note:
            onsets.append(i * frame_time)
            in_note = True
        elif not v and in_note:
            in_note = False

    predicted_onsets = np.array(onsets)

    predicted_matched = set()
    reference_matched = set()

    for i, ref in enumerate(reference_onsets):
        diffs = np.abs(predicted_onsets - ref)
        idx = np.argmin(diffs)
        if diffs[idx] <= 0.05 and idx not in predicted_matched:
            predicted_matched.add(idx)
            reference_matched.add(i)
    matched = len(reference_matched)

    precision = matched / len(predicted_onsets)
    recall = matched / len(reference_onsets)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

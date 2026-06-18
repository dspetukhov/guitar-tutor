import librosa
import numpy as np

from .settings import EPS


def evaluate_harmony(chromagram, T, predictions, eps=EPS):
    """
    Reconstruction-based metrics:
        how well extracted triads can "rebuild" the original chroma features.
        - Lower RMSE and higher cosine similarity mean
        triads better capture the harmonic content of the audio.

    Signal-level proxy metrics:
        direct comparison of the extracted triad to the observed chroma vector for each segment.
        - L2 Distance
        - Energy coverage: fraction of total chroma energy explained by the triad's pitch classes
        - [Not implemented] Peak-picking accuracy: do the triad notes correspond to the top-3 peaks in the chroma vector?

    To Do:
    - add more evaluation metrics (no ground-truth)
    """
    reconstructed = np.zeros_like(chromagram)
    energy_coverage = []
    total_energy = chromagram.sum()
    for p in range(predictions.size):
        indices = np.nonzero(T[predictions[p]])[0]
        reconstructed[indices, p] = 1
        # Energy coverage
        energy_coverage.append(chromagram[indices, p].sum() / total_energy)

    rmse = np.sqrt(((chromagram - reconstructed) ** 2).mean())
    chromagram_norm = chromagram / (np.linalg.norm(chromagram) + eps)
    reconstructed_norm = reconstructed / (np.linalg.norm(reconstructed) + eps)
    # Chroma fidelity
    cosine_similarity = np.array(
        [
            np.dot(chromagram_norm[:, idx], reconstructed_norm[:, idx])
            for idx in range(predictions.size)
        ]
    ).mean()
    l2_distance = np.linalg.norm(chromagram_norm - reconstructed_norm)

    return {
        "RMSE": float(rmse),
        "Cosine similarity": float(cosine_similarity),
        "L2 Distance": float(l2_distance),
        "Energy coverage": float(np.mean(energy_coverage)),
    }


def evaluate_melody(waveform, sample_rate, params) -> dict:
    """
    audio resynthesis + chromagram similarity for overall harmonic match,
    direct chroma/CQT similarity for framewise note activity,
    onset alignment and per-note salience for timing and note presence

    To Do:
        - add more parameters to PyIN (+onset) and cqt/cqt_frequencies
    """
    # Extract fundamental frequency (f0) using pYIN
    # https://librosa.org/doc/0.11.0/generated/librosa.pyin.html
    f0, voiced_flag, _ = librosa.pyin(
        y=waveform,
        sr=sample_rate,
        # frame_length=params["frame_length"],
        hop_length=params["hop_length"],
        fmin=params["fmin"],
        fmax=params["fmax"],
        resolution=params["resolution"],
        switch_prob=params["switch_prob"],
        center=params["center"],
        pad_mode=params["pad_mode"],
        fill_na=np.nan,
    )
    # Onset alignment to compare detected note onsets to energy spikes
    # in the original audio, using precision, recall, and F1 metrics.
    onset_env = librosa.onset.onset_strength(
        y=waveform, sr=sample_rate, hop_length=params["hop_length"]
    )
    # Extract reference onsets from waveform
    reference_onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=params["hop_length"],
        backtrack=True,
        units="time",
    )

    frame_time = params["hop_length"] / sample_rate  # Duration of one frame in seconds

    onsets_times = []
    onsets_indices = []
    in_note = False
    for i, v in enumerate(voiced_flag):
        if v and not in_note:
            onsets_times.append(i * frame_time)
            onsets_indices.append(i)
            in_note = True
        elif not v and in_note:
            in_note = False

    predicted_onsets = np.array(onsets_times)

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
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Energy salience checks if each detected note is physically present
    # in the audio spectrum at its onset.
    # High salience means the note is physically present in the audio at that time
    # Low salience suggests a weak note

    cqt = np.abs(
        librosa.cqt(
            waveform,
            sr=sample_rate,
            hop_length=params["hop_length"],
            fmin=params["fmin"],
            bins_per_octave=params["bins_per_octave"],
            pad_mode=params["pad_mode"],
        )
    )
    freqs = librosa.cqt_frequencies(
        cqt.shape[0], fmin=params["fmin"], bins_per_octave=params["bins_per_octave"]
    )

    salience = []
    for i in onsets_indices:
        freq_bin = np.argmin(np.abs(freqs - f0[i]))
        pitch_energy = cqt[freq_bin, i]
        avg_energy = np.mean(cqt[:, i])
        salience.append(pitch_energy / (avg_energy + EPS))

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Salience": np.mean(salience),
    }

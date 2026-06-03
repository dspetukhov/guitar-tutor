import librosa
import numpy as np


def evaluate_harmony(chroma, T, predictions, tol=1e-9):
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
    chroma_norm = chroma / (np.linalg.norm(chroma) + tol)
    reconstructed_norm = reconstructed / (np.linalg.norm(reconstructed) + tol)
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


def evaluate_melody():
    pass

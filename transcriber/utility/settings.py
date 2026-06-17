import numpy as np

PITCHES = "C C# D D# E F F# G G# A A# B".split()
CHORDS = [f"{p}:{q}" for q in ("maj", "min") for p in PITCHES]
EPS = 1e-9


def triad_templates():
    # 12-dim pitch-class templates for triads
    NP = len(PITCHES)
    I = np.arange(NP)
    T = []  # Triads
    L = []  # Labels
    S = {
        "maj": [4, 7],
        "min": [3, 7],
        "dim": [3, 6],
        "aug": [4, 8],
        "sus2": [2, 7],
        "sus4": [5, 7],
    }
    for root in range(NP):
        for quality, v in S.items():
            c = np.isin(
                I, [(root) % NP, (root + v[0]) % NP, (root + v[1]) % NP]
            ).astype(float)
            c /= np.linalg.norm(c) + EPS
            T.append(c)
            L.append(f"{PITCHES[root]}:{quality}")
    T = np.stack(T, axis=0)
    assert T.shape == (48, 12)
    return T, L

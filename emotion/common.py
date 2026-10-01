"""Shared emotion helpers: GoEmotions labels, Ekman grouping (GoEmotions paper, ekman_mapping.json), metrics (no sklearn)."""
import numpy as np

GO = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment",
      "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
      "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
EKMAN = {"anger": ["anger", "annoyance", "disapproval"], "disgust": ["disgust"], "fear": ["fear", "nervousness"],
         "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
         "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
         "surprise": ["surprise", "realization", "confusion", "curiosity"]}
E6 = list(EKMAN)


def f1(y, p):
    tp = float((y & p).sum()); fp = float((~y & p).sum()); fn = float((y & ~p).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def best_thresholds(Y, S, grid=np.arange(0.02, 0.96, 0.02)):
    """per-column threshold maximising F1 on a dev set (Y bool (n,k), S scores (n,k))"""
    return np.array([max(grid, key=lambda t: f1(Y[:, j], S[:, j] >= t)) for j in range(Y.shape[1])])


def macro_f1(Y, S, thr):
    P = S >= thr
    return float(np.mean([f1(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))

import os
def hard_exit(code=0):
    """os._exit after killing our child processes: a vLLM engine core left alive keeps the job's log pipe open and the job
    never ends (2026-09-30: an eval job ran on for 30 min after writing its results)."""
    import signal, sys
    sys.stdout.flush(); me = str(os.getpid())
    for p in os.listdir("/proc"):
        try:
            if p.isdigit() and open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1] == me:
                os.kill(int(p), signal.SIGKILL)
        except Exception:
            pass
    os._exit(code)

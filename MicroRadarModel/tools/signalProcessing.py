import numpy as np
from tools import signalSimulation as sS
import math

def detection_bins(target_x, theta):
    t = np.linspace(0, 15e-3, 2000)
    T = t[1] - t[0]  # sampling interval
    N = t.size
    f = np.linspace(0, 1 / T, N)
    starting_bin = np.argwhere(f >= 5000 - 1)[0][0]
    index = [starting_bin + i * 5 for i in range(20)]

    Baseline = [sS.burst_signal(t_, reflection_=False) for t_ in t]
    fft_wo = np.fft.fft(Baseline)

    function_ = sS.create_IF_function(target_x, theta)
    RF_signal_w_refl = [function_(t_) for t_ in t]
    fft_w = np.fft.fft(RF_signal_w_refl)

    detection = - np.abs(fft_wo)[index] + np.abs(fft_w)[index]
    detection_bins = len(detection[detection > 0.2])
    detection_norms = np.linalg.norm(detection[detection > 0], ord=1)

    return detection_bins, detection_norms


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))


def length(v):
  return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
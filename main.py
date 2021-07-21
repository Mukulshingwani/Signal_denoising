from ops import deblur, denoise
from numpy import genfromtxt
import utils
from utils import misc_utils
import numpy as np


class Metrics:
    l2_dist: np.float64
    l1_dist: np.float64
    correlation: np.float64
    energy_diff: np.float64


class RecoveredSignal:
    signal: np.ndarray
    metrics: Metrics
    denoise_kernel_size: int    # size of denoising kernel that was used


def get_data():
    data = genfromtxt('data.csv', delimiter=',')

    x = data[1:, 0]
    y = data[1:, 1]
    h = np.array([1/16, 4/16, 6/16, 4/16, 1/16])

    # each entry `k_size` corresponds to the length of the kernel size that we
    # will use for denoising operation. Then we de-blur each of the obtained
    # denoised signal and recover the signal corresponding to each operation.
    # then we plot the signal which has the highest correlation with the
    # original signal `x`.
    kernel_sizes = [3, 5, 7, 9]

    return x, y, h, kernel_sizes


def calc_metrics(signal: np.ndarray, ref_signal: np.ndarray):
    metrics = Metrics()
    metrics.l2_dist = utils.l2_dist(signal, ref_signal)
    metrics.l1_dist = utils.l1_dist(signal, ref_signal)
    metrics.correlation = utils.correlation(signal, ref_signal)
    metrics.energy_diff = utils.energy_diff(signal, ref_signal)

    return metrics


if __name__ == '__main__':
    recovered_signals_method_a = list()
    recovered_signals_method_b = list()

    x, y, blur_kernel, denoise_kernel_sizes = get_data()

    # this loop helps in determining, the size of the kernel for which
    # the recovery of the signal system is most optimal
    for k_size in denoise_kernel_sizes:
        # Method A ------------------------------------------------------
        denoised_signal = denoise(y, k_size)
        recovered_signal = deblur(denoised_signal, blur_kernel)

        metrics = calc_metrics(recovered_signal, x)

        rec_signal = RecoveredSignal()
        rec_signal.signal = recovered_signal
        rec_signal.metrics = metrics
        rec_signal.denoise_kernel_size = k_size
        recovered_signals_method_a.append(rec_signal)

        # Method B -------------------------------------------------------
        deblurred_signal = deblur(y, blur_kernel)
        recovered_signal = denoise(deblurred_signal, k_size)

        metrics = calc_metrics(recovered_signal, x)

        rec_signal = RecoveredSignal()
        rec_signal.signal = recovered_signal
        rec_signal.metrics = metrics
        rec_signal.denoise_kernel_size = k_size
        recovered_signals_method_b.append(rec_signal)

    # Find the best signal in Method A and B
    best_signal_a = recovered_signals_method_a[0]
    for signal in recovered_signals_method_a:
        if signal.metrics.energy_diff < best_signal_a.metrics.energy_diff:
            best_signal_a = signal

    best_signal_b = recovered_signals_method_b[0]
    for signal in recovered_signals_method_b:
        if signal.metrics.energy_diff < best_signal_b.metrics.energy_diff:
            best_signal_b = signal
    print("Best Size of Denoising Kernel for singal x1[n] : ",
          best_signal_a.denoise_kernel_size)
    print("Best Size of Denoising Kernel for signal x2[n] : ",
          best_signal_b.denoise_kernel_size)
    print()
    print("Energy Difference of x1[n] from x[n] : ",
          best_signal_a.metrics.energy_diff)
    print("Energy Difference of x2[n] from x[n] : ",
          best_signal_b.metrics.energy_diff)
    print()
    print("Correlation of x1[n] with x[n] : ",
          best_signal_a.metrics.correlation)
    print("Correlation of x1[n] with x[n] : ",
          best_signal_b.metrics.correlation)
    misc_utils.graph_plot(x, best_signal_a.signal, 50, "x[n] vs x1[n]")
    misc_utils.graph_plot(x, best_signal_b.signal, 50, "x[n] vs x2[n]")
    misc_utils.graph_plot(best_signal_a.signal, best_signal_b.signal, 50,
                          "x1[n] v/s x2[n]")

import matplotlib.pyplot as plt
from tools import signalSimulation as sS
from parameters.microradarParameters import *
from scipy import optimize
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_evolution_of_target():
    t = np.linspace(0, 15e-3, 2000)
    for x in [5,4,3,2,1,0]:
        f = sS.create_IF_function(x, 0)
        RF_signal = [f(t_) for t_ in t]
        plt.plot(t, RF_signal, label="Target at {} m".format(x))
    plt.legend()
    plt.title("IF signals for micro_radars, isotropic propagation, gaussian amortissement")
    plt.show()


def plot_real_gain_1D():
    xs = [4*conversion_feet_meters, 6*conversion_feet_meters, 8*conversion_feet_meters, 10*conversion_feet_meters]
    conversion_factor = 0.55/np.power(10, 32/20)
    amplitudes = [conversion_factor*np.power(10, 36/20), conversion_factor*np.power(10, 32/20),
                  conversion_factor*np.power(10, 24/20), conversion_factor*np.power(10, 12/20)]

    def func(x, a=0, b=0):
        # a= 1
        return a*np.exp(b*x)

    popt, pcov = optimize.curve_fit(func, np.array(xs), np.array(amplitudes),
                                    p0=[0, 0])

    # b= 1
    plt.figure()
    plt.plot(xs, amplitudes, marker=".", label="Amplitudes from Gdb")
    plt.plot(xs, [a*target_x +b for target_x in xs], label=r'$A(d) = \alpha d + \beta$')
    plt.plot(np.linspace(1, 3.5, 100), [func(target_x, a=popt[0], b=popt[1]) for target_x in np.linspace(1, 3.5, 100)], label=r'$A(d) = \alpha d + \beta$')
    plt.xlabel(r'$d (m)$')
    plt.ylabel(r'$G(d,\theta=0)$')
    plt.title(r'Identification : $\alpha = -0.4562$ and $\beta = 1.397$ ')
    plt.show()

    plt.figure()
    xs = np.linspace(0,5,100)
    plt.plot(xs, [np.exp((-0.5 * (target_x) ** 2 / sigma_amortissement ** 2)) for target_x in xs])
    plt.show()


def detection_bins_test():
    t = np.linspace(0, 15e-3, 2000)
    T = t[1] - t[0]  # sampling interval
    N = t.size
    f = np.linspace(0, 1 / T, N)

    RF_signal_wo_refl = [sS.burst_signal(t_, reflection_=False) for t_ in t]
    fft_wo = np.fft.fft(RF_signal_wo_refl)
    targets_x = np.linspace(0, 3, 6)

    starting_bin = np.argwhere(f>=5000-1)[0][0]
    index2 = [starting_bin +i*5 for i in range(20)]
    detection_bins, detection_norms = [], []

    fig, ax = plt.subplots()
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.plot(f[index2], np.abs(fft_wo)[index2] * 1 / N + 0.5/N, marker="x", color = "black", label="BASELINE")
    for x in targets_x:
        f_ = sS.create_IF_function(x, 0)
        RF_signal_w_refl = [f_(t_) for t_ in t]
        fft_w = np.fft.fft(RF_signal_w_refl)
        plt.plot(f[index2], np.abs(fft_w)[index2] * 1 / N, marker="o", linestyle=":", label="Target at {} m".format(x))
        detection = - np.abs(fft_wo)[index2] + np.abs(fft_w)[index2]
        detection_bins.append(len(detection[detection>0.2]))
        detection_norms.append(np.linalg.norm(detection[detection>0], ord=1))
    plt.title("Detection bins for different targets")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(targets_x, detection_bins, marker="o")
    plt.title("Number of detection bins above the baseline")
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(targets_x, detection_norms, marker="o")
    plt.title("L 1 norm for the detection bins above the baseline")
    plt.show()


def get_amplitude_from_db(db):
    return np.power(10, db/20)


def function_normal(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def plot_radian_gain_1D():
    t4, a4, t6, a6 = get_data_from_documentation()
    # conversion_factor = 1 #0.55 / np.power(10, 32 / 20)
    # a4 = [conversion_factor * np.power(10, a / 20) for a in a4]
    # a6 = [conversion_factor * np.power(10, a / 20) for a in a6]

    popt, pcov = optimize.curve_fit(function_normal, np.array(t4), np.array(a4),
                                    p0=[71, 0, 1])
    popt6, pcov = optimize.curve_fit(function_normal, np.array(t6), np.array(a6),
                                    p0=[56, 0, 1])

    plt.figure()
    ts = np.linspace(t4[0], t4[-1], 100)
    plt.plot(t4, a4, marker=".", label=r'$G(4 feet, theta)$')
    plt.plot(ts, [function_normal(target_x, popt[0], popt[1], popt[2]) for target_x in ts], label=r'$A(4)*B(\theta)$')
    plt.plot(t6, a6, marker=".", label=r'$G(6 feet, theta)$')
    plt.plot(ts, [function_normal(target_x, popt6[0], popt6[1], popt6[2]) for target_x in ts], label=r'$A(4)*B(\theta)$')
    plt.xlabel(r'$\theta (rad)$')
    plt.ylabel(r'$G(d,\theta)$')
    plt.title(r'Identification : $\sigma = 0.54$ ')
    plt.legend()
    plt.show()

    def B(theta):
        return np.power(10,(1/20)*function_normal(theta, popt[0], popt[1], popt[2]))

    plt.figure()
    plt.plot(ts, [B(t) for t in ts])
    plt.show()




def plot_gain_3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the mesh in polar coordinates and compute corresponding Z.
    r = np.linspace(0, 4, 50)
    p = np.linspace(0, 2 * np.pi, 50)
    R, P = np.meshgrid(r, p)
    # Express the mesh in the cartesian system.
    X, Y = R * np.cos(P), R * np.sin(P)
    # radial_gain(R,P)
    Z = np.zeros(R.shape)
    for i in range(len(R)):
        r = R[i]
        p = P[i]
        for j in range(len(P)):
            Z[i][j] = sS.radial_gain(r[j], p[j])

    # Express the mesh in the cartesian system.
    # X, Y = R * np.cos(P), R * np.sin(P)

    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

    # Tweak the limits and add latex math labels.
    ax.set_zlim(0, 4)
    ax.set_xlabel(r'$x(\theta)$')
    ax.set_ylabel(r'$y(\theta)$')
    ax.set_zlabel(r'Amplitude IF Signal')

    plt.show()


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

def scatter_gain_detection_bins():
    # t = np.linspace(0, 15e-3, 2000)
    detections = []
    detection1 = []
    gains =[]
    for x in np.linspace(0, 5, 200):
        # f = sS.create_IF_function(x, 0)
        detections.append(detection_bins(x,0)[0])
        detection1.append(detection_bins(x, 0)[1])
        gains.append(sS.radial_gain(x,0))
        # RF_signal = [f(t_) for t_ in t]
        # plt.plot(t, RF_signal, label="Target at {} m".format(x))
    plt.scatter(gains, detections, marker="x", color="red")
    plt.grid(True)
    plt.xlabel("Radar Gain")
    plt.ylabel("Detection bins")
    plt.title("Detection bins and Radar Gain")
    plt.show()

    plt.scatter(gains, detection1)
    plt.legend()
    plt.grid(True)
    plt.xlabel("Radar Gain")
    plt.ylabel("L1 norm of Detection bins array")
    plt.title("L1 norm of Detection bins array and Radar Gain")
    plt.show()


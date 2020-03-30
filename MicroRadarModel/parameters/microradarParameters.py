import numpy as np
from scipy import optimize

frequency = 1/2e-4
pulse_width = 3e-3
conversion_feet_meters = 0.3048
If_c_round_trip = (6*0.3048)/ (11e-3 - 2.5e-3) #6 feet/difference in time
sigma_amortissement = np.sqrt((6*0.3048)**2/(-2*np.log(500/900)))
radian_sigma = 0.539
a, b = -0.4562, 1.397
alpha, beta = 3.32108927, -1.07250361


def get_amplitude_from_db(db):
    return np.power(10, db/20)


def get_coef_1d():
    xs = [4 * conversion_feet_meters, 6 * conversion_feet_meters, 8 * conversion_feet_meters,
          10 * conversion_feet_meters]
    conversion_factor = 0.55 / np.power(10, 32 / 20)
    amplitudes = [conversion_factor * np.power(10, 36 / 20), conversion_factor * np.power(10, 32 / 20),
                  conversion_factor * np.power(10, 24 / 20), conversion_factor * np.power(10, 12 / 20)]
    L = np.polyfit(xs, amplitudes, 1)
    return L[0], L[1]


def process_data(thetas, amplitudes):
    opposite_theta_s_4, n_amplitude = [], []
    for i, t in enumerate(thetas.copy()):
        opposite_theta_s_4.append(360 * 2 - t)
        n_amplitude.append(amplitudes[i])
    opposite_theta_s_4.reverse()
    n_amplitude.reverse()
    theta_s_4 = thetas + opposite_theta_s_4
    amplitudes_4 = amplitudes + n_amplitude
    theta_s_4 = [(360 - t) * ((2 * np.pi) / 360) for t in theta_s_4]
    return theta_s_4, amplitudes_4



def get_data_from_documentation():
    theta_s_4 = [270, 280, 295, 310, 322, 340, 355]
    amplitudes_4 = [get_amplitude_from_db(10), get_amplitude_from_db(15), get_amplitude_from_db(20),
                    get_amplitude_from_db(25), get_amplitude_from_db(30), get_amplitude_from_db(35),
                    get_amplitude_from_db(37)]

    theta_s_6 = [270, 285, 295, 310, 322, 340, 355]
    amplitudes_6 = [get_amplitude_from_db(5), get_amplitude_from_db(10), get_amplitude_from_db(15),
                    get_amplitude_from_db(20), get_amplitude_from_db(25), get_amplitude_from_db(30),
                    get_amplitude_from_db(32)]

    theta_s_4, amplitudes_4 = process_data(theta_s_4, amplitudes_4)
    theta_s_6, amplitudes_6 = process_data(theta_s_6, amplitudes_6)
    return theta_s_4, amplitudes_4, theta_s_6, amplitudes_6


def normal_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def plot_coef_radian_gain_1D():
    # xs = [4*conversion_feet_meters, 6*conversion_feet_meters, 8*conversion_feet_meters, 10*conversion_feet_meters]
    t4, a4, t6, a6 = get_data_from_documentation()
    popt4, pcov4 = optimize.curve_fit(normal_function, np.array(t4), np.array(a4), p0=[71, 0, 1])
    popt6, pcov6 = optimize.curve_fit(normal_function, np.array(t4), np.array(a4), p0=[71, 0, 1])
    return popt4


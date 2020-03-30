from parameters.microradarParameters import *
import math
import tools.signalProcessing as sP


def burst_signal(t, t0=0, amplitude=0, reflection_ = True):
    amortissement = np.exp(-(2*np.pi*t)/pulse_width)
    s = amortissement*np.sin(2*np.pi*frequency*t)
    if reflection_:
        s += reflection(t, t0, amplitude)
    return s


def reflection(t, t0, amplitude):
    amortissement = amplitude*np.exp(-(2*np.pi*abs(t-t0))/pulse_width)
    return amortissement*np.sin(2*np.pi*frequency*t)


def amortissement(target_x, mu_0 = 0):
    r = a*abs((target_x - mu_0)) + b
    return r

def process_trajectory(list_of_steps, v0=0.5, time_steps=0.2, feet=True):
    locations = dict()
    for l in list_of_steps:
        x_i, x_i1 = np.array(l[0]), np.array(l[1])
        t = 0
        temp_loc = x_i
        for i in range(len(l) - 1):
            x_i, x_i1 = np.array(l[i]), np.array(l[i + 1])
            direction = x_i1 - x_i
            direction = direction / np.linalg.norm(direction)
            last_distance = np.linalg.norm(temp_loc - x_i1)
            while np.linalg.norm(temp_loc - x_i1) <= last_distance:
                last_distance = np.linalg.norm(temp_loc - x_i1)
                t += time_steps
                temp_loc = temp_loc + v0*time_steps*direction
                if t not in locations:
                    locations[t] = [temp_loc]
                else:
                    locations[t].append(temp_loc)
    return locations


def distance_angle(x, y, S):
    x_0, y_0, theta_0 = S
    target_x = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
    theta = math.atan2(x_0-x, y-y_0) - theta_0
    return target_x, theta


def run_simulation(micro_radars, timed_trajectories):
    time = list(timed_trajectories.keys())
    detection_bins = dict([(k, [0.] * len(time)) for k in micro_radars])
    for sensor in micro_radars:
        for i,t in enumerate(time):
            pedestrian = timed_trajectories[t][0]
            target_x, theta = distance_angle(pedestrian[0], pedestrian[1], micro_radars[sensor])
            db, detection_norms = sP.detection_bins(target_x, theta)
            detection_bins[sensor][i] = db
    return detection_bins

def radial_gain(target_x, theta):
    if theta > np.pi:
        theta = theta - 2*np.pi
    elif theta < -np.pi:
        theta = theta + 2*np.pi
    r_gain = normal_function(theta, 1, 0 , radian_sigma)
    absiss_gain = decay_function(target_x, alpha, beta)
    return absiss_gain*r_gain


def create_IF_function(target_x, theta):
    transmission_time_round_trip = target_x/If_c_round_trip
    amplitude = radial_gain(target_x, theta)
    return lambda t : burst_signal(t, t0=transmission_time_round_trip, amplitude=amplitude, reflection_=True)


def normal_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def decay_function(x, a, b):
    return a*np.exp(b*x)
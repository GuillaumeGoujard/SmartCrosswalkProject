import numpy as np
from scipy import integrate
import tools.signalSimulation as sS
from parameters.microradarParameters import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
from tools.optimizationCallBack import MyCallBack

def D(Trajectories, S, l=(-5,20.7)):
    T = [(Trajectories[i][0], from_traj_to_ab(Trajectories[i][1])) for i in range(len(Trajectories))]
    gain_function = lambda y : gain_for_y_max(y, T, S)
    return integrate.quad(gain_function, l[0], l[1], epsabs=1e-3)[0]

def logD(Trajectories, S, l=(-5,20.7)):
    T = [(Trajectories[i][0], from_traj_to_ab(Trajectories[i][1])) for i in range(len(Trajectories))]
    integrals = np.zeros(len(Trajectories))
    for i, t in enumerate(T):
        # print(S)
        func = lambda y : log_gain_for_y_one_traj(y, t, S)
        integrals[i] = integrate.quad(func, l[0], l[1], epsabs=1e-4, limit=25)[0]
    return sum(integrals)

def log_gain_for_y_one_traj(y, t, S):
    S = S.reshape(-1, 3)
    result_Sensors = np.zeros(S.shape[0])
    omega, (a, b) = t
    x = a * y + b
    for i, theta in enumerate(S):
        target_x, theta_ = sS.distance_angle(x, y, theta)
        result_Sensors[i] = beta*target_x - (1/(2*radian_sigma**2))*(theta_)**2
    return omega*alpha_softmax(result_Sensors)

def compute_scores(micro_radars, Trajectories, dimensions, plot=False):
    sensors = np.array(list(micro_radars.values()))
    D2 = logD(Trajectories, sensors, l=(-5, dimensions[1]+5))
    D1 = D(Trajectories, sensors, l=(-5, dimensions[1]+5))
    mC = MyCallBack(sensors.reshape(-1), lambda x: D(Trajectories, x, l=(-5, dimensions[1]+5)),
                           gain, Trajectories=Trajectories)
    if plot:
        mC.plot_iterations_gain((dimensions[0], dimensions[1]), l=(-5, -5), precision=(50,100),
                            title="Arrangement")
    return D2, D1


def gain(x,y,S):
    S = S.reshape(-1, 3)
    result_Sensors = np.zeros(S.shape[0])
    for i, theta in enumerate(S):
        target_x, theta_ = sS.distance_angle(x, y, theta)
        result_Sensors[i] = sS.radial_gain(target_x, theta_)
    return alpha_softmax(result_Sensors)


def loggain(x,y,S):
    S = S.reshape(-1, 3)
    result_Sensors = np.zeros(S.shape[0])
    for i, theta in enumerate(S):
        target_x, theta_ = sS.distance_angle(x, y, theta)
        result_Sensors[i] = beta * target_x - (1 / 2 * (radian_sigma) ** 2) * (
            theta_) ** 2
    return alpha_softmax(result_Sensors)

def gain_for_y_max(y, T, S):
    S = S.reshape(-1, 3)
    result_Sensors = np.zeros(S.shape[0])
    max_gain = 0
    for t in T:
        omega, (a, b) = t
        x = a * y + b
        for i, theta in enumerate(S):
            target_x, theta_ = sS.distance_angle(x, y, theta)
            result_Sensors[i] = sS.radial_gain(target_x, theta_)
        max_gain += omega*max(result_Sensors)
    return max_gain


def alpha_softmax(x, alpha=5):
    return sum(x*np.exp(alpha*x)) / sum(np.exp(alpha*x))

def from_traj_to_ab(traj):
    A = np.array([[traj[0][1], 1 ], [traj[1][1], 1]])
    b = np.array([traj[0][0], traj[1][0]])
    ab = np.linalg.inv(A)@b
    return ab[0], ab[1]

def launch_grid_search_1Sensor(theta_0=0, Sensors=None, trajectories=None, dimensions=()):
    x = np.linspace(0, dimensions[0], 20)
    y = np.linspace(0, dimensions[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            S = np.array([X[i][j], Y[i][j], theta_0])
            Z[i][j] = logD(trajectories, S, l=dimensions[1])
    fig = plt.figure(figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    for s in Sensors:
        plt.plot([s[0]], [s[1]], marker="x", markersize=10)
    plt.colorbar()
    plt.axis('equal')
    plt.show()

def initial_conditions(k, dimensions):
    # dimensions = (3.048, 20.7264)
    x1 = dimensions[0] / 2
    x0 = [x1, 0., 0.]
    if k > 1:
        theta_step = np.pi / (k-1)
        sub_div_y = (dimensions[1] / (k-1))
    for i in range(1, k - 1):
        x0 = x0 + [x1, i * sub_div_y, i * theta_step]
    if k > 1:
        x0 = x0 + [x1, dimensions[1], -np.pi]
    return np.array(x0)


def BFGS(x0, Trajectories, maxiter, eps=1e1, method="BFGS", gtol=1e-1):
    f = lambda x: -logD(Trajectories, x, l=(-5, 25))
    mC = MyCallBack(x0, f, gain, Trajectories)
    res = minimize(f,
                   x0, method=method,
                   options={'disp': True,
                            'eps': eps,
                            'gtol': gtol,
                            'maxiter': maxiter},
                   callback=mC.callbackF,
                   )
    mC.res = res
    return mC

def convergence_plot(Trajectories):
    MCS = []
    for k in range(5, 10):
        MCS.append([k, optimize_BFGS(Trajectories, k, maxiter=30)])
    for l in MCS :
        k, mC = l
        y_s = [mC.f(r) for r in mC.results]
        plt.plot(range(len(mC.results)), y_s, label="Convergence results : {} sensors".format(k), marker="x")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Trajectories = [(0.5, [(-0.5, 25), (3.5, -5)]), (0.5, [(3, 25), (0, -5)])]
    dimensions = [3, 20.7]
    # x0 = np.array([dimensions[0]/2, dimensions[1]/2, 0])
    x0 = initial_conditions(1)
    mC = BFGS(x0, Trajectories, 20)
    mC.plot_iterations_gain((dimensions[0], dimensions[1]), l=(-5, -5), precision=(20, 50))

    # res = mC.results[-1]
    # x0 = np.array(list(res) + list(x0))
    # mC = BFGS(x0, Trajectories, 20)
    # mC.plot_iterations_gain((dimensions[0], dimensions[1]), l=(-5, -5), precision=(20, 50))
    #
    # res = mC.results[-1]
    # res2 = [res[0], res[1], -np.pi]
    # x0 = np.array(list(res) + list(res2))
    # mC = BFGS(x0, Trajectories, 20)
    # mC.plot_iterations_gain((dimensions[0], dimensions[1]), l=(-5, -5), precision=(20, 50))
    #
    # # log_gain_for_y_one_traj(Trajectories, )
    # logD(Trajectories, np.array([1.61483419,  1.89702926,  0.02479667, 1.8,10,0]), l=(-5, 20.7))
    # -D(Trajectories, mC.results[-1], l=(-5, 25))
    # -D(Trajectories, np.array([1.8,10,0]), l=(-5, 25))
    # maxiter = 20

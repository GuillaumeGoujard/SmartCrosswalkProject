import matplotlib.pyplot as plt
import numpy as np
from tools import scoringFunctions as scores

def show_set_up(dimensions, micro_radars, list_of_steps):
    fig = plt.figure(figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.axes()
    rectangle = plt.Rectangle((0, 0), dimensions[0], dimensions[1], fc='white', ec="red")
    ax = plt.gca()
    ax.axis('equal')
    ax.add_patch(rectangle)
    for s in micro_radars:
        plt.plot(micro_radars[s][0], micro_radars[s][1], marker="x",
                 linestyle="", markersize=15,
                 color="blue", label="Final position")
        # circle = plt.Circle((self.micro_radars[s][0], self.micro_radars[s][1]), 0.2)
        # ax.add_patch(circle)
    for s in micro_radars:
        plt.arrow(micro_radars[s][0], micro_radars[s][1],
                  np.cos(np.pi / 2 + micro_radars[s][2]), np.sin(np.pi / 2 + micro_radars[s][2]),
                  head_width=0.2,
                  width=0.1,
                  head_length=0.3, fc='blue', ec='blue')
    for i, list_of_steps in enumerate(list_of_steps):
        array = np.array(list_of_steps)
        ax.plot(array[:, 0], array[:, 1], label="Trajectory nb " + str(i))
    ax.set_xlim(-1, dimensions[0] + 1)
    ax.set_ylim(-5, dimensions[1] + 5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sensor arrangement")
    plt.show()

def convergence_plot(SC, Trajectories, n=3):
    MCS = []
    MCS_test = []
    for k in range(1, n):
        """
        test 1
        """
        if k > 1:
            print("{}, heuristic !".format(k))
            if k==2:
                x = MCS[-1][1].results[-1]
            else:
                x = MCS_test[-1][1].results[-1]

            if k % 4 == 2:
                x0 = np.array(list(x)+[0,0,-np.pi/4])
            elif k % 4 == 3:
                x0 = np.array(list(x) + [SC.dimensions[0], SC.dimensions[1], 3*np.pi / 4])
            elif k % 4 == 0:
                x0 = np.array(list(x) + [SC.dimensions[0], 0, np.pi / 4])
            elif k % 4 == 1:
                x0 = np.array(list(x) + [0, SC.dimensions[1], -3*np.pi / 4])
            MCS_test.append([k, scores.BFGS(x0, Trajectories, 40, eps=1e-1, gtol=1e-1, method="BFGS")])

        print("{}, simple !".format(k))
        x0 = scores.initial_conditions(k, SC.dimensions)
        MCS.append([k, scores.BFGS(x0, Trajectories, 40, eps=1e-1, gtol=1e-1, method="BFGS")])
    """
    Number of iterations per sensor and in function of init
    """
    #a)
    fig, ax = plt.subplots(figsize=(7,5), dpi=80, facecolor='w', edgecolor='k')
    index = np.arange(1,n)
    bar_width = 0.35
    opacity = 0.8
    n_iterations_normal_init = [m[1].Nfeval for m in MCS]
    n_iterations_past_init = [m[1].Nfeval for m in MCS_test]

    rects1 = plt.bar(index, n_iterations_normal_init, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Simple initialization')

    rects2 = plt.bar(index[1:] + bar_width, n_iterations_past_init, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Heuristic initialization')

    plt.xlabel('Number of sensors')
    plt.ylabel('Number of iterations')
    plt.title('Number of iterations in function of initialization and number of sensors')
    plt.xticks(index + bar_width/2, [str(k) for k in range(1,n)])
    plt.legend()

    plt.tight_layout()
    plt.show()

    #$b)
    fig, ax = plt.subplots(figsize=(7,5), dpi=80, facecolor='w', edgecolor='k')
    index = np.arange(1, n)
    bar_width = 0.35
    opacity = 0.8
    n_iterations_normal_init = [m[1].res.nfev for m in MCS]
    n_iterations_past_init = [m[1].res.nfev for m in MCS_test]

    rects1 = plt.bar(index, n_iterations_normal_init, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Simple initialization')

    rects2 = plt.bar(index[1:] + bar_width, n_iterations_past_init, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Heuristic initialization')

    plt.xlabel('Number of sensors')
    plt.ylabel('Number of function evaluation')
    plt.title('Number of function evaluation in function of initialization and number of sensors')
    plt.xticks(index + bar_width / 2, [str(k) for k in range(1, n)])
    plt.legend()

    plt.tight_layout()
    plt.show()


    """
    Store the results
    """
    for k, mC in MCS:
        fig = mC.plot_iterations_gain((SC.dimensions[0], SC.dimensions[1]), l=(-5, -5), precision=(20, 50),
                                      title="{} sensors, simple initialization".format(k))
        # fig.suptitle("test1", fontsize)
        fig.savefig("simple_{}_sensors.png".format(k))
    for k, mC in MCS_test:
        fig = mC.plot_iterations_gain((SC.dimensions[0], SC.dimensions[1]), l=(-5, -5), precision=(20, 50),
                                      title="{} sensors, heuritic initialization".format(k))
        fig.savefig("heuristic_{}_sensors.png".format(k))

    """
    Convergence of scores
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')
    logDv = [-scores.logD(Trajectories, m[1].results[-1], l=(-5, 25)) for m in MCS]
    logDv_test = [-scores.logD(Trajectories, m[1].results[-1], l=(-5, 25)) for m in MCS_test]
    ax.plot(index, logDv, marker="x", linestyle="dashed", label="Simple initialization")
    ax.plot(index[1:], logDv_test, marker="o", linestyle="dashed", label="Heuristic initialization")
    ax.axhline(y=168, color="red",label="-logD Danville arangement")
    plt.xlabel('Number of sensors')
    plt.ylabel('logD')
    plt.title('Final logD in function of initialization and number of sensors')
    plt.xticks(index, [str(k) for k in range(1, n)])
    plt.legend()
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
    Dv = [scores.D(Trajectories, m[1].results[-1], l=(-5, 25)) for m in MCS]
    Dv_test = [scores.D(Trajectories, m[1].results[-1], l=(-5, 25)) for m in MCS_test]
    ax.plot(index, Dv, marker="x", linestyle="dashed", label="Simple initialization")
    ax.plot(index[1:], Dv_test, marker="o", linestyle="dashed", label="Heuristic initialization")
    plt.xlabel('Number of sensors')
    plt.ylabel('D')
    ax.axhline(y=10.2, color="red", label="Detection score Danville arangement")
    plt.title('Final D in function of initialization and number of sensors')
    plt.xticks(index, [str(k) for k in range(1, n)])
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_grid_search(trajectories, dimensions, score_function, theta_0=0, detection_f="D"):
    x = np.linspace(0, dimensions[0], 20)
    y = np.linspace(0, dimensions[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            S = np.array([X[i][j], Y[i][j], theta_0])
            Z[i][j] = score_function(trajectories, S, l=dimensions)
    fig = plt.figure(figsize=(4, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.contourf(X, Y, Z, 20, cmap='RdGy')
    for i, t in enumerate(trajectories):
        omega, [a, b] = t
        plt.plot([a[0], b[0]], [a[1], b[1]], label="Proba: " + str(omega), linewidth=3)
    plt.title('{}(T;x,y,theta={})'.format(detection_f, theta_0))
    plt.colorbar()
    plt.axis('equal')
    plt.legend(loc=1)
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.xlim(-1, dimensions[0] + 1)
    plt.ylim(-1, dimensions[1] + 1)
    plt.show()


def plot_result_sensor(detection_bins, micro_radars, sensor, simulation_trajectories):
    time = list(simulation_trajectories.keys())
    x, y, theta = micro_radars[sensor]
    center = np.array(([x, y]))

    plt.figure()
    plt.subplot(211)
    plt.plot(time, detection_bins[sensor], marker=".", label="detection_bins from sensor " + str(sensor))
    plt.ylabel("Detection bins")
    plt.title("detection_bins from sensor " + str(sensor))

    plt.subplot(212)
    plt.plot(time, [np.linalg.norm(center - simulation_trajectories[t][0]) for t in time], color='tab:orange', linestyle='--')
    plt.ylabel("Distance to sensor " + str(sensor))
    plt.xlabel("Time (s)")
    plt.show()




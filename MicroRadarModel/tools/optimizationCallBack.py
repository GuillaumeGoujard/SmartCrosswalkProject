import matplotlib.pyplot as plt
import numpy as np

class MyCallBack:

    def __init__(self, x0, f, gain, Trajectories):
        self.results = [x0]
        self.Nfeval = 1
        self.gain = gain
        self.f = f
        self.Trajectories = Trajectories
        self.res = None

    def callbackF(self, Xi):
        # print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(self.Nfeval, Xi[0], Xi[1], Xi[2], self.f(Xi)))
        self.Nfeval += 1
        self.results.append(Xi)


    def plot_iterations_grid_search(self, dimensions, l= (0,0), theta_0=0, precision=(10,25)):
        converted_results = [r.reshape(r.shape[0] // 3, 3) for r in self.results]
        x = np.linspace(l[0], dimensions[0]+5, precision[0])
        y = np.linspace(l[1], dimensions[1]+5, precision[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                S = np.array([X[i][j], Y[i][j], theta_0])
                Z[i][j] = self.f(S)
                # Z[i][j] = self.gain(X[i][j], Y[i][j], self.results[-1])
        fig = plt.figure(figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.axes()
        rectangle = plt.Rectangle((0, 0), dimensions[0], dimensions[1], fc='white', ec="red", fill=False)
        ax = plt.gca()
        ax.add_patch(rectangle)
        plt.contourf(X, Y, Z, 20, cmap='binary')
        plt.plot([s[:, 0] for s in converted_results][:-1], [s[:, 1] for s in converted_results][:-1], marker="x",
                 linestyle="dashed", markersize=5,
                 color="blue")
        plt.plot(converted_results[-1][:, 0], converted_results[-1][:, 1], marker="x",
                 linestyle="", markersize=15,
                 color="red", label="Final position")
        for s in converted_results:
            for i in range(converted_results[0].shape[0]):
                plt.arrow(s[i, 0], s[i, 1], np.cos(np.pi / 2 + s[i, 2]), np.sin(np.pi / 2 + s[i, 2]), head_width=0.05,
                          width=0.002,
                          head_length=0.005, fc='b', ec='b')
        s = converted_results[-1]
        for i in range(converted_results[0].shape[0]):
            plt.arrow(s[i, 0], s[i, 1], np.cos(np.pi / 2 + s[i, 2]), np.sin(np.pi / 2 + s[i, 2]), head_width=0.2,
                      width=0.1,
                      head_length=0.3, fc='red', ec='red')
        for i, list_of_steps in enumerate(self.Trajectories):
            array = np.array(list_of_steps[1])
            ax.plot(array[:,0], array[:,1], label="Trajectory nb " + str(i))
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel("x")
        plt.legend()
        plt.ylabel("y")
        plt.title("Sensor arrangement k = " + str(len(converted_results)))
        plt.show()


    def plot_iterations_gain(self, dimensions, l= (0,0), theta_0=0, precision=(10,25), title=""):
        converted_results = [r.reshape(r.shape[0] // 3, 3) for r in self.results]
        x = np.linspace(l[0], dimensions[0]+5, precision[0])
        y = np.linspace(l[1], dimensions[1]+5, precision[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # S = np.array([X[i][j], Y[i][j], theta_0])
                # Z[i][j] = self.f(S)
                Z[i][j] = self.gain(X[i][j], Y[i][j], self.results[-1])
        fig = plt.figure(figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.axes()
        rectangle = plt.Rectangle((0, 0), dimensions[0], dimensions[1], fc='white', ec="red", fill=False)
        ax = plt.gca()
        ax.add_patch(rectangle)
        plt.contourf(X, Y, Z, 20, cmap='binary')
        plt.plot([s[:, 0] for s in converted_results][:-1], [s[:, 1] for s in converted_results][:-1], marker="x",
                 linestyle="dashed", markersize=5,
                 color="blue")
        plt.plot(converted_results[-1][:, 0], converted_results[-1][:, 1], marker="x",
                 linestyle="", markersize=15,
                 color="red", label="Final position")
        for i, list_of_steps in enumerate(self.Trajectories):
            array = np.array(list_of_steps[1])
            ax.plot(array[:,0], array[:,1], label="Trajectory nb " + str(i))
        for s in converted_results:
            for i in range(converted_results[0].shape[0]):
                plt.arrow(s[i, 0], s[i, 1], np.cos(np.pi / 2 + s[i, 2]), np.sin(np.pi / 2 + s[i, 2]), head_width=0.05,
                          width=0.002,
                          head_length=0.005, fc='b', ec='b')
        s = converted_results[-1]
        for i in range(converted_results[0].shape[0]):
            plt.arrow(s[i, 0], s[i, 1], np.cos(np.pi / 2 + s[i, 2]), np.sin(np.pi / 2 + s[i, 2]), head_width=0.2,
                      width=0.1,
                      head_length=0.3, fc='red', ec='red')
        plt.colorbar()
        plt.axis('equal')
        plt.xlabel("x")
        plt.legend()
        plt.ylabel("y")
        plt.title(title)
        plt.show()
        return fig

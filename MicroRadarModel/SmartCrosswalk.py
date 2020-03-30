# import matplotlib.pyplot as plt
from parameters.microradarParameters import *
# from tools import signalProcessing as sP
from tools import signalSimulation as sS
from tools import scoringFunctions as scores
# from mpl_toolkits.mplot3d import Axes3D
from tools import plottingTools as pt

class SmartCrosswalk:
    """
    Class useful :
        @ to simulate output (detection bins in function of time) of micro-radars at a crosswalk
        according to pedestrian trajectories
        @ to find the best arrangement of k micro-radarsa at a crosswalk according to a set of frequent pedestrian
        trajectories

    Instance of the class:
        - micro_radars : dictionnary of micro_radar
        - cpt_sensors : number of micro_radars
        - dimensions : list of 2 dimensions (L,l)
        - trajectories : list of trajectories [w_i, ((x_i1, y_i1) , (x_i2 ,y_i2))]  where w_i is a proba,
        x_i y_i are points of the trajectory
        - list_of_steps : list of steps taken by a pedestrian
        - detection_bins : output of a simulation

    """

    def __init__(self, sensors, dimensions, feet=True):
        """
        :param sensors: list of microradars [[x_0, y_0, theta_0], [... ] , ...]
        :param dimensions: list of dimensions of crosswalk ex : [3, 20]
        :param feet: if the dimensions are in feet or meters
        """
        if feet:
            conversion_rate = conversion_feet_meters
        else:
            conversion_rate = 1
        self.micro_radars = {}
        self.cpt_sensors = 0
        self.dimensions = [dimensions[0] * conversion_rate, dimensions[1] * conversion_rate]
        for s in sensors:
            self.add_micro_radar(s[0] * conversion_rate, s[1] * conversion_rate, s[2])
        self.trajectories = []
        self.list_of_steps = []
        self.detection_bins = None


    def add_micro_radar(self, x, y, theta):
        """
        add a micro-radar of location x, y and angle theta relative to the y axis
        :param x:
        :param y:
        :param theta: in (-np.pi, np.pi)
        :return:
        """
        self.micro_radars[self.cpt_sensors] = [x, y, theta]
        self.cpt_sensors += 1

    def show_set_up(self):
        """
        plot a graph of the crosswalk, with the sensors represented with a blue cross and an arrow for the angle
        also represents the trajectories of the crosswalk
        :return:
        """
        pt.show_set_up(self.dimensions, self.micro_radars, self.list_of_steps)

    def set_up_trajectories(self, trajectories):
        """
        add trajectories to the object and transcribe them to list of steps
        :param trajectories: list of trajectories [w_i, ((x_i1, y_i1) , (x_i2 ,y_i2))]  where w_i is a proba,
        ((x_i1, y_i1) , (x_i2 ,y_i2)) are two points of the trajectory
        :return:
        """
        self.trajectories = trajectories
        self.list_of_steps = [c[1] for c in trajectories]

    def add_simulation_trajectory(self, list_of_steps, feet=False):
        """
        add a list_of_steps
        :param list_of_steps:  list of trajectories [(x_i1, y_i1) , (x_i2 ,y_i2), (x_i3, y_i3) , ... ] are  points
        of the trajectory
        :param feet: if the dimensions are in feet or meters
        :return:
        """
        if feet:
            conversion_rate = conversion_feet_meters
            for i in range(len(list_of_steps)):
                x, y = list_of_steps[i]
                list_of_steps[i] = [conversion_rate * x, conversion_rate * y]
        self.list_of_steps.append(list_of_steps)
        return True

    def launch_grid_search_1Sensor(self, theta_0=0, trajectories=None, detection_f="D"):
        """
        plot the crosswalk with for each x,y the gain of a single sensor located in (x,y) and aligned with the y axis
        :param theta_0: angle of the sensors with respect to the y axis
        :param trajectories:  list of trajectories [w_i, ((x_i1, y_i1) , (x_i2 ,y_i2))]  where w_i is a proba,
        ((x_i1, y_i1) , (x_i2 ,y_i2)) are two points of the trajectory
        :param detection_f: either "D" or "logD" (see the report)
        :return:
        """
        score_function = scores.D if detection_f == "D" else scores.logD
        if trajectories is None:
            trajectories = self.trajectories
        pt.plot_grid_search(trajectories, self.dimensions, score_function, theta_0=theta_0, detection_f=detection_f)


    def run_simulation(self, v0=5, time_steps=0.2):
        """
        will store in self.detection_bins for each sensor, (time, detection bins)
        :param v0: speed of pedestrian in m/s
        :param time_steps: in s
        :return:
        """
        self.timed_trajectories = sS.process_trajectory(self.list_of_steps, v0=v0, time_steps=time_steps)
        self.detection_bins = sS.run_simulation(self.micro_radars, self.timed_trajectories)
        return self.detection_bins

    def plot_sensors_result(self, sensor):
        """
        plot the detection bins in function of time for sensors with id 'sensor'
        :param sensor: id of the sensor (from 0 to cpt_sensor)
        :return:
        """
        pt.plot_result_sensor(self.detection_bins, self.micro_radars, sensor, self.timed_trajectories)

    def optimize_sensors(self, trajectories=None, initial_conditions=None, maxiter=20, show=False, save=True):
        """
        According to a set of trajectories and initial conditions, find the best arangement to maximize :
        logD(S;T) with S the sensors
        :param trajectories: list of trajectories [w_i, ((x_i1, y_i1) , (x_i2 ,y_i2))]  where w_i is a proba,
        ((x_i1, y_i1) , (x_i2 ,y_i2)) are two points of the trajectory
        :param initial_conditions: numpy array of unravelled sensors, for example:
         np.array([x_0, y_0, theta_0, x_1, y_1, theta_1])
        :param maxiter: maximum iteration of the BFGS algorithm
        :param show: if you want to show the iterations on the crosswalk
        :param save: if you want to save the results to self.micro_radars
        :return:
        """
        if trajectories is None:
            trajectories = self.trajectories
        if initial_conditions is None:
            x0 = np.array(list(self.micro_radars.values())).reshape(-1)
        else:
            x0 = initial_conditions
        res = scores.BFGS(x0, trajectories, maxiter)
        if show:
            res.plot_iterations_gain((self.dimensions[0], self.dimensions[1]), l=(-5, -5), precision=(20, 50))
        sensors = dict()
        array_of_sensors = res.results[-1].reshape(-1, 3)
        for j in range(len(array_of_sensors)):
            sensors[j] = list(array_of_sensors[j])
        if save:
            self.micro_radars = sensors
        return res, sensors


    def set_up_k_sensors(self, trajectories=None, k=1, maxiter=20, show=True, save=True):
        """
        find the best arrangement for k sensors with initial conditions specified in the report
        :param trajectories:
        :param k:
        :param maxiter:
        :param show:
        :param save:
        :return:
        """
        x0 = scores.initial_conditions(k)
        return self.optimize_sensors(trajectories=trajectories, initial_conditions=x0, maxiter=maxiter, show=show, save=save)


    def compute_scores(self, Trajectories=None, plot=False):
        """
        Returns two reals : D(T,S) and LogD(T,S) (See report)
        :param Trajectories:
        :param plot:
        :return:
        """
        Trajectories = self.trajectories if Trajectories is None else Trajectories
        return scores.compute_scores(self.micro_radars, Trajectories, self.dimensions, plot=plot)


def createSensysCrosswalk():
    sensors = [
        (4.25, -10.50, -np.pi/4),
        (2.33, 6.33, 0),
        (7.08, 8.25, 0),
        (3.75, 17.33, np.pi/2),
        (5.75, 18.25, -np.pi/2),
        (5, 23.92, 0),
        (5, 23.92, -np.pi),
        (2.33, 44.58, -np.pi),
        (7.25, 46.58, -np.pi),
        (2.42, 57, -np.pi),
        (7.33, 59.25, -np.pi),
        (6.58, 74.76, -3*np.pi/4),
    ]
    dimensions = (10, 68)  # 10 feet wide and 68 feet long Crosswalk
    SC = SmartCrosswalk(sensors, dimensions, feet=True)  # Creation of the object SC
    return SC


if __name__ == '__main__':
    """
    Creation of a simple crosswalk
    """
    sensors = [(6.33, 2.33, 0)] #Microradar at (6.33, 2.33) and oriented along the vertical axis
    dimensions = (10, 68) #10 feet wide and 68 feet long Crosswalk
    SC = SmartCrosswalk(sensors, dimensions, feet=True) # Creation of the object SC

    """
    Creation of a pedestrian list of steps on the crosswalk and simulation of the output
    """
    lists_of_steps = [(3, 21), (2, 10), (2, -1)] #Successions of steps of a pedestrian
    SC.add_simulation_trajectory(lists_of_steps, feet=False)
    SC.show_set_up()  # Show the crosswalk, sensors and trajectories
    SC.run_simulation(v0=2, time_steps=0.2)
    SC.plot_sensors_result(0)  # Plot the detection bins in function of time for sensor 0

    """
    Let's optimize the location of the single sensor on the crosswalk in function of two trajectories of same probability
    """
    Trajectories = [(0.5, [(-0.5, 25), (3.5, -5)]), (0.5, [(3, 25), (0, -5)])]
    SC.launch_grid_search_1Sensor(trajectories=Trajectories, detection_f="D") #to get a sense of the Detection function for a single microradar
    SC.optimize_sensors(trajectories=Trajectories, initial_conditions=None, maxiter=20, show=True, save=True)
    logD1, D1 = scores.compute_scores(SC.micro_radars, Trajectories, SC.dimensions, plot=True)

    """
    Let's load Sensys arrangement
    """
    SC = createSensysCrosswalk()
    # Trajectories = [(0.5, [(-0.5, 25), (3.5, -5)]), (0.5, [(3, 25), (0, -5)])]
    Trajectories = [(0.5, [(-0.5, 25), (3.7, -5)]), (0.5, [(3.7, 25), (-0.5, -5)])]
    SC.set_up_trajectories(Trajectories)
    SC.show_set_up()
    logD2, D2 = SC.compute_scores(plot=True)
    SC.optimize_sensors(trajectories=Trajectories, initial_conditions=None, maxiter=20, show=True, save=True)
    logD3, D3 = SC.compute_scores(plot=True)

    # SC.optimize_sensors(trajectories=Trajectories, initial_conditions=scores.initial_conditions(9, SC.dimensions),
    #                     maxiter=50, show=True, save=True)
#
    # SC.set_up_k_sensors(trajectories=Trajectories, k=11, maxiter=20, show=True, save=True)
#     # scores.compute_scores(SC.micro_radars, Trajectories, SC.dimensions, plot=True)

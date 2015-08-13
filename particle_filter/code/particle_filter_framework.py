from math import *
import numpy as np
import random
import argparse
import scipy.stats
from collections import defaultdict
import math
import matplotlib.pyplot as plt

def read_world_data(filename):
    world_dict = defaultdict()
    f = open(filename)
    for line in f:
        line_s  = line.split('\n')
        line_spl  = line_s[0].split(' ')
        world_dict[float(line_spl[0])] = [float(line_spl[1]),float(line_spl[2])]
    return world_dict

def read_sensor_data(filename):
    data_dict = defaultdict()
    id_arr =[]
    range_arr=[]
    bearing_arr=[]
    first_time = True
    timestamp=-1
    f = open(filename)
    for line in f:
        line_s = line.split('\n') # remove the new line character
        line_spl = line_s[0].split(' ') # split the line

        if (line_spl[0]=='ODOMETRY'):
            data_dict[timestamp,'odom'] = {'r1':float(line_spl[1]),'t':float(line_spl[2]),'r2':float(line_spl[3])}
            if (first_time == True):
                first_time= False
            else:
                data_dict[timestamp,'sensor'] = {'id':id_arr,'range':range_arr,'bearing':bearing_arr}
                id_arr=[]
                range_arr = []
                bearing_arr = []
            timestamp = timestamp+1

        if(line_spl[0]=='SENSOR'):
            id_arr.append(int(line_spl[1]))
            range_arr.append(float(line_spl[2]))
            bearing_arr.append(float(line_spl[3]))
    data_dict[timestamp-1,'sensor'] = {'id':id_arr,'range':range_arr,'bearing':bearing_arr}
    return data_dict

class robot():
    # --------
    # init:
    #    creates robot and initializes location/orientation
    #

    def __init__(self):
        self.x = random.random()  # initial x position
        self.y = random.random() # initial y position
        self.orientation = random.uniform(-math.pi,math.pi) # initial orientation
        self.weights = 1.0

    # --------
    # set:
    #    sets a robot coordinate
    #
    def set(self, new_x, new_y, new_orientation):
        #if new_orientation < -math.pi or new_orientation >= math.pi:
        #    raise ValueError, 'Orientation must be in [-pi..pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    # --------
    # measurement_prob:
    #    computes the probability of a measurement
    # This function takes in the ids and ranges of the sensor data for each measurement update
    # World data is passed as the dictionary. The positions of the landmarks in the world
    # can be accessed as wrld_dict[ids[i]][0],wrld_dict[ids[i]][1]
    # where i is looped over the number of measurements for the current timestamp
    def measurement_prob_range(self, ids, ranges, wrld_dict):
        sigma_measurement = 0.2
        error = 1.0
        for i in range(len(ids)):
            # particle distance
            x_delta = wrld_dict[ids[i]][0] - self.x
            y_delta = wrld_dict[ids[i]][1] - self.y
            landmark_vector = [x_delta, y_delta]
            dist = np.linalg.norm(landmark_vector)
            unit_landmark_vector = landmark_vector / dist

            #pdf
            dist_pdf = scipy.stats.norm(ranges[i], sigma_measurement).pdf(dist)
            error = error * dist_pdf
        return error

    # --------
    # move_odom:
    # Takes in Odometry data
    def mov_odom(self,odom,noise):
        # Calculate the distance and Guassian noise
        dist  = odom['t']
        # calculate delta rotation 1 and delta rotation 2
        delta_rot1 = odom['r1']
        delta_rot2 = odom['r2']

        # noise sigma for delta_rot1
        sigma_delta_rot1 = noise[0]*abs(delta_rot1) + noise[1]*abs(dist)
        delta_rot1_noisy = delta_rot1 + random.gauss(0,sigma_delta_rot1)

        # noise sigma for translation
        sigma_translation = noise[2]*abs(dist) + noise[3]*abs(delta_rot1+delta_rot2)
        translation_noisy = dist + random.gauss(0,sigma_translation)

        # noise sigma for delta_rot2
        sigma_delta_rot2 = noise[0]*abs(delta_rot2) + noise[1]*abs(dist)
        delta_rot2_noisy = delta_rot2 + random.gauss(0,sigma_delta_rot2)

        # Estimate of the new position of the robot
        x_new = self.x  + translation_noisy * cos(self.orientation+delta_rot1_noisy)
        y_new = self.y  + translation_noisy * sin(self.orientation+delta_rot1_noisy)
        theta_new = self.orientation + delta_rot1_noisy + delta_rot2_noisy

        result = robot()
        result.set(x_new, y_new, theta_new)
        return result

    # Set the weight of the particles
    def set_weights(self, weight):
        #noise parameters
        self.weights = float(weight)

# --------
#
# extract position from a particle set
#
def get_mean_position(p):
    x = 0.0 # sum total of all x positions
    y = 0.0 # sum total of all y positions
    x_pos = []
    y_pos = []
    angle_x = 0.0
    angle_y = 0.0

    ''' Write the code here to calculate the mean position and orientation of all the particles'''
    for pt in p:
        x_pos.append(pt.x)
        y_pos.append(pt.y)
        x = x + pt.x
        y = y + pt.y
        angle_x = angle_x + cos(pt.orientation)
        angle_y = angle_y + sin(pt.orientation)

    avg_angle_x = angle_x / len(p)
    avg_angle_y = angle_y / len(p)
    avg_orient = math.atan2(avg_angle_y, avg_angle_x)

    # Particles are plotted here
    ''' Lists x and y contains the x and y positions of all the particles which can be accessed from p[i].x , p[i].y
		avg_orient is the average orientation of all the particles'''

    plt.plot(x_pos,y_pos,'r.')
    quiver_len = 3.0
    plt.quiver(x / len(p), y / len(p), quiver_len * np.cos(avg_orient), quiver_len * np.sin(avg_orient),angles='xy',scale_units='xy')

    plt.plot(lx,ly,'bo',markersize=10)
    plt.axis([-2, 15, 0, 15])
    plt.draw()
    plt.clf()
    return [x / len(p), y / len(p), avg_orient ]

''' Resampling the Particles
    Complete this Stub to resample the weights of the particles  '''
def resample_particles(weights, particles):
        # Sum the weights
        Sum = 0.0
        for w in weights:
            Sum = Sum + w

        # Normalize the weights
        norm_weights = []
        for w in weights:
            norm_weights.append((w/Sum))

        # calculate the PDF of the weights
        pdf=[]
        Norm_Sum = 0.0
        for k in range(len(particles)):
            Norm_Sum = Norm_Sum + norm_weights[k]
            pdf.append(Norm_Sum)

        # Calculate the step for random sampling, it depends on number of particles
        step = 1 / len(particles)

        # Sample a value in between [0,step) uniformly
        seed = random.uniform(0,step)
        #print 'Seed is %0.15s and step is %0.15s' %(seed, step)

        # resample the particles based on the seed, step and calculated pdf
        p_sampled = []
        index = 0
        for h in range(len(particles)):
            val = seed + step * h
            index = sample_index(val, index, pdf)
            p_sampled.append(particles[index])
        return p_sampled

def sample_index(val, index, pdf):
        while pdf[index] < val:
            index+1
        return index

def particle_filter(data_dict,world_dict, N): #
    # --------
    #
    # Make a list of particles
    #
    p = []
    for i in range(N):
        r = robot()
        p.append(r)

    # --------
    #
    # Update particles
    # sensor.dat file contains odometry + sensor updates -> length/2 entries
    #
    for t in range(len(data_dict)/2):
        # Step 1: motion update (prediction)
        p2 = []
        for i in range(N):
            p2.append(p[i].mov_odom(data_dict[t,'odom'],noise_param))
        p = p2

        # Step 2: measurement update
        w = []
        for i in range(N):
            w.append(p[i].measurement_prob_range(data_dict[t,'sensor']['id'], data_dict[t,'sensor']['range'], world_dict))

		# Step 3: resample particles to calculate new belief
        p = resample_particles(w,p)
    return get_mean_position(p)


## Main loop Starts here
parser = argparse.ArgumentParser()
parser.add_argument('sensor_data', type=str, help='Sensor Data')
parser.add_argument('world_data', type=str, help='World Data')
parser.add_argument('N', type=int, help='Number of particles')

args = parser.parse_args()
N = args.N
noise_param = [0.1, 0.1 ,0.05 ,0.05]

plt.axis([0, 15, 0, 15])
plt.ion()
plt.show()
data_dict = read_sensor_data(args.sensor_data)
world_data = read_world_data(args.world_data)

lx=[]
ly=[]

for i in range (len(world_data)):
    lx.append(world_data[i+1][0])
    ly.append(world_data[i+1][1])

estimated_position = particle_filter(data_dict,world_data,N)
print "Final Pose Estimate: " + str(estimated_position)

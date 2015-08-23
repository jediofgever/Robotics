from math import *
import numpy as np
import random
import argparse
import scipy.stats as sts
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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

''' Normalize the angle between -pi and pi'''
def normalize_angle(angle):
    if(len(angle)>1):
        angle=angle[0]

    while(angle > math.pi):
        angle = angle - 2*math.pi

    while (angle < - math.pi):
        angle = angle + 2*math.pi
    return angle

class robot():
    # --------
    # init:
    #    creates robot and initializes location/orientation
    #
    def __init__(self):
        self.pose = np.zeros((3,1))
        self.weight = 1.0
        self.history = np.empty(self.pose.shape)
        self.landmarks = np.empty((len(world_data)+1), dtype=object)
        for i in range(len(world_data)):
            self.landmarks[i+1]= [False,np.zeros((2,1)),np.zeros((2,2))] # observed, mu , sigma

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

    '''compute the expected measurement for a landmark
        and the Jacobian with respect to the landmark
    '''
    def measurement_model(self, landmark_id):
        # A 2D vector for the position (x,y) of the observed landmark
        landmarkPos = self.landmarks[landmark_id][1];

        # use the current state of the particle to predict the measurment
        landmarkX = landmarkPos[0];
        landmarkY = landmarkPos[1];

        delta = [float(landmarkX - self.pose[0]), float(landmarkY - self.pose[1])]
        expectedRange = np.sqrt(np.dot(delta, delta))
        expectedBearing = normalize_angle(atan2(delta[1], delta[0]) - self.pose[2])

        h = np.empty((2,1))
        h[0] = expectedRange
        h[1] = expectedBearing

        # Compute the Jacobian H of the measurement function h wrt the landmark location
        H = np.zeros((2,2))
        H[0,0] = ((landmarkX - self.pose[0])/h[0])[0]
        H[0,1] = ((landmarkY - self.pose[1])/h[0])[0]
        H[1,0] = ((self.pose[1] - landmarkY)/(h[0]**2))[0]
        H[1,1] = ((landmarkX - self.pose[0])/(h[0]**2))[0]
        return h,H

    # --------
    # measurement_prob
    #    computes the probability of a measurement
    #
    def correction_step(self, ids, ranges, bearings, wrld_dict):
        self.weight = 1.0
        # Construct the sensor noise matrix Q_t
        Q = [[0.5, 0], [0, 0.5]]
        num_measurements = len(ids)

        # Loop over each measurement
        for i in range(num_measurements):
            '''
            The (2x2) EKF of the landmark is given by
            its mean particles(i).landmarks(l).mu
            and by its covariance particles(i).landmarks(l).sigma
            '''
            # If the landmark is observed for the first time:
            if not self.landmarks[ids[i]][0]:
                 # Initialize its position based on the measurement and the current robot pose:
                 abs_bearing = normalize_angle(bearings[i] + self.pose[2])
                 self.landmarks[ids[i]][1] = [cos(abs_bearing)*ranges[i]+self.pose[0], sin(abs_bearing)*ranges[i]+self.pose[1]]

                 # Get the Jacobian with respect to the landmark position
                 [h, H] = self.measurement_model(ids[i])

                 # Initialize the covariance for this landmark
                 invH = np.linalg.inv(H)
                 self.landmarks[ids[i]][2] = np.dot(np.dot(invH, Q), np.transpose(invH))

                 # Indicate that this landmark has been observed
                 self.landmarks[ids[i]][0] = True
            else:
                  # Get the expected measurement
                  [expectedZ, H] = self.measurement_model(ids[i])
                  sigma = self.landmarks[ids[i]][2]

                  # Calculate the Kalman gain
                  # Measurement Covariance
                  S = np.dot(np.dot(H, sigma), np.transpose(H)) + Q

                  # Kalman Gain
                  K = np.dot(np.dot(sigma, np.transpose(H)), np.linalg.inv(S))

                  # Compute the error between the z and expectedZ
                  deltaZ = [ranges[i] - expectedZ[0], bearings[i] - expectedZ[1]]

                  # Update the mean and covariance of the EKF
                  self.landmarks[ids[i]][1] += np.dot(K, deltaZ)
                  self.landmarks[ids[i]][2] = np.dot((np.identity(len(deltaZ)) - np.dot(K, H)), sigma)

                  # Compute the likelihood of this observation, multiply with the former weight
                  # to account for observing several features in one time step
                  likelihood = sts.multivariate_normal.pdf([ranges[i], bearings[i]], [float(expectedZ[0]), float(expectedZ[1])], S)
                  print [ranges[i], bearings[i]]
                  print expectedZ
                  print S
                  print "####"
                  self.weight *= likelihood
        print "-------------------------"
        return self.weight

    # --------
    # move_odom:
    # Takes in Odometry ~ poses and updates the particles
    def mov_odom(self,odom,noise):
        #append the old position
        np.hstack((self.history,self.pose))

        # Calculate the distance and Guassian noise
        dist  = odom['t']

        # calculate delta rotation 1 and delta rotation 1
        delta_rot1  = odom['r1']
        delta_rot2 = odom['r2']

        # noise sigma for delta_rot1
        sigma_delta_rot1 = noise[0]
        delta_rot1_noisy = delta_rot1 + random.gauss(0,sigma_delta_rot1)

        # noise sigma for translation
        sigma_translation = noise[1]
        translation_noisy = dist + random.gauss(0,sigma_translation)

        # noise sigma for delta_rot2
        sigma_delta_rot2 = noise[2]
        delta_rot2_noisy = delta_rot2 + random.gauss(0,sigma_delta_rot2)

        # Estimate of the new position of the robot
        x_new = self.pose[0]  + translation_noisy * cos(self.pose[2]+delta_rot1_noisy)
        y_new = self.pose[1]  + translation_noisy * sin(self.pose[2]+delta_rot1_noisy)
        theta_new = normalize_angle(self.pose[2] + delta_rot1_noisy + delta_rot2_noisy)
        self.pose = np.array([x_new,y_new,theta_new])

    # Set the weight of the particles
    def set_weights(self, weight):
        #noise parameters
        self.weights  = float(weight)

    def sense_range(self,add_noise):
        Z = []
        for i in range(len(landmarks)):
            anchor = landmarks[i]
            delta_x =  anchor[1] - self.x
            delta_y =  anchor[0] - self.y
            distance = sqrt ((delta_x ** 2) + (delta_y ** 2))
            if(add_noise):
                distance += random.gauss(0.0,self.range_noise)
            Z.append(distance)
        return Z

def plotCovariance(center, matrix, color):
    eigvalue, eigvector = np.linalg.eig(matrix)
    theta = atan2(eigvector[0][1], eigvector[0][0])
    ellipse = Ellipse([center[0], center[1]], 2*pow(eigvalue[0], 0.5), 2*pow(eigvalue[1], 0.5), theta)
    ellipse.set_facecolor(color)
    fig = plt.gcf()
    fig.gca().add_artist(ellipse)

def draw_particle(sensor_data, p):
    # draw expected landmark positions
    for ldmk in p.landmarks:
        # seen landmark before
        if ldmk and ldmk[0]:
            plotCovariance(ldmk[1], ldmk[2], [1.0, 0.0, 0.0])

    # draw landmark measurements
    ldmk_x = []
    ldmk_y = []
    for i in range(len(sensor_data['id'])):
        rng = sensor_data['range'][i]
        bearing = normalize_angle(sensor_data['bearing'][i] + p.pose[2])
        ldmk_x.append(cos(bearing)*rng+p.pose[0])
        ldmk_y.append(sin(bearing)*rng+p.pose[1])
    plt.plot(ldmk_x,ldmk_y,'go', markersize=5)

# --------
#
# extract position from a particle set
#
def get_position(p):
    x = 0.0
    y = 0.0
    x_pos = []
    y_pos = []
    angle_x = 0.0
    angle_y = 0.0

    for pt in p:
        x += pt.pose[0]
        y += pt.pose[1]
        x_pos.append(pt.pose[0])
        y_pos.append(pt.pose[1])
        angle_x = angle_x + cos(pt.pose[2])
        angle_y = angle_y + sin(pt.pose[2])

    avg_angle_x = angle_x / len(p)
    avg_angle_y = angle_y / len(p)
    avg_orient = math.atan2(avg_angle_y, avg_angle_x)

    plt.plot(x_pos,y_pos,'r.')
    quiver_len = 3.0
    plt.quiver(x / len(p), y / len(p), quiver_len * np.cos(avg_orient), quiver_len * np.sin(avg_orient),angles='xy',scale_units='xy')

    plt.plot(lx,ly,'bo',markersize=10)
    plt.axis([-2, 15, 0, 15])
    plt.draw()
    plt.clf()
    return [x / len(p), y / len(p), avg_orient ]

# --------
#
def particle_filter(data_dict,world_dict, N): #
    # --------
    #
    # Make particles
    #
    p = []
    for i in range(N):
        r = robot()
        p.append(r)

    # --------
    #
    # Update particles
    #
    for t in range(len(data_dict)/2):
        print "Round: " + str(t)
        # motion update (prediction)
        for i in range(N):
            p[i].mov_odom(data_dict[t,'odom'],noise_param)

        # measurement update
        w = []
        for i in range(N):
            w.append(p[i].correction_step(data_dict[t,'sensor']['id'],data_dict[t,'sensor']['range'],data_dict[t,'sensor']['bearing'],world_dict))

        # resampling
        S = sum(w)
        minW = min(w)
        maxW = max(w)
        minIndex = w.index(minW)
        maxIndex = w.index(maxW)
        print S

        #normalize the weights
        w_norm=[w[j]/S for j in range(N)]

        #print 'Weights are normalized with total weight of %0.3s' % (numpy.sum(w_norm))
        cdf_sum=0
        p_cdf=[]

        for k in range(len(p)):
            cdf_sum = cdf_sum + w_norm[k]
            p_cdf.append(cdf_sum)
        #print 'CDF calculated'

        # Calculate the step for random sampling
        step = 1.0/N

        # Sample a value in between [0,step)
        seed = random.uniform(0, step)
        #print 'Seed is %0.15s and step is %0.15s' %(seed, step)

        p_sampled=[]
        last_index = 0
        for h in range(len(p)):
            while seed > p_cdf[last_index]:
                last_index+=1
            p_sampled.append(p[last_index])
            seed += step
        p = p_sampled

        #draw_particle(data_dict[t, 'sensor'], p[minIndex])
        draw_particle(data_dict[t, 'sensor'], p[maxIndex])
        print get_position(p)
    return get_position(p)

## Main loop Starts here
parser = argparse.ArgumentParser()
parser.add_argument('sensor_data', type=str, help='Sensor Data')
parser.add_argument('world_data', type=str, help='World Data')
parser.add_argument('N', type=int, help='Number of particles')

args = parser.parse_args()
N = args.N
noise_param = np.array([0.005, 0.01, 0.005]).T

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

print world_data
print estimated_position

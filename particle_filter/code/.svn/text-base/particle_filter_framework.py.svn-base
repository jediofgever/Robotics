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
            
            id_arr.append(float(line_spl[1]))    
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
    # measurement_prob
    #    computes the probability of a measurement 
   ''' Complete this Stub for the measurement model which is based on range only
        Range only implies that the error has to be calculated based only on the range values'''
        
        
        # This function takes in the ranges , and ids of the sensor data for each measurement update
        # World data is passed as the dictionary. The positions of the landmarks in the world 
        # can be accessed as wrld_dict[ids[i]][0],wrld_dict[ids[i]][1] 
        # where i is looped over the number of measurements for the current timestamp
       
       
       
    def measurement_prob_range(self, ids, ranges, wrld_dict):

            
        return error
   





    # --------
    # move_odom: 
    # Takes in Odometry data

    def mov_odom(self,odom,noise):
    
        # Calculate the distance and Guassian noise
       
        dist  = odom['t']

        # calculate delta rotation 1 and delta rotation 1
        
        delta_rot1  = odom['r1']
        
        delta_rot2 = odom['r2']
        

        # noise sigma for delta_rot1 
        sigma_delta_rot1 = noise[0]*abs(delta_rot1)  + noise[1]*abs(dist)
        delta_rot1_noisy = delta_rot1 + random.gauss(0,sigma_delta_rot1)

        # noise sigma for translation
        sigma_translation = noise[2]*abs(dist)  + noise[3]*abs(delta_rot1+delta_rot2)
        translation_noisy = dist + random.gauss(0,sigma_translation)

        # noise sigma for delta_rot2
        sigma_delta_rot2 = noise[0]*abs(delta_rot2)  + noise[1]*abs(dist)
        delta_rot2_noisy = delta_rot2 + random.gauss(0,sigma_delta_rot2)



        # Estimate of the new position of the robot
        x_new = self.x  + translation_noisy * cos(self.orientation+delta_rot1_noisy)
        y_new = self.y  + translation_noisy * sin(self.orientation+delta_rot1_noisy)
        theta_new = self.orientation + delta_rot1_noisy + delta_rot2_noisy
       
       
       
        

        result = robot()
        result.set(x_new, y_new,theta_new )
        
        return result




    # Set the weight of the particles
    def set_weights(self, weight):
        #noise parameters
        self.weights  = float(weight)





# --------
#
# extract position from a particle set
# 

def get_mean_position(p):
    x = 0.0
    y = 0.0
    orientation = 0.0
   
    ''' Write the code here to calculate the mean position and orientation of all the particles'''






    # Particles are plotted here 
    ''' Lists x_pos and y_pos contains the x and y positions of all the particles which can be accessed from 
        p[i].x , p[i].y         , avg_oreint is the average orientation of all the particles'''
        
        
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
       ''' Resmapling the Particles 
           Complete this Stub to resample the weights of the particles  '''
       
def resample_particles(weights, particles):
        
      
        #Sum the weights 
       
       
       
       
       
        #normalize the weights
       
       

        
        
        
        # calculate the PDF of the weights
        pdf=[]        
        for k in range(len(p)):
            

        

        # Calculate the step for random sampling , it depends on number of 
        #particles

        
        step=        
        
        
        # Sample a value in between [0,step) uniformly
       
        seed = 
        #print 'Seed is %0.15s and step is %0.15s' %(seed, step)


        # resample the particles based on the seed , step and cacluated pdf
        for h in range(len(p)):
             
             
            '''Write the code here'''                 
             
             
            
        return p_sampled

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
    
       
           
        
        # motion update (prediction)
        p2 = []
        for i in range(N):
            p2.append(p[i].mov_odom(data_dict[t,'odom'],noise_param))
            
        p = p2

        # measurement update
        w = []
        for i in range(N):
            w.append(p[i].measurement_prob_range(data_dict[t,'sensor']['id'],data_dict[t,'sensor']['range'],world_dict))
           
        
        
       ''' Resmapling the Particles 
           Complete this Stub to resample the weights of the particles  '''
       
       
      p = resample_particles(w,p)
        
    return get_position(p)


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
world_data  =read_world_data(args.world_data)

lx=[]
ly=[]

for i in range (len(world_data)):
    lx.append(world_data[i+1][0])
    ly.append(world_data[i+1][1])

estimated_position = particle_filter(data_dict,world_data,N)

print estimated_position



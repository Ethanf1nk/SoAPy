from SoAPy.functions import set_options
from SoAPy.functions import convert_GROMACS
from SoAPy.functions import generate_files
#from SoAPy.functions import generate_coordinates
from SoAPy.functions import generate_batch_submission_script
from SoAPy.functions import collect_data
from SoAPy.functions import custom_collect_data
from SoAPy.analysis import opt_rot_averaging
import time
import os
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd()

# Location of MD trajectory.
#trajectory_location = cwd + '/' + 'SoAPy/SoAPy/tests/B_glucose_test_MD.arc'

# Molecule name
molecule_name = 'H2O'

# Set testing parameters in lists.
spectroscopy = ['OptRot']
shell_type = ['spherical']
functional = ['CAM-B3LYP']
basis_set = ['SadleJ pVTZ']
distance_threshold = ['3', '3.5', '4.5', '5']
snapshots = ['1']
frequency = ['633nm', '589nm', '436nm', '355nm']

t0 = time.time()

# What type of test is being researched, basis sets, distance thresholds, or number of snapshots?
dir_list, dir_parameters, relative_dir_list = set_options(spectroscopy, shell_type, functional, basis_set, distance_threshold, snapshots, frequency)
t1 = time.time()

#Initializing lists for each spectrum
snaps_633_spec = []
snaps_589_spec = []
snaps_436_spec = []
snaps_355_spec = []

#The number of snapshots we will average NOT THE TOTAL NUMBER OF SNAPSHOTS THAT WE RAN IN GAUSSIAN
collections_of_interest =  [1]


#Loop over the different numbers of snapshots averaged
for i in range(len(collections_of_interest)):
    #Loop over the number of different choices for distance threshold
        #starter_cmpd = int(dir_parameters[b][5])//collections_of_interest[i]
        print("Number of Snapshots to average: ", collections_of_interest[i])
        #print("Cmpd # we start with/interval: ", starter_cmpd)
        #cmpd_interval = starter_cmpd
        
        #Function to collect data from Gaussian outputs, the cmpd number sampled and interval between cmpds are controllable with the default arguments
        dir_frequencies, dir_intensities, dir_nbf = custom_collect_data(cwd, dir_list, dir_parameters, conformer = 1, interval = 1, top_out = collections_of_interest[i], averaging = True)
        t2 = time.time()
        
        #Function to average together the optical rotations for each test
        avg_633_list, avg_589_list, avg_436_list, avg_355_list = opt_rot_averaging(dir_frequencies, dir_intensities, dir_parameters, dir_list, radius = True, averaging = True)
        


#plotting individual lines for each frequency
plt.plot(distance_threshold, avg_633_list, label= "633")
plt.plot(distance_threshold, avg_589_list, label= "589")
plt.plot(distance_threshold, avg_436_list, label= "436")
plt.plot(distance_threshold, avg_355_list, label= "355")

#Axis labels
plt.xlabel('Solvent Shell Radius (A)')
plt.ylabel('Average Normalized Optical Rotation')

#Axis bound limits
#plt.ylim(-4,4)

#Plotting the legend
plt.legend(loc="upper right")

#Saving the figure with a descriptive title
plt.savefig(f'/Users/ehfhi/SoAPy/tests-radius-sadlej-100snap.pdf')

print("Total Time: ", t2-t0)

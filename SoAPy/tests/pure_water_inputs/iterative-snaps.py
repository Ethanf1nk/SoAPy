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
distance_threshold = ['2.75']
snapshots = ['1000']
frequency = ['633nm', '589nm', '436nm', '355nm']

t0 = time.time()

# What type of test is being researched, basis sets, distance thresholds, or number of snapshots?
dir_list, dir_parameters, relative_dir_list = set_options(spectroscopy, shell_type, functional, basis_set, distance_threshold, snapshots, frequency)
t1 = time.time()

snaps_633 = []
snaps_589 = []
snaps_436 = []
snaps_355 = []
collections_of_interest =  [25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
#collections_of_interest = np.arange(1, 900, 25, dtype=int)
for b in range(len(dir_list)):
    for i in range(len(collections_of_interest)):
        #starter_cmpd = int(dir_parameters[b][5])//collections_of_interest[i]
        print("Number of Snapshots to average: ", collections_of_interest[i])
        #print("Cmpd # we start with/interval: ", starter_cmpd)
        #cmpd_interval = starter_cmpd
        
        #Function to collect data from Gaussian outputs, the cmpd number sampled and interval between cmpds are controllable with the default arguments
        dir_frequencies, dir_intensities, dir_nbf = custom_collect_data(cwd, dir_list, dir_parameters, conformer = 1, interval = 1, top_out = collections_of_interest[i], averaging = True)
        t2 = time.time()
        
        #Function to average together the optical rotations for each test
        avg_633_list, avg_589_list, avg_436_list, avg_355_list = opt_rot_averaging(dir_frequencies, dir_intensities, dir_parameters, dir_list, radius = False, averaging = True)
        snaps_633.append(avg_633_list)
        snaps_589.append(avg_589_list)
        snaps_436.append(avg_436_list)
        snaps_355.append(avg_355_list)
print(snaps_355)
#plotting individual lines for each frequency
#collects = np.ndarray(collections_of_interest)
plt.plot(collections_of_interest, snaps_633, label= "633")
plt.plot(collections_of_interest, snaps_589, label= "589")
plt.plot(collections_of_interest, snaps_436, label= "436")
plt.plot(collections_of_interest, snaps_355, label= "355")

#Axis labels
plt.xlabel('Number of Snapshots Averaged')
plt.ylabel('Average Normalized Optical Rotation')

#Axis bound limits
#plt.ylim(-4,4)

#Plotting the legend
plt.legend(loc="upper right")

#Saving the figure with a descriptive title
plt.savefig(f'/Users/ehfhi/SoAPy/test.pdf')

print("Total Time: ", t2-t0)

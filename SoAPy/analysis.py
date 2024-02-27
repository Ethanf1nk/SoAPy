"""Provide functions for analyzing the resultant data."""

import numpy as np
import time 
import os
import shutil
import math
import matplotlib as mpl 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 



def single_denominator_overlap(dir_frequency_axis, dir_intensity_axis, sample_index, reference_index):
    """
    Function for computing the overlap between two spectra. This function includes a normalization associated with only one of the two spectra rather than both.
    This is specifically designed to include discrepancies between intensities which the "double_denominator_overlap" would neglect.
    """
    # Set sample variables.
    sample_intensity = np.array(dir_intensity_axis[sample_index])

    # Set reference variables.
    reference_frequency = np.array(dir_frequency_axis[reference_index])
    reference_intensity = np.array(dir_intensity_axis[reference_index])

    # Calculate numerator.
    numerator = np.trapz(reference_intensity * sample_intensity, x = reference_frequency)

    # Calculate denominator.
    #if np.trapz(reference_intensity * reference_intensity, x = reference_frequency) > np.trapz(sample_intensity * sample_intensity, x = reference_frequency):
    denominator = np.trapz(reference_intensity * reference_intensity, x = reference_frequency)
    #else:
    #    denominator = np.trapz(sample_intensity * sample_intensity, x = reference_frequency)

    # Calculate overlap.
    overlap = numerator / denominator

    return overlap



def double_denominator_overlap(dir_frequency_axis, dir_intensity_axis, sample_index, reference_index):
    """ 
    Function for computing the overlap between two spectra. This function includes a normalization associated with only one of the two spectra rather than both.
    This is specifically designed to include discrepancies between intensities which the "double_denominator_overlap" would neglect.
    """
    # Set sample variables.
    sample_intensity = np.array(dir_intensity_axis[sample_index])

    # Set reference variables.
    reference_frequency = np.array(dir_frequency_axis[reference_index])
    reference_intensity = np.array(dir_intensity_axis[reference_index])

    # Calculate numerator.
    numerator = np.trapz(reference_intensity * sample_intensity, x = reference_frequency)

    # Calculate denominator.
    denominator = np.sqrt(np.trapz(reference_intensity * reference_intensity, x = reference_frequency) * np.trapz(sample_intensity * sample_intensity, x = reference_frequency))

    # Calculate overlap.
    overlap = numerator / denominator

    return overlap



def integrated_difference(dir_frequency_axis, dir_intensity_axis, sample_index, reference_index):
    """
    Integrates the difference between two functions and makes a ratio of them of the combined area of each spectrum.
    """
    # Set sample variables.
    sample_intensity = np.array(dir_intensity_axis[sample_index])

    # Set reference variables.
    reference_frequency = np.array(dir_frequency_axis[reference_index])
    reference_intensity = np.array(dir_intensity_axis[reference_index])

    # Calculate numerator.
    numerator = np.trapz(reference_intensity**2, x = reference_frequency) - np.trapz(sample_intensity**2, x = reference_frequency)

    # Calculate denominator.
    denominator = np.trapz(reference_intensity**2, x = reference_frequency)

    # Calculate integrated difference.
    int_diff = numerator / denominator

    return int_diff



def compute_statistics(dir_intensities, sample_index, reference_index):
    """
    Computes the differences in mean, variance, and standard deviation with respect to some reference.
    """
    # Calculate mean intensity.
    sample_mean = np.average(dir_intensities[sample_index])
    reference_mean = np.average(dir_intensities[reference_index])

    # Calculate variance in intensities.
    sample_variance_numerator = 0
    reference_variance_numerator = 0
    for b in range(len(dir_intensities[sample_index])):
        sample_variance_numerator += (dir_intensities[sample_index][b] - sample_mean)**2
        sample_variance = sample_variance_numerator / len(dir_intensities[sample_index])

        reference_variance_numerator += (dir_intensities[reference_index][b] - reference_mean)**2
        reference_variance = reference_variance_numerator / len(dir_intensities[reference_index])

    # Calculate the standard deviation.
    sample_standard_deviation = np.sqrt(sample_variance)
    reference_standard_deviation = np.sqrt(reference_variance)

    # Calculate absolute differences.
    mean_diff = abs(reference_mean - sample_mean)
    variance_diff = abs(reference_variance - sample_variance)
    standard_deviation_diff = abs(reference_standard_deviation - sample_standard_deviation)

    return mean_diff, variance_diff, standard_deviation_diff



def wrong_signs(dir_frequencies, dir_intensities, sample_index, reference_index):
    """
    Prints out the frequencies at which there is a sign change. This could be due to normal mode rearrangements or wrong signs. Further developement is needed to distinguish.
    """
    # Set sample variables.
    sample_frequency = np.array(dir_frequencies[sample_index])
    sample_intensity = np.array(dir_intensities[sample_index])

    # Set reference variables.
    reference_frequency = np.array(dir_frequencies[reference_index])
    reference_intensity = np.array(dir_intensities[reference_index])

    # Calculate frequency displacement and sign change.
    delta_freq = []
    sign_change = []
    for a in range(len(reference_frequency)):
        frequency_displacement = reference_frequency[a] - sample_frequency[a]
        intensity_sign = sample_intensity[a] / reference_intensity[a]
        if intensity_sign > 0:
            sign = 1
        elif intensity_sign < 0:
            sign = -1

        delta_freq.append(frequency_displacement)
        sign_change.append(sign)

    return delta_freq, sign_change



def normal_mode_reordering(dir_frequencies, dir_intensities, sample_index, reference_index):
    """
    Looks for normal mode reorderings by sign changes in the slope between consecutive normal modes.
    """
    # Set sample variables.
    sample_frequency = np.array(dir_frequencies[sample_index])
    sample_intensity = np.array(dir_intensities[sample_index])

    # Set reference variables.
    reference_frequency = np.array(dir_frequencies[reference_index])
    reference_intensity = np.array(dir_intensities[reference_index])

    # Calculate frequency displacement and sign change.
    slope_sign = []
    for a in range(len(reference_frequency)):
        if a == len(reference_frequency)-1:
            slope_sign.append(0)
        else:
            delta_reference_frequency = reference_frequency[a] - reference_frequency[a+1]
            delta_reference_intensity = reference_intensity[a] - reference_intensity[a+1]
            reference_derivative = delta_reference_intensity / delta_reference_frequency

            delta_sample_frequency = sample_frequency[a] - sample_frequency[a+1]
            delta_sample_intensity = sample_intensity[a] - sample_intensity[a+1]
            sample_derivative = delta_sample_intensity / delta_sample_frequency

            derivative_sign = sample_derivative / reference_derivative

            if derivative_sign > 0:
                sign = 1
            elif derivative_sign < 0:
                sign = -1

            slope_sign.append(sign)

    return slope_sign


def opt_rot_averaging(dir_frequencies, dir_intensities, dir_parameters, dir_list, radius, averaging):
    #Beta version of the running average analysis
    #Initialize running sums for each test frequency
    ints_633 = 0.0
    ints_589 = 0.0
    ints_436 = 0.0
    ints_355 = 0.0

    #Tracking the total number of negative signs in rotations
    #Doing this to test the idea of stricted conformation sign bias
    neg_633 = 0
    neg_589 = 0
    neg_436 = 0
    neg_355 = 0
    
    #Lists to store the averages
    avg_633_list = []
    avg_589_list = []
    avg_436_list = []
    avg_355_list = []

    #radius is a function arg that tells the function if we are investigating a changing radius
    if radius == True:
        for b in range(len(dir_list)):
            ints_633 = 0.0
            ints_589 = 0.0
            ints_436 = 0.0
            ints_355 = 0.0
            chopper = 4*int(dir_parameters[b][5])
            #The chopper splits the data in dir_frequencies and dir_intensities by radius
            #This is done by calculating the number of entries that are associated with each radius and sampling them seperately
            for a in range(0, chopper):
                #Logic gates to split up the intensities to totals for their corresponding frequency
                if dir_frequencies[b][a] == 633.0:
                    ints_633 += dir_intensities[b][a]
                    #The number of negative signs per frequency is tracked to show if a test/range of tests has a tendency to favor one sign
                    if dir_intensities[b][a] < 0.0:
                        neg_633 += 1
                elif dir_frequencies[b][a] == 589.0:
                    ints_589 += dir_intensities[b][a]
                    if dir_intensities[b][a] < 0.0:
                        neg_589 += 1
                elif dir_frequencies[b][a] == 436.0:
                    ints_436 += dir_intensities[b][a]
                    if dir_intensities[b][a] < 0.0:
                        neg_436 += 1
                elif dir_frequencies[b][a] == 355.0:
                    ints_355 += dir_intensities[b][a]
                    if dir_intensities[b][a] < 0.0:
                        neg_355 += 1
                    
            #Finding total number of snaps
            tot_snaps = int(dir_parameters[b][5])
            #Calculating the average intensity for each freq
            #Checks the averaging argument, should only be set to True if custom_collect_data is also set to True
            #If the averaging argument is true then we are dividing by the molar mass of one atom in the system
            #THE SYSTEM IS WATER ONLY AT THIS TIME
            if averaging == True:
                avg_633 = ints_633/(tot_snaps*18.02)
                avg_589 = ints_589/(tot_snaps*18.02)
                avg_436 = ints_436/(tot_snaps*18.02)
                avg_355 = ints_355/(tot_snaps*18.02)
            
            elif averaging == False:
                avg_633 = ints_633/(tot_snaps)
                avg_589 = ints_589/(tot_snaps)
                avg_436 = ints_436/(tot_snaps)
                avg_355 = ints_355/(tot_snaps)

            #Assembling all the averages in a list
            avg_633_list.append(avg_633)
            avg_589_list.append(avg_589)
            avg_436_list.append(avg_436)
            avg_355_list.append(avg_355)
            
            #Print statements to inspect data if needed
            #print(len(avg_355_list))
            #print(ints_633, ints_589, ints_436, ints_355)
            #print(avg_355_list)
                
        
    
    elif radius == False:    
        #Pulling the intensities that correspond to each individual frequency tested
        #This loop uses dir_frequecies because we are assembling a UNIQUE dir_frequencies for EACH time we run the collect_data function
        #####
        #This ONLY works properly when inside of a loop within the script
        #####
        for a in range(len(dir_frequencies[0])):
            #Logic gates to split up the intensities to totals for their corresponding frequency
            if dir_frequencies[b][a] == 633.0:
                ints_633 += dir_intensities[b][a]
                #The number of negative signs per frequency is tracked to show if a test/range of tests has a tendency to favor one sign
                if dir_intensities[0][a] < 0.0:
                    neg_633 += 1
            elif dir_frequencies[0][a] == 589.0:
                ints_589 += dir_intensities[0][a]
                if dir_intensities[0][a] < 0.0:
                    neg_589 += 1
            elif dir_frequencies[0][a] == 436.0:
                ints_436 += dir_intensities[0][a]
                if dir_intensities[0][a] < 0.0:
                    neg_436 += 1
            elif dir_frequencies[0][a] == 355.0:
                ints_355 += dir_intensities[0][a]
                if dir_intensities[0][a] < 0.0:
                    neg_355 += 1
                
        #Finding total number of snaps
        tot_snaps = len(dir_frequencies[0])/4
        
        #Calculating the average intensity for each freq
        #Checks the averaging argument, should only be set to True if custom_collect_data is also set to True
        if averaging == True:
            avg_633 = ints_633/(tot_snaps*18.02)
            avg_589 = ints_589/(tot_snaps*18.02)
            avg_436 = ints_436/(tot_snaps*18.02)
            avg_355 = ints_355/(tot_snaps*18.02)
        
        elif averaging == False:
            avg_633 = ints_633/(tot_snaps)
            avg_589 = ints_589/(tot_snaps)
            avg_436 = ints_436/(tot_snaps)
            avg_355 = ints_355/(tot_snaps)

        #Assembling all the averages in a list
        avg_633_list.append(avg_633)
        avg_589_list.append(avg_589)
        avg_436_list.append(avg_436)
        avg_355_list.append(avg_355)
                
        #avg_list = [avg_633, avg_589, avg_436, avg_355]
        #print(avg_list)

    #Printing the number of negative signs for each test
    print("Number of negative signs: 633nm = ", neg_633, " 589nm = ", neg_589, " 436nm = ", neg_436, " 355nm = ", neg_355)

    return avg_633_list, avg_589_list, avg_436_list, avg_355_list








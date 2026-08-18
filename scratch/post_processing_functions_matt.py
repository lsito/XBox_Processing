"""These functions can be used to add scaled gradient to the 
standard Xbox2/3 processing setup
This is a separate file for easier version control

Functions:
Scaled Gradient - used to calculate scaled gradient. 
This is a separate function for code readability

Add Scaled Gradient - follows the conventions of previous codes 
and includes Xbox2 and Xbox3 support
"""

import numpy as np

def scale_gradient(gradient,
                    bdr,
                    bdr_ref,
                    pulse_length,
                    pulse_length_ref):
    """Scales the input gradient to a nominal pulse length and breakdown rate
    
    This is done using the assumed relation between Gradient, BDR and Pulse Length
    
    Arguments
    gradient -- Current Accelerating Gradient
    bdr -- Current Breakdown Rate
    bdr_ref -- Nominal Breakdown Rate
    pulse_length -- Current Pulse Length
    pulse_length_ref -- Nominal Pulse Length 
    """
    scaled_gradient = (gradient
                       * np.power(pulse_length / pulse_length_ref, 1 / 6)
                       / np.power(bdr / bdr_ref, 1 / 30))
    return scaled_gradient

def add_scaled_gradient(
        input_data,
        bdr_ref,
        pulse_length_ref,
        xbox,
        structure_stand
        ):
    """" Adds column to input data containing scaled gradients
    
    Arguments
    input_data -- data taken from tdms file, in a dataFrame
    bdr_ref -- Nominal Breakdown Rate
    pulse_length_ref -- Nominal Pulse Length
    xbox -- Xbox 2 or Xbox 3
    structure_ stand -- Structure 1 or 2 in Xbox 3
    """
    if xbox==2:
        input_data['scaled_gradient_A'] = scale_gradient(
                                            input_data['gradient_A'],
                                            input_data['BDR_DUT_A'],
                                            bdr_ref,
                                            input_data['PSIA_peak_length'],
                                            pulse_length_ref)
        input_data['scaled_gradient_B'] = scale_gradient(
                                            input_data['gradient_B'],
                                            input_data['BDR_DUT_B'],
                                            bdr_ref,
                                            input_data['PSIB_peak_length'],
                                            pulse_length_ref)
    elif xbox == 3:
        if structure_stand == 1:
            input_data['scaled_gradient_A'] = scale_gradient(
                                            input_data['gradient_A'],
                                            input_data['BDR_DUT_A'],
                                            bdr_ref,
                                            input_data['PSIA_peak_length'],
                                            pulse_length_ref)
        elif structure_stand==2:
            input_data['scaled_gradient_B'] = scale_gradient(
                                            input_data['gradient_B'],
                                            input_data['BDR_DUT_B'],
                                            bdr_ref,
                                            input_data['PSIB_peak_length'],
                                            pulse_length_ref)
    return input_data

"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
from functools import reduce

def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.
    
    :param data: inflammation data
    :returns: array of means
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: inflammation data
    :returns: max per day
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: inflammation data
    :returns: min per day
    """
    return np.min(data, axis=0)

def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """

    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')

    max_data = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised

def daily_above_threshold(row, data, threshold):
    """
    number of daily values above the threshold
    :param row:
    :param data:
    :param threshold:
    :returns:
    """
    above = map(lambda x: x > threshold, data[row, ])

    def count_above_threshold(a, b):
        if b:
            return a + 1
        else:
            return a

    return reduce(count_above_threshold, above, 0)
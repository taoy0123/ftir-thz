import numpy as np

def load_spectrum(filename):
    """Load a Numpy array from a text file.

    :param filename: Filename of txt to load
    :return: 2D Numpy array of data
    """
    spectrum = np.loadtxt(filename)

    return spectrum

def save_spectrum(filename, wavenumber, values):
    data = np.column_stack((wavenumber, values))
    np.savetxt(filename, data, delimiter=",", header="wavenumber,values", comments="")

def calculate_difference_absorbance(sample_intensities, reference_intensities):
    """
    Calculate the difference absorbance (A_sample - A_reference) between a sample and a reference.

    Parameters:
    sample_intensities (array-like): The sample data.
    reference_intensities (array-like): The reference data.

    Returns:
    array-like: The difference absorbance.
    """

    # Convert inputs to numpy arrays for element-wise operations
    sample = np.array(sample_intensities)
    reference = np.array(reference_intensities)

    # Check if the shapes of the sample and reference match
    if sample.shape != reference.shape:
        raise ValueError("Sample and reference must have the same shape.")

    # Calculate the difference absorbance
    difference = -np.log(sample / reference)

    return difference
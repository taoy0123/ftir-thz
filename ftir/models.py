import numpy as np

def load_spectrum(filename):
    """Load a Numpy array from a text file.

    :param filename: Filename of txt to load
    :return: 2D Numpy array of data
    """
    spectrum = np.loadtxt(filename)
    frequencies = spectrum[:, 0]  # Assuming the first column is frequency
    intensities = spectrum[:, 1]  # Assuming the second column is intensity
    return frequencies, intensities

def save_spectra(filename, spectra):
    """spectra: list of (label, frequency, delta_A). Each pair gets two columns."""
    n_rows = max(len(freq) for _, freq, _ in spectra)
    columns, header = [], []
    for label, freq, delta_A in spectra:
        label = label.replace(",", "_")  # commas would break the CSV header
        for name, values in ((f"frequency [{label}]", freq), (f"delta_A [{label}]", delta_A)):
            col = np.full(n_rows, np.nan)  # pad shorter spectra with NaN
            col[: len(values)] = values
            columns.append(col)
            header.append(name)
    np.savetxt(
        filename, np.column_stack(columns),
        delimiter=",", header=",".join(header), comments="",
    )

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
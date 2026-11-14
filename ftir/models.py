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

def empty_cell_thickness(
    empty_intensity,
    background_intensity,
    wavenumber,
    wavenumber_range=None,
    min_periods=3,
    min_thickness_um=5.0,
    highpass_width=500.0,
    min_prominence=6.0):
    """Estimate the gap thickness (micrometres) of an empty cell.

    The difference absorbance of an empty cell shows interference fringes
    (an etalon formed between the two windows). If the fringe period is P
    (in cm^-1), the gap is d = 1 / (2 P), taking the gap to be air (n = 1).

    Parameters
    ----------
    empty_intensity, background_intensity : array-like
        Intensities measured with the empty cell and with the background.
    wavenumber : array-like
        Shared wavenumber axis in cm^-1.
    wavenumber_range : (low, high), optional
        Only search inside this range, e.g. to skip noisy spectrum edges.
    min_periods : float
        Fewest fringes that must fit inside the range for an oscillation to
        be accepted. This sets the thinnest gap that can be detected.
    min_thickness_um : float
        Thinnest gap to look for. Slower oscillations (thinner gaps) are
        ignored, which keeps broad absorption bands from being mistaken for
        fringes.
    highpass_width : float
        Width (cm^-1) of the moving average subtracted before the search.
        Structure broader than this, such as absorption bands and baseline
        drift, is removed. It should be wider than the fringe period
        1 / (2 d) of the gaps you expect (500 suits gaps of about 10 um and up).
    min_prominence : float
        How many times larger than the typical (median) amplitude the strongest
        oscillation must be to count as real. Pure noise gives about 3-4, clear
        fringes give 10 or more; raise this to be stricter.

    Returns
    -------
    float
        Gap thickness in micrometres.

    Raises
    ------
    ValueError
        If there are too few valid points or no clear oscillation is found.
    """
    wavenumber = np.asarray(wavenumber, dtype=float)
    delta_A = calculate_difference_absorbance(
        np.asarray(empty_intensity, dtype=float),
        np.asarray(background_intensity, dtype=float),
    )

    # Keep finite points, optionally restricted to a wavenumber range
    keep = np.isfinite(wavenumber) & np.isfinite(delta_A)
    if wavenumber_range is not None:
        low, high = wavenumber_range
        keep &= (wavenumber >= low) & (wavenumber <= high)
    wavenumber, delta_A = wavenumber[keep], delta_A[keep]
    if wavenumber.size < 16:
        raise ValueError("Not enough valid points to look for an oscillation.")

    order = np.argsort(wavenumber)
    wavenumber, delta_A = wavenumber[order], delta_A[order]

    # Resample onto an evenly spaced axis so the FFT is valid
    step = np.median(np.diff(wavenumber))
    if not step > 0:
        raise ValueError("The wavenumber axis must be increasing.")
    grid = np.arange(wavenumber[0], wavenumber[-1], step)
    signal = np.interp(grid, wavenumber, delta_A)
    span = grid[-1] - grid[0]

    # High-pass filter: subtract a moving average so that broad absorption
    # bands and baseline drift disappear and only fast fringes remain
    half = min(int(round(highpass_width / step)) // 2, signal.size - 1)
    padded = np.pad(signal, half, mode="reflect")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    signal = signal - np.convolve(padded, kernel, mode="valid")

    # Window to reduce spectral leakage
    signal = signal * np.hanning(signal.size)

    # FFT over wavenumber: a fringe of period P appears at 1/P (in cm), which
    # equals 2*d. Zero-padding gives a finer sampling of the peak position.
    n_fft = 1 << int(np.ceil(np.log2(signal.size * 16)))
    amplitude = np.abs(np.fft.rfft(signal, n=n_fft))
    path = np.fft.rfftfreq(n_fft, d=step)

    # Ignore components too slow to be fringes: fewer than min_periods fringes
    # in the range, or a gap thinner than min_thickness_um (path = 2 * gap)
    floor = max(min_periods / span, 2.0 * min_thickness_um * 1e-4)
    allowed = np.flatnonzero(path >= floor)
    if allowed.size < 3:
        raise ValueError("The wavenumber range is too short to find an oscillation.")
    peak = allowed[np.argmax(amplitude[allowed])]
    if peak == allowed[0] or peak == allowed[-1]:
        raise ValueError("No clear oscillation found in the difference absorbance.")
    # The strongest peak must stand well above the rest, otherwise it is just noise
    if amplitude[peak] < min_prominence * np.median(amplitude[allowed]):
        raise ValueError("No clear oscillation found in the difference absorbance.")

    # Refine the peak position with a parabola through the three highest points
    a, b, c = amplitude[peak - 1], amplitude[peak], amplitude[peak + 1]
    denominator = a - 2 * b + c
    offset = 0.5 * (a - c) / denominator if denominator != 0 else 0.0
    path_difference = path[peak] + offset * (path[1] - path[0])

    period = 1.0 / path_difference        # fringe period, cm^-1
    thickness_cm = 1.0 / (2.0 * period)   # same as path_difference / 2
    return thickness_cm * 1e4             # micrometres

def align_spectra(freq_a, int_a, freq_b, int_b):
    """Restrict two spectra to their common wavenumber range on one shared axis.

    The shared axis is taken from whichever spectrum has fewer points inside
    the overlap (the coarser one, so nothing is upsampled); the other spectrum
    is linearly interpolated onto it. Returns (frequency, int_a, int_b), all
    the same length.

    Raises ValueError if the two spectra do not overlap.
    """
    freq_a, int_a = np.asarray(freq_a, dtype=float), np.asarray(int_a, dtype=float)
    freq_b, int_b = np.asarray(freq_b, dtype=float), np.asarray(int_b, dtype=float)

    # Sort each by wavenumber (interpolation needs an increasing axis)
    order = np.argsort(freq_a)
    freq_a, int_a = freq_a[order], int_a[order]
    order = np.argsort(freq_b)
    freq_b, int_b = freq_b[order], int_b[order]

    low = max(freq_a[0], freq_b[0])
    high = min(freq_a[-1], freq_b[-1])
    if low >= high:
        raise ValueError("The two files have no overlapping wavenumber range.")

    in_a = (freq_a >= low) & (freq_a <= high)
    in_b = (freq_b >= low) & (freq_b <= high)
    if in_a.sum() < 2 or in_b.sum() < 2:
        raise ValueError("The overlapping wavenumber range is too small to use.")

    if in_a.sum() <= in_b.sum():
        frequency = freq_a[in_a]
        return frequency, int_a[in_a], np.interp(frequency, freq_b, int_b)
    frequency = freq_b[in_b]
    return frequency, np.interp(frequency, freq_a, int_a), int_b[in_b]

def absorption_coefficient(delta_A, thickness_um):
    """Convert difference absorbance to difference absorption coefficient (cm^-1).

    alpha = delta_A / d, with d in cm. This assumes delta_A is the natural-log
    form, as in calculate_difference_absorbance. For base-10 absorbance, multiply by ln(10).
    """
    return np.asarray(delta_A, dtype=float) / (thickness_um * 1e-4)

def group_by_axis(spectra, tolerance=1e-6):
    """Group spectra that share the same wavenumber axis.
 
    spectra: list of (label, frequency, values).
    Returns a list of (frequency, [(label, values), ...]), one entry per
    distinct axis, in order of first appearance.
    """
    groups = []
    for label, freq, values in spectra:
        freq = np.asarray(freq, dtype=float)
        for axis, members in groups:
            if axis.shape == freq.shape and np.allclose(axis, freq, rtol=0, atol=tolerance):
                members.append((label, values))
                break
        else:
            groups.append((freq, [(label, values)]))
    return groups
 
 
def resample_to_common_axis(spectra):
    """Put every spectrum on one shared wavenumber axis.
 
    Keeps only the wavenumber range that all spectra cover. The axis is taken
    from the spectrum with the fewest points in that range (so nothing is
    upsampled) and the others are linearly interpolated onto it.
    Returns a list of (label, frequency, values) sharing one frequency array.
 
    Raises ValueError if the spectra have no wavenumber range in common.
    """
    prepared = []
    for label, freq, values in spectra:
        freq = np.asarray(freq, dtype=float)
        values = np.asarray(values, dtype=float)
        order = np.argsort(freq)
        prepared.append((label, freq[order], values[order]))
 
    low = max(freq[0] for _, freq, _ in prepared)
    high = min(freq[-1] for _, freq, _ in prepared)
    if low >= high:
        raise ValueError("The spectra have no wavenumber range in common.")
 
    counts = [int(((freq >= low) & (freq <= high)).sum()) for _, freq, _ in prepared]
    best = int(np.argmin(counts))
    if counts[best] < 2:
        raise ValueError("The common wavenumber range is too small to use.")
    reference = prepared[best][1]
    axis = reference[(reference >= low) & (reference <= high)]
 
    return [(label, axis, np.interp(axis, freq, values)) for label, freq, values in prepared]
 
 
def save_spectra(filename, spectra, y_name="delta_A"):
    """Save several spectra to one CSV file.
 
    spectra: list of (label, frequency, values). Spectra that share the same
    wavenumber axis share one wavenumber column, which is written once. With
    several distinct axes, each axis gets its own wavenumber column
    (wavenumber_1, wavenumber_2, ...) followed by the spectra that use it,
    padded with NaN where the lengths differ. y_name names the value columns,
    e.g. "delta_A" or "alpha_per_cm".
    """
    groups = group_by_axis(spectra)
    n_rows = max(len(axis) for axis, _ in groups)
 
    def padded(data):
        col = np.full(n_rows, np.nan)
        col[: len(data)] = data
        return col
 
    columns, header = [], []
    for number, (axis, members) in enumerate(groups, start=1):
        header.append("wavenumber" if len(groups) == 1 else f"wavenumber_{number}")
        columns.append(padded(axis))
        for label, values in members:
            header.append(f"{y_name} [{label.replace(',', '_')}]")  # commas would break the header
            columns.append(padded(values))
 
    np.savetxt(
        filename,
        np.column_stack(columns),
        delimiter=",",
        fmt="%.10g",
        header=",".join(header),
        comments="",
    )
 

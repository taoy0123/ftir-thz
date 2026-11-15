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

def correct_water_vapor(
    frequencies,
    values,
    water_frequencies,
    water_values,
    fit_range=(60.0, 400.0),
    highpass_width=30.0,
    baseline_order=2,
    return_scale=False,
):
    """Remove water vapour lines from a spectrum (e.g. difference absorbance).

    Here only the SHARP structure is fitted:
      * the reference is high-passed (moving average of `highpass_width` cm^-1
        removed), so only the lines remain and the baseline is never touched;
      * the scale s is fitted by least squares with a low-order polynomial
        baseline as extra free terms, over `fit_range` (where the signal is
        good; above ~500 cm^-1 the HCl/water spectra are noise);
      * s is signed, because the sample may contain more or less vapour than the
        reference;
      * corrected = values - s * high-passed reference.

    Also: the reference is sorted (np.interp needs an ascending axis), NaN points
    are left out of the fit and kept as NaN, and the result has the same length
    and axis as `values`.

    Returns corrected, or (corrected, s) if return_scale is True.
    """
    x = np.asarray(frequencies, dtype=float)
    y = np.asarray(values, dtype=float)
    wf = np.asarray(water_frequencies, dtype=float)
    wv = np.asarray(water_values, dtype=float)

    ok = np.isfinite(wf) & np.isfinite(wv)
    wf, wv = wf[ok], wv[ok]
    order = np.argsort(wf)
    wf, wv = wf[order], wv[order]
    if wf.size < 3:
        raise ValueError("The water vapour spectrum has too few valid points.")

    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 10:
        raise ValueError("Too few valid points to remove water vapour.")
    srt = np.argsort(x[good])
    xg, yg = x[good][srt], y[good][srt]

    ref = np.interp(xg, wf, wv, left=0.0, right=0.0)
    step = np.median(np.abs(np.diff(xg)))
    half = max(int(round(highpass_width / step)) // 2, 1)
    padded = np.pad(ref, half, mode="reflect")
    smooth = np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")
    lines = ref - smooth

    lo, hi = fit_range
    sel = (xg >= lo) & (xg <= hi)
    if sel.sum() < 3 * (baseline_order + 2):
        raise ValueError(
            f"The spectrum does not cover the {lo:g}-{hi:g} cm\u207b\u00b9 range "
            "used to fit the water vapour."
        )
    if not np.any(np.abs(lines[sel]) > 0):
        raise ValueError("The water vapour spectrum is empty in the fitting range.")

    t = (xg[sel] - xg[sel].mean()) / (np.ptp(xg[sel]) or 1.0)
    design = np.column_stack([lines[sel]] + [t**k for k in range(baseline_order + 1)])
    coef, *_ = np.linalg.lstsq(design, yg[sel], rcond=None)
    scale = float(coef[0])

    corrected = np.full(y.shape, np.nan)
    idx = np.flatnonzero(good)[srt]
    corrected[idx] = yg - scale * lines
    return (corrected, scale) if return_scale else corrected

def _sinc_interp(y, positions, half_width=8, beta=6.0):
    """Interpolate evenly spaced samples `y` at fractional index `positions`.

    Windowed-sinc (Kaiser) interpolation. Near the ends only real samples are
    used and the weights are renormalised, so nothing is invented outside the data.
    A plain linear interpolation here would leave a few percent of a fringe
    uncancelled when there are only 3-5 points per fringe.
    """
    n = y.size
    positions = np.clip(positions, 0, n - 1)
    base = np.floor(positions).astype(int)
    index = base[:, None] + np.arange(-half_width + 1, half_width + 1)[None, :]
    distance = positions[:, None] - index
    window = np.i0(beta * np.sqrt(np.clip(1.0 - (distance / half_width) ** 2, 0.0, None))) / np.i0(beta)
    weights = np.sinc(distance) * window
    weights = np.where((index >= 0) & (index < n), weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights * y[np.clip(index, 0, n - 1)], axis=1)


def _fringe_axis(wavenumbers, absorbance):
    """Sort a spectrum, put it on an evenly spaced axis and fill NaN points.

    Returns (order, sorted_wavenumbers, grid, values, was_nan, step, uniform).
    `values` has NaN points filled by interpolation (only used for the maths);
    `was_nan` marks them so they can be restored afterwards.
    """
    wn = np.asarray(wavenumbers, dtype=float)
    ab = np.asarray(absorbance, dtype=float)
    if wn.ndim != 1 or wn.shape != ab.shape:
        raise ValueError("Wavenumbers and values must be 1D arrays of the same length.")
    if wn.size < 32:
        raise ValueError("Too few points to remove fringes.")
    if not np.all(np.isfinite(wn)):
        raise ValueError("The wavenumber axis contains NaN or infinite values.")

    order = np.argsort(wn)
    wn_sorted, values = wn[order], ab[order]
    spacing = np.diff(wn_sorted)
    step = float(np.median(spacing))
    if not step > 0:
        raise ValueError("The wavenumber axis must be increasing.")

    uniform = bool(np.allclose(spacing, step, rtol=1e-3, atol=0.0))
    if uniform:
        grid = wn_sorted
    else:  # uneven spacing: work on an evenly spaced axis and map back at the end
        grid = np.arange(wn_sorted[0], wn_sorted[-1] + step / 2, step)
        values = np.interp(grid, wn_sorted, values)

    was_nan = ~np.isfinite(values)
    if (~was_nan).sum() < 32:
        raise ValueError("Too few valid points to remove fringes.")
    if was_nan.any():
        values = np.interp(grid, grid[~was_nan], values[~was_nan])
    return order, wn_sorted, grid, values, was_nan, step, uniform


def _ripple_peak(grid, values, step, shortest, longest):
    """Strongest ripple with a period between `shortest` and `longest` (cm^-1).

    Works on one stretch of an evenly spaced, NaN-free spectrum. Returns
    (period, prominence), or None if the stretch is too short or the strongest
    peak sits at the edge of the search band (so it is not a real peak).
    Prominence is the peak height divided by the typical (median) height.
    """
    span = grid[-1] - grid[0] if grid.size else 0.0
    if grid.size < 16 or span < longest:
        return None

    # Remove everything broader than the fringes (bands, baseline, cell fringes)
    half = int(round(min(5.0 * longest, span / 2.0) / step)) // 2
    half = max(1, min(half, values.size - 1))
    padded = np.pad(values, half, mode="reflect")
    ripple = values - np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")

    # Spectrum of the ripple; zero-padding gives a finely sampled peak
    n_fft = 1 << int(np.ceil(np.log2(ripple.size * 16)))
    amplitude = np.abs(np.fft.rfft(ripple * np.hanning(ripple.size), n=n_fft))
    frequency = np.fft.rfftfreq(n_fft, d=step)  # cycles per cm^-1
    band = np.flatnonzero((frequency >= 1.0 / longest) & (frequency <= 1.0 / shortest))
    if band.size < 3:
        return None

    peak = band[np.argmax(amplitude[band])]
    if peak == band[0] or peak == band[-1]:
        return None
    prominence = amplitude[peak] / np.median(amplitude[band])

    # Refine the peak position with a parabola through the three highest points
    a, b, c = amplitude[peak - 1], amplitude[peak], amplitude[peak + 1]
    denominator = a - 2.0 * b + c
    offset = 0.5 * (a - c) / denominator if denominator != 0 else 0.0
    return 1.0 / (frequency[peak] + offset * (frequency[1] - frequency[0])), prominence


def _fringe_period(grid, values, step, period_range, region, min_prominence, tolerance=0.05):
    """Measure the fringe period (cm^-1), or raise ValueError if there are no clear fringes.

    Two tests guard against mistaking other structure for fringes. The ripple
    peak must stand well above the typical ripple (min_prominence), and, when
    the spectrum is long enough, a third of the spectrum at a time must give
    the same period (within `tolerance`) in at least two of three parts.
    Etalon fringes have one period everywhere; chance structure does not.
    """
    shortest, longest = period_range
    shortest = max(shortest, 2.0 * step)  # periods under two points cannot be resolved
    if not 0 < shortest < longest:
        raise ValueError("period_range does not suit the point spacing of this spectrum.")

    if region is not None:
        keep = (grid >= region[0]) & (grid <= region[1])
        grid, values = grid[keep], values[keep]
    if grid.size < 16 or grid[-1] - grid[0] < longest:
        raise ValueError("The wavenumber region is too short to measure the fringe period.")

    found = _ripple_peak(grid, values, step, shortest, longest)
    if found is None or found[1] < min_prominence:
        raise ValueError(
            "No clear fringes found. Check period_range, give the period directly, "
            "or measure it in a region where the fringes are visible (region=...)."
        )
    period = found[0]

    if grid[-1] - grid[0] >= 6.0 * longest:
        agreeing = 0
        for part in np.array_split(np.arange(grid.size), 3):
            result = _ripple_peak(grid[part], values[part], step, shortest, longest)
            if result is not None and abs(result[0] - period) <= tolerance * period:
                agreeing += 1
        if agreeing < 2:
            raise ValueError(
                "No consistent fringes found: the ripple period changes along the "
                "spectrum, so it is probably not an etalon fringe. Give the period "
                "directly, or measure it in a clean region (region=...)."
            )
    return period


def remove_fringes(
    wavenumbers,
    absorbance,
    period=None,
    period_range=(2.0, 10.0),
    region=(30, 100),
    min_prominence=5.0,
):
    """Suppress etalon fringes (e.g. from diamond windows) by half-period averaging.

    A fringe of period P cancels when the spectrum is averaged with a copy of
    itself shifted by P/2. Here each point becomes the mean of the spectrum at
    x - P/4 and x + P/4, so results stay centred on the original wavenumbers.
    Both shifted copies are read with sinc interpolation, which keeps the
    cancellation accurate even though P/4 is rarely a whole number of points.
    Near the ends of the spectrum the pair of samples is moved inwards so that
    only real data is used. The output therefore has the same length, order
    and wavenumbers as the input.

    Parameters
    ----------
    wavenumbers, absorbance : array-like
        The spectrum. Any order (increasing or decreasing). NaN values in
        `absorbance` are ignored in the calculation and stay NaN in the result.
    period : float, optional
        Fringe period in cm^-1. If None it is measured from the spectrum.
    period_range : (shortest, longest), optional
        Periods (cm^-1) considered when measuring. The shortest is limited to
        two points of the spectrum.
    region : (low, high), optional
        Only measure the period in this wavenumber range (the correction is
        still applied to the whole spectrum). Use it if something else in the
        spectrum ripples at similar spacing, for example water vapour lines.
    min_prominence : float
        How many times higher than the typical ripple the fringe peak must be
        for the fringes to count as clear. Lower it only if you trust weak fringes.

    Returns
    -------
    corrected : ndarray
        Spectrum with fringes suppressed, same length and order as the input.
    period : float
        The fringe period used (cm^-1).

    Raises
    ------
    ValueError
        If the data cannot be used, or no clear fringes are found when the
        period is being measured.

    Notes
    -----
    The fundamental and its odd harmonics cancel; even harmonics (period P/2)
    do not, which matters little for the low-finesse fringes of diamond
    windows. Features narrower than about P/2 are slightly smoothed. Slow
    structure, including the much slower fringes of a thin cell gap, is not
    affected. Instrument spectra are normally evenly spaced. If yours is not,
    it is linearly resampled onto an even axis first, which is much less
    accurate (roughly a fifth of a 4 cm^-1 fringe can remain).
    """
    order, wn_sorted, grid, values, was_nan, step, uniform = _fringe_axis(
        wavenumbers, absorbance
    )

    if period is None:
        period = _fringe_period(grid, values, step, period_range, region, min_prominence)
    else:
        period = float(period)
        if not np.isfinite(period) or period < 2.0 * step:
            raise ValueError("period must be at least two points of the spectrum.")

    half = period / 2.0
    if half >= grid[-1] - grid[0]:
        raise ValueError("The fringe period is longer than the spectrum.")

    # Position of the first of the two averaged samples, in index units: P/4
    # before each point, but moved inwards near the ends so both stay in range
    n = grid.size
    index = np.arange(n)
    half_in_points = half / step
    first = index + np.minimum(
        np.maximum(-half_in_points / 2.0, -index), (n - 1 - index) - half_in_points
    )
    corrected = 0.5 * (
        _sinc_interp(values, first) + _sinc_interp(values, first + half_in_points)
    )
    corrected[was_nan] = np.nan

    if not uniform:
        corrected = np.interp(wn_sorted, grid, corrected)
    result = np.empty_like(corrected)
    result[order] = corrected  # back to the order the spectrum came in
    return result, period

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
 

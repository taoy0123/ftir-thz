"""
The controller connects the view's controls to the models:
  * sample/reference pairs are aligned to their common wavenumber range
  * empty cell + background give the gap thickness (enables the absorption
    coefficient switch)
  * a water vapour reference file enables the water vapour switch
  * the fringe switch needs no extra file
  * corrections are applied in this order: water vapour, fringes, then the
    conversion to absorption coefficient
  * Save CSV asks what to do when the displayed spectra use different
    wavenumber axes

It uses your own models functions: load_spectrum, calculate_difference_absorbance,
remove_water_vapor and remove_fringes_halfperiod (through the correct_* wrappers).
"""

from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

from ftir import models
from ftir.views import FILETYPES, LoadPairsDialog


class ProcessingError(Exception):
    """A correction failed for a spectrum. `which` is "water" or "fringes"."""

    def __init__(self, which, message):
        super().__init__(message)
        self.which = which


class Controller:
    def __init__(self, view):
        self.view = view
        self.spectra = []              # one dict per pair: label, frequency, delta_A
        self.cell_thickness_um = None  # set when empty cell and background are both valid
        self.water = None              # (wavenumbers, values) of the water vapour reference

        # Connect every control in the view to its action
        view.load_button.configure(command=self.load_files)
        view.save_button.configure(command=self.save_csv)
        view.clear_button.configure(command=self.clear_spectra)
        view.empty_button.configure(command=self.choose_empty_cell)
        view.empty_clear.configure(command=self.clear_empty_cell)
        view.background_button.configure(command=self.choose_background)
        view.background_clear.configure(command=self.clear_background)
        view.water_button.configure(command=self.choose_water)
        view.water_clear.configure(command=self.clear_water)
        view.display_switch.configure(command=self.toggle_display)
        view.water_switch.configure(command=self.toggle_display)
        view.fringe_switch.configure(command=self.toggle_display)

    # ------------------------------------------------------------------
    # Sample/reference pairs
    # ------------------------------------------------------------------
    def load_files(self):
        """Ask for one or more pairs and ADD them to what is already shown."""
        dialog = LoadPairsDialog(self.view)
        self.view.wait_window(dialog)

        if dialog.result is None:  # user cancelled
            return

        first_number = len(self.spectra) + 1  # keeps labels unique: Pair 3, Pair 4, ...
        new_spectra = []
        try:
            for number, (sample_file, reference_file) in enumerate(
                dialog.result, start=first_number
            ):
                new_spectra.append(self._process_pair(number, sample_file, reference_file))
        except (OSError, ValueError) as err:
            # If any new pair fails, none are added and the plot is unchanged
            messagebox.showerror("Could not load files", str(err), parent=self.view)
            return

        self.spectra.extend(new_spectra)
        self._warn_about_bad_points(new_spectra)
        self._redraw()

    def clear_spectra(self):
        """Remove all pairs from the plot (the reference files stay set)."""
        self.spectra = []
        self._redraw()

    # ------------------------------------------------------------------
    # Empty cell and background
    # ------------------------------------------------------------------
    def choose_empty_cell(self):
        self._choose_file(self.view.empty_var, "Select empty cell file", self._update_cell)

    def choose_background(self):
        self._choose_file(self.view.background_var, "Select background file", self._update_cell)

    def clear_empty_cell(self):
        self.view.empty_var.set("")
        self._update_cell()

    def clear_background(self):
        self.view.background_var.set("")
        self._update_cell()

    def _update_cell(self):
        """Recompute the gap thickness; enable the switch only if it is available."""
        empty = self.view.empty_var.get()
        background = self.view.background_var.get()
        self.cell_thickness_um = None

        if empty and background:  # both must be specified
            try:
                empty_wn, empty_int = self._load_xy(empty)
                background_wn, background_int = self._load_xy(background)
                wavenumber, empty_int, background_int = models.align_spectra(
                    empty_wn, empty_int, background_wn, background_int
                )
                self.cell_thickness_um = models.empty_cell_thickness(
                    empty_int, background_int, wavenumber
                )
            except (OSError, ValueError) as err:
                messagebox.showerror("Empty cell / background", str(err), parent=self.view)

        # Order matters: this disables and resets the switch when thickness is None
        self.view.set_cell_thickness(self.cell_thickness_um)
        self._redraw()

    # ------------------------------------------------------------------
    # Water vapour reference
    # ------------------------------------------------------------------
    def choose_water(self):
        self._choose_file(self.view.water_var, "Select water vapour spectrum", self._update_water)

    def clear_water(self):
        self.view.water_var.set("")
        self._update_water()

    def _update_water(self):
        """Load the water vapour reference; enable its switch only if it is valid."""
        path = self.view.water_var.get()
        self.water = None

        if path:
            try:
                wavenumber, values = self._load_xy(path)
                wavenumber = np.asarray(wavenumber, dtype=float)
                values = np.asarray(values, dtype=float)
                good = np.isfinite(wavenumber) & np.isfinite(values)
                if good.sum() < 2:
                    raise ValueError("The water vapour file has too few valid points.")
                self.water = (wavenumber[good], values[good])
            except (OSError, ValueError) as err:
                self.view.water_var.set("")
                messagebox.showerror("Water vapour spectrum", str(err), parent=self.view)

        # Disables and resets the switch when there is no valid reference
        self.view.set_water_available(self.water is not None)
        self._redraw()

    # ------------------------------------------------------------------
    # Display and saving
    # ------------------------------------------------------------------
    def toggle_display(self):
        """Called when any of the switches is flipped. Keeps the current zoom."""
        self._redraw(keep_view=True)

    def save_csv(self):
        """Save all displayed spectra (as currently shown) to one CSV file."""
        if not self.spectra:
            messagebox.showinfo("No data", "Load and process files first.", parent=self.view)
            return

        spectra, _, y_name = self._displayed_safe()

        # Check whether the spectra need more than one wavenumber axis
        if len(models.group_by_axis(spectra)) > 1:
            answer = messagebox.askyesnocancel(
                "Different wavenumber axes",
                "The displayed spectra do not all share the same wavenumbers.\n\n"
                "Yes: keep every spectrum on its original wavenumbers "
                "(separate wavenumber columns).\n"
                "No: resample all spectra onto one shared axis, keeping only the "
                "range they all cover.\n"
                "Cancel: do not save.",
                parent=self.view,
            )
            if answer is None:  # cancel
                return
            if answer is False:  # resample
                try:
                    spectra = models.resample_to_common_axis(spectra)
                except ValueError as err:
                    messagebox.showerror("Could not resample", str(err), parent=self.view)
                    return

        path = filedialog.asksaveasfilename(
            parent=self.view,
            title="Save spectra",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="spectra.csv",
        )
        if not path:  # user cancelled
            return

        try:
            models.save_spectra(path, spectra, y_name)
        except OSError as err:
            messagebox.showerror("Could not save file", str(err), parent=self.view)

    def _displayed(self):
        """Return (spectra, axis_label, csv_column_name) for what the switches select.

        Raises ProcessingError if a correction fails for a spectrum.
        """
        use_coefficient = self.view.coefficient_mode() and self.cell_thickness_um is not None
        use_water = self.view.water_mode() and self.water is not None
        use_fringes = self.view.fringe_mode()

        spectra = []
        for s in self.spectra:
            frequency, values = s["frequency"], s["delta_A"]

            if use_water:
                try:
                    values = models.correct_water_vapor(frequency, values, *self.water)
                except ValueError as err:
                    raise ProcessingError("water", f"{s['label']}: {err}") from err

            if use_fringes:
                try:
                    values, _ = models.remove_fringes(frequency, values)
                except ValueError as err:
                    raise ProcessingError("fringes", f"{s['label']}: {err}") from err

            if use_coefficient:
                values = models.absorption_coefficient(values, self.cell_thickness_um)
            spectra.append((s["label"], frequency, values))

        # CSV column names say which corrections were applied
        y_name = "alpha_per_cm" if use_coefficient else "delta_A"
        if use_water:
            y_name += "_no_water"
        if use_fringes:
            y_name += "_no_fringes"
        ylabel = (
            "Difference absorption coefficient (cm⁻¹)"
            if use_coefficient
            else "Difference absorbance, ΔA"
        )
        return spectra, ylabel, y_name

    def _displayed_safe(self):
        """Like _displayed, but if a correction fails, tell the user and turn it off."""
        while True:  # ends because each failure switches one correction off
            try:
                return self._displayed()
            except ProcessingError as err:
                messagebox.showerror(
                    "Correction failed",
                    f"{err}\n\nThe switch has been turned off.",
                    parent=self.view,
                )
                self.view.reset_switch(err.which)

    def _redraw(self, keep_view=False):
        spectra, ylabel, _ = self._displayed_safe()
        self.view.plot_spectra(spectra, ylabel, keep_view=keep_view)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _choose_file(self, variable, title, after):
        path = filedialog.askopenfilename(parent=self.view, title=title, filetypes=FILETYPES)
        if path:  # empty string means the user cancelled
            variable.set(path)
            after()

    @staticmethod
    def _load_xy(path):
        """Return (wavenumber, intensity) arrays for one data file.

        Uses models.load_spectrum. If yours returns something other than a
        two-column array or a (wavenumber, intensity, ...) tuple, adapt it here
        so that every file is read in this one place.
        """
        data = models.load_spectrum(path)
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 0], data[:, 1]
        return data[0], data[1]

    def _process_pair(self, number, sample_file, reference_file):
        """Load one sample/reference pair, align it and compute its absorbance."""
        sample_wn, sample_int = self._load_xy(sample_file)
        ref_wn, ref_int = self._load_xy(reference_file)

        try:
            frequency, sample_int, ref_int = models.align_spectra(
                sample_wn, sample_int, ref_wn, ref_int
            )
        except ValueError as err:
            raise ValueError(f"Pair {number}: {err}") from err

        delta_A = models.calculate_difference_absorbance(sample_int, ref_int)

        return {
            "label": f"Pair {number}: {Path(sample_file).stem}",
            "frequency": frequency,
            "delta_A": delta_A,
        }

    def _warn_about_bad_points(self, spectra):
        """Warn about points that could not be converted in the newly added pairs."""
        problems = []
        for s in spectra:
            n_bad = int(np.isnan(s["delta_A"]).sum())
            if n_bad:
                problems.append(f"{s['label']}: {n_bad} points")
        if problems:
            messagebox.showwarning(
                "Invalid points",
                "Some points could not be converted to absorbance "
                "(intensity ratio was zero or negative):\n\n" + "\n".join(problems),
                parent=self.view,
            )
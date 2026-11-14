"""Complete replacement for ftir/controller.py.

Adds empty cell / background handling (gap thickness) and the switch between
difference absorbance and difference absorption coefficient. The controller now
connects the view's buttons itself, so the .configure(command=...) lines in
main.py are no longer needed (leaving them in is harmless).
"""

from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

from ftir import models
from ftir.views import FILETYPES, LoadPairsDialog


class Controller:
    def __init__(self, view):
        self.view = view
        self.spectra = []              # one dict per pair: label, frequency, delta_A
        self.cell_thickness_um = None  # set when empty cell and background are both valid

        # Connect every control in the view to its action
        view.load_button.configure(command=self.load_files)
        view.save_button.configure(command=self.save_csv)
        view.clear_button.configure(command=self.clear_spectra)
        view.empty_button.configure(command=self.choose_empty_cell)
        view.empty_clear.configure(command=self.clear_empty_cell)
        view.background_button.configure(command=self.choose_background)
        view.background_clear.configure(command=self.clear_background)
        view.display_switch.configure(command=self.toggle_display)

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
        """Remove all pairs from the plot (the empty cell/background stay set)."""
        self.spectra = []
        self._redraw()

    # ------------------------------------------------------------------
    # Empty cell and background
    # ------------------------------------------------------------------
    def choose_empty_cell(self):
        self._choose_file(self.view.empty_var, "Select empty cell file")

    def choose_background(self):
        self._choose_file(self.view.background_var, "Select background file")

    def clear_empty_cell(self):
        self.view.empty_var.set("")
        self._update_cell()

    def clear_background(self):
        self.view.background_var.set("")
        self._update_cell()

    def _choose_file(self, variable, title):
        path = filedialog.askopenfilename(parent=self.view, title=title, filetypes=FILETYPES)
        if path:  # empty string means the user cancelled
            variable.set(path)
            self._update_cell()

    def _update_cell(self):
        """Recompute the gap thickness; enable the switch only if it is available."""
        empty = self.view.empty_var.get()
        background = self.view.background_var.get()
        self.cell_thickness_um = None

        if empty and background:  # both must be specified
            try:
                wavenumber, empty_int = self._load_xy(empty)
                bg_wavenumber, background_int = self._load_xy(background)
                wavenumber, empty_int = self._load_xy(empty)
                bg_wavenumber, background_int = self._load_xy(background)
                wavenumber, empty_int, background_int = models.align_spectra(wavenumber, empty_int, bg_wavenumber, background_int)

                self.cell_thickness_um = models.empty_cell_thickness(empty_int, background_int, wavenumber)

            except (OSError, ValueError) as err:
                messagebox.showerror(
                    "Empty cell / background", str(err), parent=self.view
                )

        # Order matters: this disables and resets the switch when thickness is None
        self.view.set_cell_thickness(self.cell_thickness_um)
        self._redraw()

    # ------------------------------------------------------------------
    # Display and saving
    # ------------------------------------------------------------------
    def toggle_display(self):
        """Called when the switch is flipped."""
        self._redraw()

    def save_csv(self):
        """Save all displayed spectra (in the units currently shown) to one CSV."""
        if not self.spectra:
            messagebox.showinfo("No data", "Load and process files first.", parent=self.view)
            return

        spectra, _, y_name = self._displayed()

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
        """Return (spectra, axis_label, csv_column_name) for what is currently shown."""
        use_coefficient = self.view.coefficient_mode() and self.cell_thickness_um is not None
        spectra = []
        for s in self.spectra:
            values = (
                models.absorption_coefficient(s["delta_A"], self.cell_thickness_um)
                if use_coefficient
                else s["delta_A"]
            )
            spectra.append((s["label"], s["frequency"], values))

        if use_coefficient:
            return spectra, "Difference absorption coefficient (cm\u207b\u00b9)", "alpha_per_cm"
        return spectra, "Difference absorbance, \u0394A", "delta_A"

    def _redraw(self):
        spectra, ylabel, _ = self._displayed()
        self.view.plot_spectra(spectra, ylabel)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_xy(path):
        """Return (wavenumber, intensity) arrays for one data file.

        ADAPT HERE: put the unpacking that already works in your current
        _process_pair in this one place, so every file is read the same way.
        """
        data = models.load_spectrum(path)
        return data[0], data[1]

    def _process_pair(self, number, sample_file, reference_file):
        """Load one sample/reference pair, validate it and compute its difference absorbance."""
        sample_freq, sample_int = self._load_xy(sample_file)
        ref_freq, ref_int = self._load_xy(reference_file)

        try:
            frequency, sample_int, ref_int = models.align_spectra(sample_freq, sample_int, ref_freq, ref_int)
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

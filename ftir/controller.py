"""Complete replacement for ftir/controller.py: handles several sample/reference pairs."""

from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

from ftir import models
from ftir.views import LoadPairsDialog


class Controller:
    def __init__(self, view):
        self.view = view
        # One entry per loaded pair: {"label", "frequency", "delta_A"}
        self.spectra = []

    # ------------------------------------------------------------------
    # Actions triggered from the view
    # ------------------------------------------------------------------
    def load_files(self):
        """Ask for one or more pairs and ADD them to what is already shown."""
        dialog = LoadPairsDialog(self.view)
        self.view.wait_window(dialog)

        if dialog.result is None:  # user cancelled
            return

        first_number = len(self.spectra) + 1  # keep labels unique: Pair 3, Pair 4, ...
        new_spectra = []
        try:
            for number, (sample_file, reference_file) in enumerate(dialog.result, start=first_number):
                new_spectra.append(self._process_pair(number, sample_file, reference_file))
        except (OSError, ValueError) as err:
            # If any new pair fails, none of them are added and the plot is unchanged
            messagebox.showerror("Could not load files", str(err), parent=self.view)
            return

        self.spectra.extend(new_spectra)
        self._warn_about_bad_points(new_spectra)
        self._redraw()

    def clear_spectra(self):
        """Remove everything from the plot."""
        self.spectra = []
        self._redraw()

    def _redraw(self):
        self.view.plot_spectra(
            [(s["label"], s["frequency"], s["delta_A"]) for s in self.spectra]
        )

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


    def save_csv(self):
        """Save all displayed spectra to one CSV file."""
        if not self.spectra:
            messagebox.showinfo("No data", "Load and process files first.", parent=self.view)
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
            models.save_spectra(
                path, [(s["label"], s["frequency"], s["delta_A"]) for s in self.spectra]
            )
        except OSError as err:
            messagebox.showerror("Could not save file", str(err), parent=self.view)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _process_pair(self, number, sample_file, reference_file):
        """Load one pair, validate it and compute its absorbance."""
        # Assumes models.load_data returns (frequency, intensity) for a file
        sample_freq, sample_int = models.load_spectrum(sample_file)
        ref_freq, ref_int = models.load_spectrum(reference_file)

        try:
            self._check_compatible(sample_freq, sample_int, ref_freq, ref_int)
        except ValueError as err:
            raise ValueError(f"Pair {number}: {err}") from err

        # Replace calculate_absorbance with your function name in models.
        delta_A = models.calculate_difference_absorbance(sample_int, ref_int)

        return {
            "label": f"Pair {number}: {Path(sample_file).stem}",
            "frequency": sample_freq,
            "delta_A": delta_A,
        }


    @staticmethod
    def _check_compatible(sample_freq, sample_int, ref_freq, ref_int):
        """Raise ValueError if the sample and reference cannot be combined."""
        if not (len(sample_freq) == len(sample_int) == len(ref_freq) == len(ref_int)):
            raise ValueError(
                f"Row counts differ: sample {len(sample_freq)}/{len(sample_int)}, "
                f"reference {len(ref_freq)}/{len(ref_int)}."
            )
        if not np.allclose(sample_freq, ref_freq):
            raise ValueError("Sample and reference frequency axes do not match.")
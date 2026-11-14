"""Controller: connects the view (customtkinter UI) to the models (data/maths)."""

import numpy as np
from tkinter import messagebox, filedialog

from ftir import models
from ftir.views import LoadFilesDialog  # adjust to wherever the dialog lives


class Controller:
    def __init__(self, view):
        self.view = view

        # Application state, filled in once files are loaded
        self.frequency = None
        self.sample_intensity = None
        self.reference_intensity = None
        self.delta_A = None

    # ------------------------------------------------------------------
    # Actions triggered from the view
    # ------------------------------------------------------------------
    def load_files(self):
        """Ask for the two files, load them, then compute and show absorbance."""
        dialog = LoadFilesDialog(self.view)
        self.view.wait_window(dialog)  # blocks until the dialog closes

        if dialog.result is None:  # user cancelled
            return
        sample_file, reference_file = dialog.result

        try:
            sample = models.load_spectrum(sample_file)
            reference = models.load_spectrum(reference_file)
            self._check_compatible(sample, reference)
        except (OSError, ValueError) as err:
            messagebox.showerror("Could not load files", str(err), parent=self.view)
            return

        # Assumes column 0 = frequency, column 1 = intensity
        self.frequency = sample[:, 0]
        self.sample_intensity = sample[:, 1]
        self.reference_intensity = reference[:, 1]

        self.compute_absorbance()

    def compute_absorbance(self):
        """Compute delta_A from the loaded data and send it to the view."""
        if self.sample_intensity is None or self.reference_intensity is None:
            messagebox.showinfo(
                "No data", "Load a sample and a reference file first.", parent=self.view
            )
            return

        # Replace calculate_absorbance with your function in models.
        # It should guard against ratios <= 0 so np.log does not return NaN.
        self.delta_A = models.calculate_difference_absorbance(
            self.sample_intensity, self.reference_intensity
        )

        n_bad = int(np.isnan(self.delta_A).sum())
        if n_bad:
            bad_freq = self.frequency[np.isnan(self.delta_A)]
            message = f"{n_bad} points could not be converted (ratio ≤ 0) at: {bad_freq}"
            messagebox.showwarning(
                "Invalid points",
                message,
                parent=self.view,
            )

        # Replace with whatever your view uses to display results.
        self.view.plot_spectrum(self.frequency, self.delta_A)

    def save_csv(self):
        if self.delta_A is None:
            messagebox.showinfo("No data", "Load and process files first.", parent=self.view)
            return

        path = filedialog.asksaveasfilename(
            parent=self.view,
            title="Save spectrum",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="spectrum.csv",
            )
        if not path:  # user cancelled
            return

        try:
            models.save_spectrum(path, self.frequency, self.delta_A)
        except OSError as err:
            messagebox.showerror("Could not save file", str(err), parent=self.view)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_compatible(sample, reference):
        """Raise ValueError if the two files cannot be combined."""
        for name, arr in (("Sample", sample), ("Reference", reference)):
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(
                    f"{name} file must have at least two columns "
                    "(frequency and intensity)."
                )
        if sample.shape[0] != reference.shape[0]:
            raise ValueError(
                f"Sample has {sample.shape[0]} rows but reference has "
                f"{reference.shape[0]}."
            )
        if not np.allclose(sample[:, 0], reference[:, 0]):
            raise ValueError("Sample and reference frequency axes do not match.")
"""Complete replacement for ftir/controller.py.

The controller connects the view's controls to the models:
  * sample/reference pairs are aligned to their common wavenumber range
  * empty cell + background give the gap thickness (enables the absorption
    coefficient switch)
  * a water vapour reference file enables the water vapour switch
  * the fringe switch needs no extra file
  * the bulk solvent switch converts to a molar absorption coefficient using each
    pair's own solvent settings (needs the gap thickness)
  * the SVD button decomposes the displayed spectra
  * corrections are applied in this order: water vapour, fringes, then the
    conversion to absorption coefficient, then solvent removal
  * Save CSV asks what to do when the displayed spectra use different
    wavenumber axes

It uses these functions from models.py: load_spectrum, calculate_difference_absorbance,
align_spectra, empty_cell_thickness, absorption_coefficient, correct_water_vapor,
remove_bulk_solvent, load_two_column, remove_fringes, group_by_axis,
resample_to_common_axis, save_spectra, svd_spectra and save_svd.
"""

from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

from ftir import models
from ftir.views import FILETYPES, LoadPairsDialog, SolventDialog, SVDDialog


class ProcessingError(Exception):
    """A correction failed for a spectrum. `which` is "water", "fringes" or "solvent"."""

    def __init__(self, which, message):
        super().__init__(message)
        self.which = which


class Controller:
    def __init__(self, view):
        self.view = view
        self.spectra = []              # one dict per pair: label, frequency, delta_A, file paths
        self.cell_thickness_um = None  # set when empty cell and background are both valid
        self._svd_dialog = None
        self._svd_state = {}  # settings/results kept between SVD windows
        self.fringe_skipped = []       # spectra the fringe correction had to skip
        self._skip_warned = set()      # ... that the user has already been told about
        self.water = None              # (wavenumbers, values) of the water vapour reference
        self._solvent_files = {}       # path -> (modified time, wavenumbers, values)

        # Connect every control in the view to its action
        view.load_button.configure(command=self.load_files)
        view.save_button.configure(command=self.save_csv)
        view.clear_button.configure(command=self.clear_spectra)
        view.svd_button.configure(command=self.show_svd)
        view.empty_button.configure(command=self.choose_empty_cell)
        view.empty_clear.configure(command=self.clear_empty_cell)
        view.background_button.configure(command=self.choose_background)
        view.background_clear.configure(command=self.clear_background)
        view.water_button.configure(command=self.choose_water)
        view.water_clear.configure(command=self.clear_water)
        view.display_switch.configure(command=self.toggle_display)
        view.water_switch.configure(command=self.toggle_display)
        view.fringe_switch.configure(command=self.toggle_display)
        view.solvent_switch.configure(command=self.toggle_display)
        view.solvent_button.configure(command=self.edit_solvent)

    # ------------------------------------------------------------------
    # Sample/reference pairs
    # ------------------------------------------------------------------
    def load_files(self):
        """Add new pairs and/or remove loaded ones, using one dialog.

        The dialog lists the pairs already loaded; the ones the user leaves in
        stay in their position and are all read again from the files shown in the
        dialog, and any new pairs are added after them.
        """
        existing = [(s["label"], s["sample_file"], s["reference_file"]) for s in self.spectra]
        dialog = LoadPairsDialog(self.view, existing)
        self.view.wait_window(dialog)

        if dialog.result is None:  # user cancelled
            return

        kept = [self.spectra[i] for i in dialog.kept]
        reloaded, new_spectra = [], []
        try:
            # Every kept pair is read again from its files, keeping its position
            # and its solvent settings
            for position, index in enumerate(dialog.kept):
                if index in dialog.reloaded:
                    sample_file, reference_file = dialog.reloaded[index]
                    fresh = self._process_pair(position + 1, sample_file, reference_file)
                    fresh["solvent"] = kept[position].get("solvent")
                    kept[position] = fresh
                    reloaded.append(fresh)
            for offset, (sample_file, reference_file) in enumerate(dialog.result):
                number = len(kept) + offset + 1  # new pairs follow the largest number
                new_spectra.append(self._process_pair(number, sample_file, reference_file))
        except (OSError, ValueError) as err:
            # If any pair fails, nothing changes (removals and reloads included)
            messagebox.showerror("Could not load files", str(err), parent=self.view)
            return

        # Pairs are numbered by position: after a removal the later ones move up
        renamed = {}
        for number, s in enumerate(kept, start=1):
            new_label = f"Pair {number}: {Path(s['sample_file']).stem}"
            renamed[s["label"]] = new_label
            s["label"] = new_label
        self._skip_warned = {renamed[l] for l in self._skip_warned if l in renamed}

        self.spectra = kept + new_spectra
        self._warn_about_bad_points(reloaded + new_spectra)
        self._redraw()

    def clear_spectra(self):
        """Remove all pairs from the plot (the reference files stay set)."""
        self.spectra = []
        self._skip_warned.clear()
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

    # ------------------------------------------------------------------
    # Bulk solvent
    # ------------------------------------------------------------------
    def edit_solvent(self):
        """Open the dialog for each pair's solvent files, concentration and mole fraction."""
        if not self.spectra:
            messagebox.showinfo("Solvent settings", "Load at least one pair first.", parent=self.view)
            return
        rows = [(s["label"], s.get("solvent")) for s in self.spectra]
        dialog = SolventDialog(self.view, rows, self._validate_solvent)
        self.view.wait_window(dialog)
        if dialog.result is None:  # cancelled
            return
        for s, params in zip(self.spectra, dialog.result):
            s["solvent"] = params
        self._redraw()  # new numbers change the scale, so show the full view

    def _load_solvent_file(self, path):
        """Read a two-column solvent spectrum (cached until the file changes)."""
        modified = Path(path).stat().st_mtime
        cached = self._solvent_files.get(path)
        if cached is None or cached[0] != modified:
            x, y = models.load_two_column(path)
            self._solvent_files[path] = (modified, x, y)
            cached = self._solvent_files[path]
        return cached[1], cached[2]

    def _validate_solvent(self, result):
        """Called by the dialog: try every chosen file and its overlap with the spectrum.

        Raises ValueError (shown in the dialog, which stays open) if anything is wrong.
        """
        for s, params in zip(self.spectra, result):
            if params is None:
                continue
            try:
                self._convert_solvent(params, s["frequency"], np.zeros_like(s["frequency"]))
            except (OSError, ValueError) as err:
                raise ValueError(f"{s['label']}: {err}") from err

    def _convert_solvent(self, params, frequency, values):
        """Difference absorption coefficient -> molar absorption coefficient."""
        alpha_f, alpha_v = self._load_solvent_file(params["alpha_file"])
        eps_f, eps_v = self._load_solvent_file(params["epsilon_file"])
        return models.remove_bulk_solvent(
            frequency, values, alpha_f, alpha_v, eps_f, eps_v,
            params["concentration"], params["mole_fraction"],
        )

    def _remove_solvent(self, spectrum, frequency, values):
        try:
            return self._convert_solvent(spectrum["solvent"], frequency, values)
        except (OSError, ValueError) as err:
            raise ProcessingError("solvent", f"{spectrum['label']}: {err}") from err

    # ------------------------------------------------------------------
    # Singular value decomposition
    # ------------------------------------------------------------------
    def show_svd(self):
        """Open the SVD window for the spectra currently displayed."""
        if len(self.spectra) < 2:
            messagebox.showinfo("SVD", "Load at least two spectra first.", parent=self.view)
            return
        if self._svd_dialog is not None and self._svd_dialog.winfo_exists():
            self._svd_dialog.lift()
            self._svd_dialog.run()  # refresh with whatever is displayed now
            return
        self._svd_dialog = SVDDialog(
            self.view, self._run_svd, self._save_svd,
            state=self._svd_state, signature=self._svd_signature,
        )

    def _svd_signature(self):
        """Fingerprint of the displayed spectra: equal when nothing that matters changed
        (switching something and switching it back gives the same fingerprint)."""
        return self._fingerprint(*self._displayed_safe_pair())

    def _displayed_safe_pair(self):
        spectra, _, y_name = self._displayed_safe()
        return spectra, y_name

    @staticmethod
    def _fingerprint(spectra, y_name):
        import hashlib
        digest = hashlib.sha1(y_name.encode())
        for label, frequency, values in spectra:
            digest.update(str(label).encode())
            digest.update(np.ascontiguousarray(frequency, dtype=float).tobytes())
            digest.update(np.ascontiguousarray(values, dtype=float).tobytes())
        return digest.hexdigest()

    def _run_svd(self, low, high, center):
        """Called by the SVD window. Uses the spectra exactly as displayed."""
        spectra, _, y_name = self._displayed_safe()
        result = models.svd_spectra(spectra, (low, high), n_components=4, center=center)
        result["source"] = f"{len(spectra)} spectra, {y_name}"
        result["signature"] = self._fingerprint(spectra, y_name)
        return result

    def _save_svd(self, result):
        path = filedialog.asksaveasfilename(
            parent=self._svd_dialog or self.view,
            title="Save SVD results (three files are written)",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="svd.csv",
        )
        if not path:
            return
        try:
            files = models.save_svd(path, result)
        except OSError as err:
            messagebox.showerror("Could not save file", str(err), parent=self._svd_dialog)
            return
        messagebox.showinfo(
            "SVD saved",
            "Saved:\n" + "\n".join(Path(f).name for f in files),
            parent=self._svd_dialog,
        )

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
        have_thickness = self.cell_thickness_um is not None
        use_solvent = self.view.solvent_mode() and have_thickness
        # Solvent removal works on the absorption coefficient, so it implies the conversion
        use_coefficient = (self.view.coefficient_mode() or use_solvent) and have_thickness
        use_water = self.view.water_mode() and self.water is not None
        use_fringes = self.view.fringe_mode()

        if use_solvent:
            missing = [s["label"] for s in self.spectra if not s.get("solvent")]
            if missing:
                raise ProcessingError(
                    "solvent",
                    "No solvent settings for:\n" + "\n".join(missing)
                    + "\n\nEnter them with \"Solvent settings...\".",
                )

        self.fringe_skipped = []  # (label, reason) of spectra without clear fringes
        spectra = []
        for s in self.spectra:
            frequency, values = s["frequency"], s["delta_A"]

            if use_water:
                try:
                    values = models.correct_water_vapor(frequency, values, *self.water)
                except ValueError as err:
                    raise ProcessingError("water", f"{s['label']}: {err}") from err

            label = s["label"]
            if use_fringes:
                try:
                    # Residual water vapour lines are ignored when measuring the period
                    values, _ = models.remove_fringes(frequency, values, vapour=self.water)
                except ValueError as err:
                    # No clear fringes in this one: show it uncorrected, carry on
                    self.fringe_skipped.append((label, str(err)))
                    label += " [fringes not removed]"

            if use_coefficient:
                values = models.absorption_coefficient(values, self.cell_thickness_um)
            if use_solvent:
                values = self._remove_solvent(s, frequency, values)
            spectra.append((label, frequency, values))

        if use_fringes and self.spectra and len(self.fringe_skipped) == len(self.spectra):
            first = self.fringe_skipped[0]
            raise ProcessingError(
                "fringes", f"No spectrum has clear fringes.\n{first[0]}: {first[1]}"
            )

        # CSV column names say which corrections were applied
        if use_solvent:
            y_name = "epsilon_L_per_mol_cm"
        else:
            y_name = "alpha_per_cm" if use_coefficient else "delta_A"
        if use_water:
            y_name += "_no_water"
        if use_fringes:
            y_name += "_no_fringes"
        if use_solvent:
            ylabel = "Solute molar absorption coefficient (L mol⁻¹ cm⁻¹)"
        elif use_coefficient:
            ylabel = "Difference absorption coefficient (cm⁻¹)"
        else:
            ylabel = "Difference absorbance, ΔA"
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

    def _warn_about_skipped_fringes(self):
        """Tell the user once about spectra left uncorrected (not on every redraw)."""
        if not self.view.fringe_mode():
            self._skip_warned.clear()
        new = [(l, r) for l, r in self.fringe_skipped if l not in self._skip_warned]
        if not new:
            return
        self._skip_warned.update(l for l, _ in new)
        messagebox.showwarning(
            "Fringes not removed",
            "No clear fringes were found in these spectra, so they are shown "
            "uncorrected (the others were corrected):\n\n"
            + "\n".join(f"{label}: {reason}" for label, reason in new),
            parent=self.view,
        )

    def _redraw(self, keep_view=False):
        spectra, ylabel, _ = self._displayed_safe()
        self.view.plot_spectra(spectra, ylabel, keep_view=keep_view)
        self._warn_about_skipped_fringes()

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
            "sample_file": sample_file,
            "reference_file": reference_file,
            "solvent": None,  # set with "Solvent settings...": files, concentration, mole fraction
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
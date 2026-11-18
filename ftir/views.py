"""Complete replacement for ftir/views.py.

Main window:
  * Empty cell, Background and Water vapour file fields (Browse and clear buttons)
  * three switches beside the graph:
      - difference absorbance <-> absorption coefficient (needs empty cell + background)
      - fringes: original <-> etalon fringes removed
      - water vapour: original <-> water vapour removed (needs a water vapour file)
  * a label showing the gap thickness
  * a bulk solvent section: "Solvent settings..." button (SolventDialog) and a switch
  * an SVD button, and SVDDialog showing principal components, scores and % variation
  * an MCR button, and MCRDialog (multivariate curve resolution with known spectra)
  * plot_spectra(spectra, ylabel) so the controller chooses the y-axis label
"""

import re
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ftir import models

FILETYPES = [("Text files", "*.txt *.dat *.csv"), ("All files", "*.*")]


class PairRow(ctk.CTkFrame):
    """One sample + reference pair inside the dialog.

    A new pair starts empty. A pair that is already loaded (`loaded` is its
    (label, sample_path, reference_path)) is shaded and shows its files. Browse
    is always available: whatever files are in the fields when the dialog is
    confirmed are (re)loaded, in the same position for a loaded pair.
    """

    def __init__(self, parent, on_remove, on_copy, loaded=None):
        super().__init__(parent, fg_color=("gray85", "gray22") if loaded else None)
        self.loaded = loaded
        self.sample = tk.StringVar(value=loaded[1] if loaded else "")
        self.reference = tk.StringVar(value=loaded[2] if loaded else "")
        self._number = 0

        self.grid_columnconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(
            self, text=loaded[0] if loaded else "", font=ctk.CTkFont(weight="bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 0), sticky="w")
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=0, column=2, padx=10, pady=(8, 0), sticky="e")
        ctk.CTkButton(
            buttons, text="Copy reference to all pairs", width=180, fg_color="gray",
            command=lambda: on_copy(self),
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            buttons, text="Remove", width=70, fg_color="gray",
            command=lambda: on_remove(self),
        ).pack(side="left")

        for row, (label, var) in enumerate(
            (("Sample:", self.sample), ("Reference:", self.reference)), start=1
        ):
            ctk.CTkLabel(self, text=label).grid(row=row, column=0, padx=(10, 5), pady=4, sticky="w")
            ctk.CTkEntry(self, textvariable=var, state="readonly").grid(
                row=row, column=1, padx=5, pady=4, sticky="ew"
            )
            ctk.CTkButton(
                self, text="Browse...", width=90,
                command=lambda v=var, name=label: self._browse(v, name),
            ).grid(row=row, column=2, padx=(5, 10), pady=4)

    def set_number(self, number):
        """Show this row's current position: Pair N (loaded pairs add their file name)."""
        self._number = number
        self._update_title()

    def _update_title(self):
        if self.loaded is None:
            self.title_label.configure(text=f"Pair {self._number}")
            return
        name = re.sub(r"^Pair\s+\d+:\s*", "", self.loaded[0])
        self.title_label.configure(text=f"Pair {self._number}: {name}")

    def _browse(self, variable, name):
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title=f"Select {name.rstrip(':').lower()} file",
            filetypes=FILETYPES,
        )
        if path:  # empty string means the user cancelled the file picker
            variable.set(path)
            self._update_title()


class LoadPairsDialog(ctk.CTkToplevel):
    """Modal dialog for adding new sample/reference pairs and removing loaded ones.

    Rows are numbered 1, 2, 3... by position: removing a pair renumbers the
    rows after it, and new pairs come after the largest number.
    existing: list of (label, sample_path, reference_path) for the pairs
    currently loaded. They are listed first (shaded, read-only); press Remove
    on any you want taken off the plot.

    After it closes (OK):
      result    list of (sample_path, reference_path) for the NEW pairs
      kept      indices (into `existing`) of the loaded pairs that stay
      reloaded  {index into `existing`: (sample_path, reference_path)} for every
                kept pair: the files now in its fields, to be read again in the
                same position among the pairs
    If the user cancelled, result is None.
    """

    def __init__(self, parent, existing=None):
        super().__init__(parent)
        self.title("Load / remove sample/reference pairs")
        self.geometry("700x560")
        self.transient(parent)
        self.after(100, self.grab_set)  # make modal (delay avoids a macOS glitch)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.result = None
        self.kept = []
        self.reloaded = {}
        self.existing = list(existing or [])
        self.rows = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hint = (
            "Loaded pairs are shaded. Remove the ones you no longer want, change a pair's files with Browse "
            "(all pairs are reloaded from the files shown when you press OK), and add new pairs below."
            if self.existing else "Choose a sample and a reference file for each pair."
        )
        ctk.CTkLabel(self, text=hint, anchor="w", justify="left").grid(
            row=0, column=0, padx=15, pady=(12, 0), sticky="w"
        )

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.grid(row=1, column=0, padx=15, pady=(8, 5), sticky="nsew")

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        ctk.CTkButton(bottom, text="+ Add pair", command=self._add_row).pack(side="left")
        ctk.CTkButton(
            bottom, text="Cancel", width=100, fg_color="gray", command=self._cancel
        ).pack(side="right", padx=(10, 0))
        ctk.CTkButton(bottom, text="OK", width=100, command=self._ok).pack(side="right")

        for item in self.existing:
            self._add_row(loaded=item)
        self._add_row()  # one empty new pair, ready to fill in

    def _add_row(self, loaded=None):
        row = PairRow(self.scroll, self._remove_row, self._copy_reference, loaded=loaded)
        row.pack(fill="x", padx=5, pady=5)
        self.rows.append(row)
        self._renumber()

    def _copy_reference(self, source):
        """Use this pair's reference file for every pair (loaded and new)."""
        reference = source.reference.get()
        if not reference:
            messagebox.showinfo(
                "Copy reference", "Choose a reference file for this pair first.", parent=self
            )
            return
        for row in self.rows:
            row.reference.set(reference)

    def _remove_row(self, row):
        self.rows.remove(row)
        row.destroy()
        self._renumber()

    def _renumber(self):
        # Pairs are numbered by position, so removing one closes the gap
        for i, row in enumerate(self.rows, start=1):
            row.set_number(i)

    def _ok(self):
        new_pairs = []
        for row in (r for r in self.rows if r.loaded is None):
            sample, reference = row.sample.get(), row.reference.get()
            if bool(sample) != bool(reference):  # only one of the two chosen
                messagebox.showwarning(
                    "Incomplete pair",
                    f"{row.title_label.cget('text')} needs both a sample and a reference file.",
                    parent=self,
                )
                return
            if sample and reference:  # an untouched empty row is simply ignored
                new_pairs.append((sample, reference))

        if not new_pairs and not self.existing:
            messagebox.showwarning(
                "No files", "Select at least one sample/reference pair.", parent=self
            )
            return

        self.kept = [self.existing.index(r.loaded) for r in self.rows if r.loaded is not None]
        self.reloaded = {
            self.existing.index(r.loaded): (r.sample.get(), r.reference.get())
            for r in self.rows
            if r.loaded is not None
        }
        self.result = new_pairs
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class SolventRow(ctk.CTkFrame):
    """The solvent settings of one pair inside SolventDialog."""

    def __init__(self, parent, label, params, on_copy):
        super().__init__(parent)
        params = params or {}
        self.label = label
        self.alpha_file = tk.StringVar(value=params.get("alpha_file", ""))
        self.epsilon_file = tk.StringVar(value=params.get("epsilon_file", ""))
        self.concentration = tk.StringVar(
            value=f"{params['concentration']:g}" if "concentration" in params else ""
        )
        self.mole_fraction = tk.StringVar(
            value=f"{params['mole_fraction']:g}" if "mole_fraction" in params else ""
        )
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text=label, font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=(8, 0), sticky="w"
        )
        ctk.CTkButton(
            self, text="Copy files to all pairs", width=150, fg_color="gray",
            command=lambda: on_copy(self),
        ).grid(row=0, column=2, padx=10, pady=(8, 0), sticky="e")

        files = (
            ("Solvent absorption α (cm⁻¹):", self.alpha_file),
            ("Solvent molar absorption ε (L mol⁻¹ cm⁻¹):", self.epsilon_file),
        )
        for row, (text, var) in enumerate(files, start=1):
            ctk.CTkLabel(self, text=text).grid(row=row, column=0, padx=(10, 5), pady=3, sticky="w")
            ctk.CTkEntry(self, textvariable=var, state="readonly").grid(
                row=row, column=1, padx=5, pady=3, sticky="ew"
            )
            ctk.CTkButton(
                self, text="Browse...", width=90,
                command=lambda v=var, name=text: self._browse(v, name),
            ).grid(row=row, column=2, padx=(5, 10), pady=3)

        numbers = ctk.CTkFrame(self, fg_color="transparent")
        numbers.grid(row=3, column=0, columnspan=3, padx=10, pady=(3, 8), sticky="w")
        ctk.CTkLabel(numbers, text="Total concentration (mol/L):").pack(side="left")
        ctk.CTkEntry(numbers, textvariable=self.concentration, width=90).pack(
            side="left", padx=(6, 20)
        )
        ctk.CTkLabel(numbers, text="Solvent mole fraction:").pack(side="left")
        ctk.CTkEntry(numbers, textvariable=self.mole_fraction, width=90).pack(
            side="left", padx=(6, 0)
        )

    def _browse(self, variable, name):
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title=f"Select {name.rstrip(':')}",
            filetypes=FILETYPES,
        )
        if path:
            variable.set(path)

    def read(self):
        """Return None if all four fields are empty, else a dict of the settings.

        Raises ValueError with a readable message for missing or invalid entries.
        """
        texts = [
            self.alpha_file.get(), self.epsilon_file.get(),
            self.concentration.get().strip(), self.mole_fraction.get().strip(),
        ]
        if not any(texts):
            return None
        if not all(texts):
            raise ValueError(
                f"{self.label}: fill in all four fields, or clear all four to leave "
                "this pair without solvent settings."
            )
        try:
            concentration = float(texts[2])
            mole_fraction = float(texts[3])
        except ValueError:
            raise ValueError(
                f"{self.label}: the concentration and mole fraction must be numbers."
            ) from None
        if not concentration > 0:
            raise ValueError(f"{self.label}: the total concentration must be positive.")
        if not 0.0 <= mole_fraction < 1.0:
            raise ValueError(
                f"{self.label}: the solvent mole fraction must be at least 0 and less than 1."
            )
        return {
            "alpha_file": texts[0],
            "epsilon_file": texts[1],
            "concentration": concentration,
            "mole_fraction": mole_fraction,
        }


class SolventDialog(ctk.CTkToplevel):
    """Modal dialog for the bulk-solvent settings of every loaded pair.

    rows: list of (label, params) where params is None or a dict with
        alpha_file, epsilon_file, concentration, mole_fraction.
    validate(result) is called on OK; it raises ValueError (with a message for
        the user) if something is wrong, and the dialog stays open.
    After it closes, `result` is a list with one entry per row (a dict, or None
    for a pair left empty), or None if the user cancelled.
    """

    def __init__(self, parent, rows, validate):
        super().__init__(parent)
        self.title("Bulk solvent settings")
        self.geometry("760x600")
        self.transient(parent)
        self.after(100, self.grab_set)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self._validate = validate
        self.result = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text=(
                "For each pair: the solvent's absorption coefficient and molar absorption "
                "coefficient at the measurement temperature (two-column files), the "
                "total concentration of the solution (total moles × density ÷ mass, in mol/L) "
                "and the solvent mole fraction (moles of solvent ÷ total moles). "
                "The same file can be used for several pairs."
            ),
            wraplength=720, justify="left", anchor="w",
        ).grid(row=0, column=0, padx=15, pady=(12, 0), sticky="w")

        scroll = ctk.CTkScrollableFrame(self)
        scroll.grid(row=1, column=0, padx=15, pady=8, sticky="nsew")
        self.rows = []
        for label, params in rows:
            row = SolventRow(scroll, label, params, self._copy_files)
            row.pack(fill="x", padx=5, pady=5)
            self.rows.append(row)

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        ctk.CTkButton(
            bottom, text="Cancel", width=100, fg_color="gray", command=self._cancel
        ).pack(side="right", padx=(10, 0))
        ctk.CTkButton(bottom, text="OK", width=100, command=self._ok).pack(side="right")

    def _copy_files(self, source):
        """Use this pair's two solvent files for every pair."""
        for row in self.rows:
            row.alpha_file.set(source.alpha_file.get())
            row.epsilon_file.set(source.epsilon_file.get())

    def _ok(self):
        try:
            result = [row.read() for row in self.rows]
            self._validate(result)
        except ValueError as err:
            messagebox.showerror("Bulk solvent settings", str(err), parent=self)
            return
        self.result = result
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class SVDDialog(ctk.CTkToplevel):
    """Window showing the first principal components and their scores.

    compute(low, high, center) -> result dict (see models.svd_spectra); it
        raises ValueError with a message the user can read.
    save(result) is called by the Save button.
    The window is not modal, so the main window stays usable; press Run again
    after changing switches or loading spectra.
    """

    def __init__(self, parent, compute, save, state=None, signature=None):
        super().__init__(parent)
        self.title("Singular value decomposition")
        self.geometry("1100x640")
        self.transient(parent)
        self._compute = compute
        self._save = save
        self.result = None   # as computed
        self.flipped = set() # indices of components whose sign is flipped
        # `state` is a dict owned by the controller: it is filled when this window
        # closes and used by the next window to restore the settings and, if the
        # displayed data are unchanged (`signature()` equal), the results too.
        self.memory = state if state is not None else {}

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, padx=15, pady=10, sticky="w")
        ctk.CTkLabel(bar, text="Wavenumber range (cm\u207b\u00b9):").pack(side="left")
        self.low_var, self.high_var = tk.StringVar(), tk.StringVar()
        ctk.CTkEntry(bar, textvariable=self.low_var, width=70, placeholder_text="from").pack(
            side="left", padx=(8, 4)
        )
        ctk.CTkLabel(bar, text="to").pack(side="left")
        ctk.CTkEntry(bar, textvariable=self.high_var, width=70, placeholder_text="to").pack(
            side="left", padx=(4, 12)
        )
        self.center_switch = ctk.CTkSwitch(bar, text="Subtract mean spectrum")
        self.center_switch.pack(side="left", padx=(0, 12))
        ctk.CTkButton(bar, text="Run", width=80, command=self.run).pack(side="left")
        ctk.CTkButton(bar, text="Save results...", width=120, command=self._on_save).pack(
            side="left", padx=(10, 0)
        )

        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=15, sticky="nsew")
        self.flip_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.flip_bar.grid(row=2, column=0, padx=15, pady=(5, 0), sticky="w")
        self.info = ctk.CTkLabel(self, text="", justify="left", anchor="w")
        self.info.grid(row=3, column=0, padx=15, pady=(5, 12), sticky="ew")

        self.protocol("WM_DELETE_WINDOW", self._close)
        saved = self.memory
        self.low_var.set(saved.get("low", ""))
        self.high_var.set(saved.get("high", ""))
        if saved.get("center"):
            self.center_switch.select()
        cached = saved.get("result")
        if cached is not None and signature is not None and cached.get("signature") == signature():
            self.result = cached
            self.flipped = set(saved.get("flipped", ()))
            self._build_flip_boxes()
            self._draw()
        else:
            self.after(100, self.run)

    def _close(self):
        self.memory.update(
            low=self.low_var.get(),
            high=self.high_var.get(),
            center=bool(self.center_switch.get()),
            result=self.result,
            flipped=set(self.flipped),
        )
        self.destroy()

    def _limit(self, variable, name):
        text = variable.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a number (wavenumber {name}).") from None

    def run(self):
        try:
            low, high = self._limit(self.low_var, "from"), self._limit(self.high_var, "to")
            self.result = self._compute(low, high, bool(self.center_switch.get()))
        except ValueError as err:
            messagebox.showerror("SVD", str(err), parent=self)
            return
        self._build_flip_boxes()
        self._draw()

    def _build_flip_boxes(self):
        """One "Flip PCn" checkbox per component (kept between runs)."""
        for child in self.flip_bar.winfo_children():
            child.destroy()
        k = self.result["components"].shape[0]
        ctk.CTkLabel(self.flip_bar, text="Flip sign (component and scores):").pack(
            side="left", padx=(0, 10)
        )
        for i in range(k):
            box = ctk.CTkCheckBox(
                self.flip_bar, text=f"PC{i + 1}", width=70,
                command=lambda i=i: self._toggle_flip(i),
            )
            if i in self.flipped:
                box.select()
            box.pack(side="left", padx=(0, 8))

    def _toggle_flip(self, i):
        self.flipped ^= {i}
        self._draw()

    def _signed(self):
        """The result with the chosen components (and their scores) sign-flipped."""
        r = dict(self.result)
        sign = np.array(
            [-1.0 if i in self.flipped else 1.0 for i in range(r["components"].shape[0])]
        )
        r["components"] = r["components"] * sign[:, None]
        r["scores"] = r["scores"] * sign[None, :]
        return r

    def _on_save(self):
        if self.result is None:
            messagebox.showinfo("SVD", "Nothing to save yet.", parent=self)
            return
        self._save(self._signed())  # saved with the signs as shown

    def _draw(self):
        r = self._signed()
        k = r["components"].shape[0]
        n = len(r["labels"])
        self.figure.clear()
        axes = self.figure.subplots(2, k, squeeze=False)
        numbers = list(range(1, n + 1))
        short_labels = models.distinguishing_labels(r["labels"])  # only what differs
        for i in range(k):
            top, bottom = axes[0, i], axes[1, i]
            top.plot(r["wavenumber"], r["components"][i])
            top.set_title(f"PC{i + 1} ({r['explained'][i]:.2f}%)", fontsize=10)
            top.set_xlabel("Wavenumber (cm\u207b\u00b9)", fontsize=8)
            if i == 0:
                top.set_ylabel("Component (a.u.)", fontsize=8)
            bottom.plot(numbers, r["scores"][:, i], marker="o", linestyle="-")
            bottom.set_xlabel("Spectrum", fontsize=8)
            if i == 0:
                bottom.set_ylabel("Score (a.u.)", fontsize=8)
            if n <= 20:
                bottom.set_xticks(numbers)
                bottom.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
            for ax in (top, bottom):
                ax.tick_params(labelsize=8)
        self.figure.tight_layout()
        self.canvas.draw()

        shown = float(r["explained"][:k].sum())
        text = "   ".join(f"PC{i + 1}: {r['explained'][i]:.2f}%" for i in range(k))
        extra = " (mean spectrum subtracted)" if r["centered"] else ""
        self.info.configure(
            text=f"Variation explained: {text}   |   first {k}: {shown:.2f}%{extra}   |   {r['source']}"
        )


class MCRDialog(ctk.CTkToplevel):
    """Window for multivariate curve resolution (MCR-ALS) of the displayed spectra.

    compute(low, high, known, n_unknown, nonneg_scores, nonneg_spectra) -> result
        dict (see models.mcr_spectra); it raises ValueError with a readable message.
    pick_known() -> list of (label, wavenumbers, values) chosen by the user, or [].
        It raises ValueError for a file that cannot be read.
    save(result) is called by the Save button.
    `state` (a dict owned by the controller) keeps the settings and results when
    the window is closed; they come back when it is reopened, and the results are
    shown again without recomputing if `signature()` (a fingerprint of the
    displayed spectra) and all the settings are unchanged.
    """

    def __init__(self, parent, compute, pick_known, save, state=None, signature=None):
        super().__init__(parent)
        self.title("Multivariate curve resolution")
        self.geometry("1150x700")
        self.transient(parent)
        self._compute = compute
        self._pick_known = pick_known
        self._save = save
        self.memory = state if state is not None else {}
        self.known = list(self.memory.get("known", []))
        self.result = None
        self.run_settings = None
        self.flipped = set()   # indices of unknown components whose sign is flipped

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, padx=15, pady=(10, 0), sticky="w")
        ctk.CTkLabel(bar, text="Wavenumber range (cm\u207b\u00b9):").pack(side="left")
        self.low_var, self.high_var = tk.StringVar(), tk.StringVar()
        ctk.CTkEntry(bar, textvariable=self.low_var, width=70, placeholder_text="from").pack(
            side="left", padx=(8, 4)
        )
        ctk.CTkLabel(bar, text="to").pack(side="left")
        ctk.CTkEntry(bar, textvariable=self.high_var, width=70, placeholder_text="to").pack(
            side="left", padx=(4, 14)
        )
        ctk.CTkLabel(bar, text="Unknown components:").pack(side="left")
        self.unknown_var = tk.StringVar(value="1")
        ctk.CTkEntry(bar, textvariable=self.unknown_var, width=40).pack(side="left", padx=(8, 14))
        self.scores_switch = ctk.CTkSwitch(bar, text="Scores \u2265 0")
        self.scores_switch.pack(side="left", padx=(0, 10))
        self.spectra_switch = ctk.CTkSwitch(bar, text="Unknown spectra \u2265 0")
        self.spectra_switch.pack(side="left", padx=(0, 14))
        ctk.CTkButton(bar, text="Run", width=80, command=self.run).pack(side="left")
        ctk.CTkButton(bar, text="Save results...", width=120, command=self._on_save).pack(
            side="left", padx=(10, 0)
        )

        self.known_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.known_bar.grid(row=1, column=0, padx=15, pady=(8, 0), sticky="ew")

        view_bar = ctk.CTkFrame(self, fg_color="transparent")
        view_bar.grid(row=2, column=0, padx=15, pady=(8, 0), sticky="w")
        self.view_buttons = ctk.CTkSegmentedButton(
            view_bar, values=["Components", "Residuals"], command=lambda _: self._draw()
        )
        self.view_buttons.set(self.memory.get("view", "Components"))
        self.view_buttons.pack(side="left")

        self.figure = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=3, column=0, padx=15, pady=(6, 0), sticky="nsew")
        self.flip_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.flip_bar.grid(row=4, column=0, padx=15, pady=(5, 0), sticky="w")
        self.info = ctk.CTkLabel(self, text="", justify="left", anchor="w", wraplength=1080)
        self.info.grid(row=5, column=0, padx=15, pady=(5, 12), sticky="ew")

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.low_var.set(self.memory.get("low", ""))
        self.high_var.set(self.memory.get("high", ""))
        self.unknown_var.set(self.memory.get("n_unknown", "1"))
        if self.memory.get("nonneg_scores"):
            self.scores_switch.select()
        if self.memory.get("nonneg_spectra"):
            self.spectra_switch.select()
        self._build_known_bar()

        cached = self.memory.get("result")
        if (
            cached is not None
            and signature is not None
            and cached.get("signature") == signature()
            and self.memory.get("run_settings") == self._settings()
        ):
            self.result = cached
            self.run_settings = self.memory["run_settings"]
            self.flipped = set(self.memory.get("flipped", ()))
            self._build_flip_boxes()
            self._draw()
        else:
            self.info.configure(
                text="Add known spectra (optional), choose the number of unknown components "
                "and press Run."
            )

    # ---------------- settings ----------------
    def _limit(self, variable, name):
        text = variable.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a number (wavenumber {name}).") from None

    def _n_unknown(self):
        text = self.unknown_var.get().strip()
        try:
            number = int(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a whole number (unknown components).") from None
        if number < 0:
            raise ValueError("The number of unknown components cannot be negative.")
        return number

    def _settings(self):
        """Everything that affects the result, in a form that can be compared."""
        known = tuple(
            (label, len(x), float(x[0]), float(x[-1]), float(np.nansum(y)))
            for label, x, y in self.known
        )
        return (
            self.low_var.get().strip(), self.high_var.get().strip(),
            self.unknown_var.get().strip(),
            bool(self.scores_switch.get()), bool(self.spectra_switch.get()), known,
        )

    # ---------------- known spectra ----------------
    def _build_known_bar(self):
        for child in self.known_bar.winfo_children():
            child.destroy()
        ctk.CTkButton(self.known_bar, text="Add known spectra...", width=150,
                      command=self._add_known).pack(side="left")
        if not self.known:
            ctk.CTkLabel(self.known_bar, text="  none (all components are unknown)").pack(side="left")
            return
        for index, (label, _, _) in enumerate(self.known):
            chip = ctk.CTkFrame(self.known_bar)
            chip.pack(side="left", padx=(8, 0))
            shown = label if len(label) <= 34 else label[:31] + "..."
            ctk.CTkLabel(chip, text=shown).pack(side="left", padx=(8, 2))
            ctk.CTkButton(chip, text="\u00d7", width=24, fg_color="gray",
                          command=lambda i=index: self._remove_known(i)).pack(side="left", padx=(0, 4), pady=3)

    def _add_known(self):
        try:
            added = self._pick_known()
        except ValueError as err:
            messagebox.showerror("Known spectra", str(err), parent=self)
            return
        self.known.extend(added)
        self._build_known_bar()

    def _remove_known(self, index):
        del self.known[index]
        self._build_known_bar()

    # ---------------- running ----------------
    def run(self):
        try:
            low, high = self._limit(self.low_var, "from"), self._limit(self.high_var, "to")
            number = self._n_unknown()
            settings = self._settings()
            self.result = self._compute(
                low, high, list(self.known), number,
                bool(self.scores_switch.get()), bool(self.spectra_switch.get()),
            )
        except ValueError as err:
            messagebox.showerror("MCR", str(err), parent=self)
            return
        self.run_settings = settings
        # keep earlier flips that still refer to an unknown component
        self.flipped = {i for i in self.flipped if self._can_flip(i)}
        self._build_flip_boxes()
        self._draw()

    # ---------------- sign flips ----------------
    def _flip_allowed(self):
        """Flipping a component would break a non-negativity constraint, so it is off then."""
        r = self.result
        return r is not None and not (r["nonneg_scores"] or r["nonneg_spectra"])

    def _can_flip(self, i):
        return (
            self._flip_allowed()
            and i < len(self.result["is_known"])
            and not self.result["is_known"][i]
        )

    def _build_flip_boxes(self):
        """One "Flip" checkbox per UNKNOWN component (a known one has a meaningful sign)."""
        for child in self.flip_bar.winfo_children():
            child.destroy()
        r = self.result
        if r is None or all(r["is_known"]):
            return
        if not self._flip_allowed():
            ctk.CTkLabel(
                self.flip_bar,
                text="Sign flip is unavailable while a non-negativity constraint is on.",
            ).pack(side="left")
            return
        ctk.CTkLabel(self.flip_bar, text="Flip sign (spectrum and contribution):").pack(
            side="left", padx=(0, 10)
        )
        for i, known in enumerate(r["is_known"]):
            if known:
                continue
            box = ctk.CTkCheckBox(
                self.flip_bar, text=r["component_labels"][i], width=90,
                command=lambda i=i: self._toggle_flip(i),
            )
            if i in self.flipped:
                box.select()
            box.pack(side="left", padx=(0, 8))

    def _toggle_flip(self, i):
        self.flipped ^= {i}
        self._draw()

    def _signed(self):
        """The result with the chosen unknown components (and their scores) sign-flipped.
        The fit and residuals do not change: flipping both factors leaves the product alone."""
        r = dict(self.result)
        sign = np.array(
            [-1.0 if i in self.flipped else 1.0 for i in range(r["components"].shape[0])]
        )
        r["components"] = r["components"] * sign[:, None]
        r["scores"] = r["scores"] * sign[None, :]
        return r

    def _on_save(self):
        if self.result is None:
            messagebox.showinfo("MCR", "Nothing to save yet. Press Run first.", parent=self)
            return
        self._save(self._signed())  # saved with the signs as shown

    def _close(self):
        self.memory.update(
            low=self.low_var.get(), high=self.high_var.get(),
            n_unknown=self.unknown_var.get(),
            nonneg_scores=bool(self.scores_switch.get()),
            nonneg_spectra=bool(self.spectra_switch.get()),
            known=list(self.known), view=self.view_buttons.get(),
            result=self.result, run_settings=self.run_settings,
            flipped=set(self.flipped),
        )
        self.destroy()

    # ---------------- drawing ----------------
    def _draw(self):
        if self.result is None:
            return
        self.figure.clear()
        if self.view_buttons.get() == "Residuals":
            self._draw_residuals()
        else:
            self._draw_components()
        self.canvas.draw()
        r = self.result
        state = "converged" if r["converged"] else "stopped at the iteration limit"
        constraints = [name for name, on in (("scores \u2265 0", r["nonneg_scores"]),
                                             ("unknown spectra \u2265 0", r["nonneg_spectra"])) if on]
        self.info.configure(
            text=f"Lack of fit {r['total_lack_of_fit']:.2f}% of the data norm   |   "
            f"{r['n_iterations']} iterations ({state})   |   "
            f"{len(r['labels'])} spectra, {len(r['component_labels'])} components"
            + (f"   |   constraints: {', '.join(constraints)}" if constraints else "   |   no constraints")
            + f"   |   {r.get('source', '')}"
        )

    def _short_labels(self):
        return models.distinguishing_labels(self.result["labels"])

    def _draw_components(self):
        import textwrap

        r = self._signed()
        k = r["components"].shape[0]
        n = len(r["labels"])
        axes = self.figure.subplots(2, k, squeeze=False)
        numbers = list(range(1, n + 1))
        short = self._short_labels()
        for i in range(k):
            top, bottom = axes[0, i], axes[1, i]
            top.plot(r["wavenumber"], r["components"][i], color="C0" if r["is_known"][i] else "C3")
            top.set_title(textwrap.fill(r["component_labels"][i], 26), fontsize=9)
            top.set_xlabel("Wavenumber (cm\u207b\u00b9)", fontsize=8)
            bottom.plot(numbers, r["scores"][:, i], marker="o", linestyle="-",
                        color="C0" if r["is_known"][i] else "C3")
            bottom.set_xlabel("Spectrum", fontsize=8)
            if i == 0:
                top.set_ylabel("Component (a.u.)", fontsize=8)
                bottom.set_ylabel("Contribution", fontsize=8)
            if n <= 20:
                bottom.set_xticks(numbers)
                bottom.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            for ax in (top, bottom):
                ax.tick_params(labelsize=8)
        self.figure.tight_layout()

    def _draw_residuals(self):
        r = self.result
        n = len(r["labels"])
        numbers = np.arange(1, n + 1)
        short = self._short_labels()
        axes = self.figure.subplots(2, 2)
        ax = axes[0, 0]
        for row, name in zip(r["residuals"], short):
            ax.plot(r["wavenumber"], row, label=name, linewidth=1)
        ax.set_title("Residual spectra (measured \u2212 fitted)", fontsize=9)
        ax.set_xlabel("Wavenumber (cm\u207b\u00b9)", fontsize=8)
        if n <= 8:
            ax.legend(fontsize=6)
        ax = axes[0, 1]
        ax.plot(numbers, r["residual_norm"], marker="o")
        ax.set_title("Residual norm per spectrum", fontsize=9)
        ax.set_xlabel("Spectrum", fontsize=8)
        if n <= 20:
            ax.set_xticks(numbers)
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax = axes[1, 0]
        limit = float(np.max(np.abs(r["residuals"]))) or 1.0
        image = ax.imshow(
            r["residuals"], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
            extent=[r["wavenumber"][0], r["wavenumber"][-1], n + 0.5, 0.5],
        )
        self.figure.colorbar(image, ax=ax, label="Residual")
        ax.set_title("Residual heat map", fontsize=9)
        ax.set_xlabel("Wavenumber (cm\u207b\u00b9)", fontsize=8)
        ax.set_ylabel("Spectrum", fontsize=8)
        ax = axes[1, 1]
        ax.plot(r["wavenumber"], r["residuals"].mean(axis=0))
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_title("Mean residual (should be ~0)", fontsize=9)
        ax.set_xlabel("Wavenumber (cm\u207b\u00b9)", fontsize=8)
        for a in axes.ravel():
            a.tick_params(labelsize=8)
        self.figure.tight_layout()


class View(ctk.CTk):
    """Main window: buttons, reference file fields, plot, switches and toolbar."""

    DEFAULT_YLABEL = "Difference absorbance, \u0394A"

    def __init__(self):
        super().__init__()
        self.title("FTIR THz analysis")
        self.geometry("1050x800")

        self.grid_columnconfigure(0, weight=1)  # plot column stretches
        self.grid_rowconfigure(2, weight=1)     # plot row stretches

        # Row 0: action buttons (the controller attaches their commands)
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, padx=15, pady=15, sticky="w")
        self.load_button = ctk.CTkButton(bar, text="Load files...")
        self.load_button.pack(side="left", padx=(0, 10))
        self.save_button = ctk.CTkButton(bar, text="Save CSV...")
        self.save_button.pack(side="left")
        self.clear_button = ctk.CTkButton(bar, text="Clear", fg_color="gray")
        self.clear_button.pack(side="left", padx=(10, 0))
        self.svd_button = ctk.CTkButton(bar, text="SVD...")
        self.svd_button.pack(side="left", padx=(10, 0))
        self.mcr_button = ctk.CTkButton(bar, text="MCR...")
        self.mcr_button.pack(side="left", padx=(10, 0))

        # Row 1: empty cell, background and water vapour files
        cells = ctk.CTkFrame(self)
        cells.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 10), sticky="ew")
        cells.grid_columnconfigure(1, weight=1)
        self.empty_var = tk.StringVar()
        self.background_var = tk.StringVar()
        self.water_var = tk.StringVar()
        self.empty_button, self.empty_clear = self._add_file_row(
            cells, 0, "Empty cell:", self.empty_var
        )
        self.background_button, self.background_clear = self._add_file_row(
            cells, 1, "Background:", self.background_var
        )
        self.water_button, self.water_clear = self._add_file_row(
            cells, 2, "Water vapour:", self.water_var
        )

        # Row 2, column 0: embedded matplotlib figure
        self._ylabel = self.DEFAULT_YLABEL
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=2, column=0, padx=(15, 5), pady=(0, 5), sticky="nsew")

        # Row 2, column 1: side panel with the switches
        panel = ctk.CTkFrame(self)
        panel.grid(row=2, column=1, padx=(0, 15), pady=(0, 5), sticky="n")
        bold = ctk.CTkFont(weight="bold")

        ctk.CTkLabel(panel, text="Display", font=bold).pack(padx=15, pady=(15, 5))
        self.display_switch = ctk.CTkSwitch(panel, text="Absorption coeff.", state="disabled")
        self.display_switch.pack(padx=15, pady=5, anchor="w")
        ctk.CTkLabel(
            panel, text="Off: \u0394A\nOn: \u03b1 (cm\u207b\u00b9)", justify="left"
        ).pack(padx=15, pady=(0, 5), anchor="w")
        self.thickness_label = ctk.CTkLabel(panel, text="Gap thickness: \u2014")
        self.thickness_label.pack(padx=15, pady=(0, 10), anchor="w")

        ctk.CTkLabel(panel, text="Corrections", font=bold).pack(padx=15, pady=(10, 5))
        self.water_switch = ctk.CTkSwitch(panel, text="Water vapour removed", state="disabled")
        self.water_switch.pack(padx=15, pady=5, anchor="w")
        ctk.CTkLabel(
            panel, text="Needs a water vapour file", justify="left", text_color="gray"
        ).pack(padx=15, pady=(0, 5), anchor="w")
        self.fringe_switch = ctk.CTkSwitch(panel, text="Fringes removed")
        self.fringe_switch.pack(padx=15, pady=5, anchor="w")
        ctk.CTkLabel(
            panel, text="Diamond window etalon\n(half-period average)",
            justify="left", text_color="gray",
        ).pack(padx=15, pady=(0, 10), anchor="w")

        ctk.CTkLabel(panel, text="Bulk solvent", font=bold).pack(padx=15, pady=(10, 5))
        self.solvent_button = ctk.CTkButton(panel, text="Solvent settings...")
        self.solvent_button.pack(padx=15, pady=(0, 5), anchor="w")
        self.solvent_switch = ctk.CTkSwitch(panel, text="Solvent removed", state="disabled")
        self.solvent_switch.pack(padx=15, pady=5, anchor="w")
        ctk.CTkLabel(
            panel,
            text="Needs empty cell + background\nand settings for every pair.\nShows molar absorption\ncoefficient (L mol\u207b\u00b9 cm\u207b\u00b9)",
            justify="left", text_color="gray",
        ).pack(padx=15, pady=(0, 15), anchor="w")

        # Row 3: zoom/pan toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.grid(row=3, column=0, padx=15, pady=(0, 15), sticky="w")

        # Mouse-wheel zoom around the cursor
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _add_file_row(self, parent, row, label, variable):
        """One read-only path field with Browse and clear (X) buttons."""
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, padx=(10, 5), pady=5, sticky="w")
        ctk.CTkEntry(parent, textvariable=variable, state="readonly").grid(
            row=row, column=1, padx=5, pady=5, sticky="ew"
        )
        browse = ctk.CTkButton(parent, text="Browse...", width=90)
        browse.grid(row=row, column=2, padx=5, pady=5)
        clear = ctk.CTkButton(parent, text="\u2715", width=32, fg_color="gray")
        clear.grid(row=row, column=3, padx=(0, 10), pady=5)
        return browse, clear

    def _style_axes(self):
        self.ax.set_xlabel("Wavenumber (cm\u207b\u00b9)")  # change if your axis differs
        self.ax.set_ylabel(self._ylabel)

    # ------------------------------------------------------------------
    # Methods the controller calls
    # ------------------------------------------------------------------
    def coefficient_mode(self):
        """True when the switch is on (show absorption coefficient)."""
        return bool(self.display_switch.get())

    def set_cell_thickness(self, thickness_um):
        """Show the gap thickness and enable the switch, or disable it if None."""
        if thickness_um is None:
            self.thickness_label.configure(text="Gap thickness: \u2014")
            self.display_switch.deselect()
            self.display_switch.configure(state="disabled")
            self.solvent_switch.deselect()
            self.solvent_switch.configure(state="disabled")
        else:
            self.thickness_label.configure(text=f"Gap thickness: {thickness_um:.1f} \u00b5m")
            self.display_switch.configure(state="normal")
            self.solvent_switch.configure(state="normal")

    def water_mode(self):
        """True when the water vapour switch is on (show water vapour removed)."""
        return bool(self.water_switch.get())

    def solvent_mode(self):
        """True when the solvent switch is on (show the molar absorption coefficient)."""
        return bool(self.solvent_switch.get())

    def fringe_mode(self):
        """True when the fringe switch is on (show fringes removed)."""
        return bool(self.fringe_switch.get())

    def set_water_available(self, available):
        """Enable the water vapour switch once a valid file is loaded, else disable it."""
        if available:
            self.water_switch.configure(state="normal")
        else:
            self.water_switch.deselect()
            self.water_switch.configure(state="disabled")

    def reset_switch(self, name):
        """Switch a correction back off (used when a correction fails)."""
        {
            "water": self.water_switch,
            "fringes": self.fringe_switch,
            "solvent": self.solvent_switch,
        }[name].deselect()

    def plot_spectra(self, spectra, ylabel=DEFAULT_YLABEL, keep_view=False):
        """Plot several spectra, each in its own colour.

        spectra: list of (label, frequency, values) tuples. Matplotlib's
        default colour cycle assigns a different colour to each line.

        keep_view: keep the current zoom (used when a switch is flipped).
        The x range is always kept; the y range is kept only if the y label
        (units) is unchanged, because absorbance and absorption coefficient
        have very different scales.
        """
        keep_x = keep_y = None
        if keep_view and spectra and self.ax.lines:
            keep_x = self.ax.get_xlim()
            if ylabel == self._ylabel:
                keep_y = self.ax.get_ylim()

        self._ylabel = ylabel
        self.ax.clear()
        # Legend: keep "Pair N:" but show only what differs between the file names
        legend_labels = models.distinguishing_labels(
            [label for label, _, _ in spectra], keep_prefix=True
        )
        for (label, frequency, values), short in zip(spectra, legend_labels):
            self.ax.plot(frequency, values, label=short)
        self._style_axes()
        if spectra:
            self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()
        self.toolbar.update()  # makes Home return to the full (unzoomed) view

        # Restore the zoom AFTER toolbar.update(), so Home still goes to the full view
        if keep_x is not None:
            self.ax.set_xlim(keep_x)
            if keep_y is not None:
                self.ax.set_ylim(keep_y)
            self.canvas.draw_idle()

    def _on_scroll(self, event):
        """Scroll up = zoom in, scroll down = zoom out, centred on the cursor."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(
            event.xdata + (x0 - event.xdata) * scale,
            event.xdata + (x1 - event.xdata) * scale,
        )
        self.ax.set_ylim(
            event.ydata + (y0 - event.ydata) * scale,
            event.ydata + (y1 - event.ydata) * scale,
        )
        self.canvas.draw_idle()


if __name__ == "__main__":
    # Quick visual test with fake data (run: python -m ftir.views).
    # The switch only does something once the controller is connected.
    app = View()
    x = np.linspace(400, 4000, 400)
    app.plot_spectra(
        [("Pair 1", x, np.exp(-x / 2000)), ("Pair 2", x, 0.5 * np.exp(-x / 4000))]
    )
    app.set_cell_thickness(50.0)  # comment this out to see the disabled switch
    app.mainloop()
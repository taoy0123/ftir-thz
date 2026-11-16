"""Complete replacement for ftir/views.py.

Main window:
  * Empty cell, Background and Water vapour file fields (Browse and clear buttons)
  * three switches beside the graph:
      - difference absorbance <-> absorption coefficient (needs empty cell + background)
      - fringes: original <-> etalon fringes removed
      - water vapour: original <-> water vapour removed (needs a water vapour file)
  * a label showing the gap thickness
  * an SVD button, and SVDDialog showing principal components, scores and % variation
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

    A new pair has Browse buttons. A pair that is already loaded (`loaded`
    is its (label, sample_path, reference_path)) shows its files read-only;
    it can only be removed.
    """

    def __init__(self, parent, on_remove, loaded=None):
        super().__init__(parent, fg_color=("gray85", "gray22") if loaded else None)
        self.loaded = loaded
        self.sample = tk.StringVar(value=loaded[1] if loaded else "")
        self.reference = tk.StringVar(value=loaded[2] if loaded else "")

        self.grid_columnconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(
            self, text=loaded[0] if loaded else "", font=ctk.CTkFont(weight="bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 0), sticky="w")
        ctk.CTkButton(
            self, text="Remove", width=70, fg_color="gray",
            command=lambda: on_remove(self),
        ).grid(row=0, column=2, padx=10, pady=(8, 0), sticky="e")

        for row, (label, var) in enumerate(
            (("Sample:", self.sample), ("Reference:", self.reference)), start=1
        ):
            ctk.CTkLabel(self, text=label).grid(row=row, column=0, padx=(10, 5), pady=4, sticky="w")
            ctk.CTkEntry(self, textvariable=var, state="readonly").grid(
                row=row, column=1, padx=5, pady=4, sticky="ew"
            )
            if loaded is None:
                ctk.CTkButton(
                    self, text="Browse...", width=90,
                    command=lambda v=var, name=label: self._browse(v, name),
                ).grid(row=row, column=2, padx=(5, 10), pady=4)

    def set_number(self, number):
        """Show this row's current position: Pair N (loaded pairs add their file name)."""
        if self.loaded is None:
            self.title_label.configure(text=f"Pair {number}")
        else:
            name = re.sub(r"^Pair\s+\d+:\s*", "", self.loaded[0])
            self.title_label.configure(text=f"Pair {number}: {name}")

    def _browse(self, variable, name):
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title=f"Select {name.rstrip(':').lower()} file",
            filetypes=FILETYPES,
        )
        if path:  # empty string means the user cancelled the file picker
            variable.set(path)


class LoadPairsDialog(ctk.CTkToplevel):
    """Modal dialog for adding new sample/reference pairs and removing loaded ones.

    Rows are numbered 1, 2, 3... by position: removing a pair renumbers the
    rows after it, and new pairs come after the largest number.
    existing: list of (label, sample_path, reference_path) for the pairs
    currently loaded. They are listed first (shaded, read-only); press Remove
    on any you want taken off the plot.

    After it closes (OK):
      result  list of (sample_path, reference_path) for the NEW pairs
      kept    indices (into `existing`) of the loaded pairs that stay
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
        self.existing = list(existing or [])
        self.rows = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        hint = (
            "Loaded pairs are shaded. Remove the ones you no longer want, "
            "and add new pairs below."
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
        row = PairRow(self.scroll, self._remove_row, loaded=loaded)
        row.pack(fill="x", padx=5, pady=5)
        self.rows.append(row)
        self._renumber()

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
        self.result = new_pairs
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

    def __init__(self, parent, compute, save):
        super().__init__(parent)
        self.title("Singular value decomposition")
        self.geometry("1100x640")
        self.transient(parent)
        self._compute = compute
        self._save = save
        self.result = None   # as computed
        self.flipped = set() # indices of components whose sign is flipped

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

        self.after(100, self.run)

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
        else:
            self.thickness_label.configure(text=f"Gap thickness: {thickness_um:.1f} \u00b5m")
            self.display_switch.configure(state="normal")

    def water_mode(self):
        """True when the water vapour switch is on (show water vapour removed)."""
        return bool(self.water_switch.get())

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
        {"water": self.water_switch, "fringes": self.fringe_switch}[name].deselect()

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
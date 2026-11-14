"""Complete replacement for ftir/views.py.

Adds to the previous version:
  * Empty cell and Background file fields (with Browse and clear buttons)
  * a switch beside the graph: difference absorbance <-> absorption coefficient
  * a label showing the gap thickness
  * plot_spectra(spectra, ylabel) so the controller chooses the y-axis label
"""

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

FILETYPES = [("Text files", "*.txt *.dat *.csv"), ("All files", "*.*")]


class PairRow(ctk.CTkFrame):
    """One sample + reference pair inside the dialog."""

    def __init__(self, parent, on_remove):
        super().__init__(parent)
        self.sample = tk.StringVar()
        self.reference = tk.StringVar()

        self.grid_columnconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(weight="bold"))
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
            ctk.CTkButton(
                self, text="Browse...", width=90,
                command=lambda v=var, name=label: self._browse(v, name),
            ).grid(row=row, column=2, padx=(5, 10), pady=4)

    def set_number(self, number):
        self.title_label.configure(text=f"Pair {number}")

    def _browse(self, variable, name):
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title=f"Select {name.rstrip(':').lower()} file",
            filetypes=FILETYPES,
        )
        if path:  # empty string means the user cancelled the file picker
            variable.set(path)


class LoadPairsDialog(ctk.CTkToplevel):
    """Modal dialog for choosing one or more sample/reference pairs.

    After it closes, `result` is a list of (sample_path, reference_path)
    tuples, or None if the user cancelled.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Load sample/reference pairs")
        self.geometry("640x460")
        self.transient(parent)
        self.after(100, self.grab_set)  # make modal (delay avoids a macOS glitch)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.result = None
        self.rows = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="nsew")

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        ctk.CTkButton(bottom, text="+ Add pair", command=self._add_row).pack(side="left")
        ctk.CTkButton(
            bottom, text="Cancel", width=100, fg_color="gray", command=self._cancel
        ).pack(side="right", padx=(10, 0))
        ctk.CTkButton(bottom, text="OK", width=100, command=self._ok).pack(side="right")

        self._add_row()  # start with one empty pair

    def _add_row(self):
        row = PairRow(self.scroll, self._remove_row)
        row.pack(fill="x", padx=5, pady=5)
        self.rows.append(row)
        self._renumber()

    def _remove_row(self, row):
        if len(self.rows) == 1:  # always keep at least one pair
            return
        self.rows.remove(row)
        row.destroy()
        self._renumber()

    def _renumber(self):
        for i, row in enumerate(self.rows, start=1):
            row.set_number(i)

    def _ok(self):
        pairs = []
        for i, row in enumerate(self.rows, start=1):
            sample, reference = row.sample.get(), row.reference.get()
            if bool(sample) != bool(reference):  # only one of the two chosen
                messagebox.showwarning(
                    "Incomplete pair",
                    f"Pair {i} needs both a sample and a reference file.",
                    parent=self,
                )
                return
            if sample and reference:
                pairs.append((sample, reference))

        if not pairs:
            messagebox.showwarning(
                "No files", "Select at least one sample/reference pair.", parent=self
            )
            return

        self.result = pairs
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class View(ctk.CTk):
    """Main window: buttons, empty cell/background fields, plot, toggle and toolbar."""

    DEFAULT_YLABEL = "Difference absorbance, \u0394A"

    def __init__(self):
        super().__init__()
        self.title("FTIR THz analysis")
        self.geometry("1000x720")

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

        # Row 1: empty cell and background files
        cells = ctk.CTkFrame(self)
        cells.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 10), sticky="ew")
        cells.grid_columnconfigure(1, weight=1)
        self.empty_var = tk.StringVar()
        self.background_var = tk.StringVar()
        self.empty_button, self.empty_clear = self._add_file_row(
            cells, 0, "Empty cell:", self.empty_var
        )
        self.background_button, self.background_clear = self._add_file_row(
            cells, 1, "Background:", self.background_var
        )

        # Row 2, column 0: embedded matplotlib figure
        self._ylabel = self.DEFAULT_YLABEL
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=2, column=0, padx=(15, 5), pady=(0, 5), sticky="nsew")

        # Row 2, column 1: side panel with the display toggle
        panel = ctk.CTkFrame(self)
        panel.grid(row=2, column=1, padx=(0, 15), pady=(0, 5), sticky="n")
        ctk.CTkLabel(panel, text="Display", font=ctk.CTkFont(weight="bold")).pack(
            padx=15, pady=(15, 5)
        )
        self.display_switch = ctk.CTkSwitch(panel, text="Absorption coeff.", state="disabled")
        self.display_switch.pack(padx=15, pady=5)
        ctk.CTkLabel(
            panel, text="Off: \u0394A\nOn: \u03b1 (cm\u207b\u00b9)", justify="left"
        ).pack(padx=15, pady=(0, 10))
        self.thickness_label = ctk.CTkLabel(panel, text="Gap thickness: \u2014")
        self.thickness_label.pack(padx=15, pady=(5, 15))

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

    def plot_spectra(self, spectra, ylabel=DEFAULT_YLABEL):
        """Plot several spectra, each in its own colour.

        spectra: list of (label, frequency, values) tuples. Matplotlib's
        default colour cycle assigns a different colour to each line.
        """
        self._ylabel = ylabel
        self.ax.clear()
        for label, frequency, values in spectra:
            self.ax.plot(frequency, values, label=label)
        self._style_axes()
        if spectra:
            self.ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()
        self.toolbar.update()  # makes Home return to this new full view

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
    import numpy as np

    app = View()
    x = np.linspace(400, 4000, 400)
    app.plot_spectra(
        [("Pair 1", x, np.exp(-x / 2000)), ("Pair 2", x, 0.5 * np.exp(-x / 4000))]
    )
    app.set_cell_thickness(50.0)  # comment this out to see the disabled switch
    app.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox
 
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

 
 
class LoadFilesDialog(ctk.CTkToplevel):
    """Modal dialog asking for the sample and reference files.
 
    After the dialog closes, `result` is (sample_path, reference_path),
    or None if the user cancelled.
    """
 
    FILETYPES = [("Text files", "*.txt *.dat *.csv"), ("All files", "*.*")]
 
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Load spectra")
        self.geometry("560x210")
        self.resizable(False, False)
        self.transient(parent)  # keep on top of the main window
        self.after(100, self.grab_set)  # make modal (delay avoids a macOS glitch)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
 
        self.result = None
        self.sample_path = tk.StringVar()
        self.reference_path = tk.StringVar()
 
        self.grid_columnconfigure(1, weight=1)
        self._add_row(0, "Sample:", self.sample_path, "Select sample file")
        self._add_row(1, "Reference:", self.reference_path, "Select reference file")
 
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=2, column=0, columnspan=3, pady=(25, 10))
        ctk.CTkButton(buttons, text="OK", width=100, command=self._ok).pack(
            side="left", padx=10
        )
        ctk.CTkButton(
            buttons, text="Cancel", width=100, fg_color="gray", command=self._cancel
        ).pack(side="left", padx=10)
 
    def _add_row(self, row, label, variable, title):
        ctk.CTkLabel(self, text=label).grid(
            row=row, column=0, padx=(15, 5), pady=(20, 0), sticky="w"
        )
        ctk.CTkEntry(self, textvariable=variable, state="readonly").grid(
            row=row, column=1, padx=5, pady=(20, 0), sticky="ew"
        )
        ctk.CTkButton(
            self,
            text="Browse...",
            width=90,
            command=lambda: self._browse(variable, title),
        ).grid(row=row, column=2, padx=(5, 15), pady=(20, 0))
 
    def _browse(self, variable, title):
        path = filedialog.askopenfilename(
            parent=self, title=title, filetypes=self.FILETYPES
        )
        if path:  # empty string means the user cancelled the file picker
            variable.set(path)
 
    def _ok(self):
        sample, reference = self.sample_path.get(), self.reference_path.get()
        if not sample or not reference:
            messagebox.showwarning(
                "Missing file", "Please select both a sample and a reference file.",
                parent=self,
            )
            return
        self.result = (sample, reference)
        self.destroy()
 
    def _cancel(self):
        self.result = None
        self.destroy()

class View(ctk.CTk):
    """Main application window: a load button and a plot area."""

    def __init__(self):
        super().__init__()
        self.title("FTIR THz analysis")
        self.geometry("900x600")

        # Make the plot area stretch when the window is resized
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

         # Top bar with the two buttons (the controller attaches their commands in main.py)
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, padx=15, pady=15, sticky="w")
        self.load_button = ctk.CTkButton(bar, text="Load files...")
        self.load_button.pack(side="left", padx=(0, 10))
        self.save_button = ctk.CTkButton(bar, text="Save CSV...")
        self.save_button.pack(side="left")

        # Embedded matplotlib figure. Figure (not pyplot) is the safe choice
        # when embedding in a tkinter window.
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(
            row=1, column=0, padx=15, pady=(0, 15), sticky="nsew"
        )

        # Standard matplotlib toolbar: Home, Back/Forward, Pan, Zoom-to-rectangle, Save image
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="w")

        # Mouse-wheel zoom around the cursor
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _style_axes(self):
        self.ax.set_xlabel("Frequency (THz)")  # change units to match your data
        self.ax.set_ylabel("Absorbance change, \u0394A")

    def plot_spectrum(self, frequency, delta_A):
        """Called by the controller once the absorbance has been computed."""
        self.ax.clear()
        self.ax.plot(frequency, delta_A)
        self._style_axes()
        self.figure.tight_layout()
        self.canvas.draw()
         # Reset the toolbar history so the Home button returns to this new full view
        self.toolbar.update()

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
    root = ctk.CTk()
    root.geometry("300x150")

    def open_dialog():
        dialog = LoadFilesDialog(root)
        root.wait_window(dialog)
        print("Result:", dialog.result)

    ctk.CTkButton(root, text="Open dialog", command=open_dialog).pack(pady=50)
    root.mainloop()
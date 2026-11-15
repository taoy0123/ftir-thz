# Overview
This program provides an user interface for calculating and displaying FTIR spectra focusing on frequency range between 30 to 600 cm-1. The default input format is text file with wavenumbers in cm-1 on the left most column, and the intensities recorded on the column to the right (i.e., suits raw exports from OPUS - the software used with Bruker VERTEX series spectrocmeters).

# Features
- Calculates the difference in absorbance for a given sample and a reference (i.e., A_sample - A_reference)
- Calculates the empty cell thickness for a given empty cell and a background
- Minimises water vapour absorption lines by subtracting a scaled water vapur spectrum contained in **WaterVapor.txt**
- Minimises fringes caused by etalons within the transmission windows by the half period averaging method

Please refer to _requirements.txt_ for required packages.

Start the program from the terminal: python3 -m ftir.main
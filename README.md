# Overview
This program provides an user interface for calculating and displaying FTIR spectra focusing on frequency range between 30 to 600 cm-1. The default input format is text file with wavenumbers in cm-1 on the left most column, and the intensities recorded on the column to the right (i.e., suits raw exports from OPUS - the software used with Bruker VERTEX series spectrocmeters).

# Features
- Calculates the difference in absorbance for a given sample and a reference (i.e., A_sample - A_reference).
- Calculates the empty cell thickness for a given empty cell and a background.
- Minimises water vapour absorption lines by subtracting an automatically scaled water vapur spectrum. Specify the water vapour lines by selecting **WaterVapor.txt**.
- Minimises fringes caused by etalons within the transmission windows using the half period averaging method. It is recommended to perform water vapour lines minimisation beforehand for easier detection of the fringe period.
- Performs principal component analysis by decomposing the spectra into the singular components and their scores.
    - Specify a subset of frequency range and click **Run**. 
    - By default, the first principal component represents the mean values of all spectra. Toggle **Subtract mean spectrum** and re-click **Run** to see the decomposition result without it.
    - Change the sign of the principal component and its score by ticking the corresponding boxes near the bottom.
- Calculates the density-corrected mole fraction contribution from all components except the bulk water/solvent component.
    - requires a bulk water/solvent absorption coefficient spectrum
    - requires a bulk water/solvent molar absorptivity (i.e., absorption eofficient per mole) spectrum
    - requires a total concentration value for each spectrum. Calculate the total concentration by summing of the number of moles of all the components in the solution, multiplying it by the density of the solution, and then divide it by the mass of the solution.
    - requires a mole fraction value for each spectrum. The mole fraction is the number of moles of the bulk water/solvent in the solution divided by the sum of the number of moles of all the components in the solution.


Please refer to _requirements.txt_ for required packages. If using venv, import the required packages by:
```bash
python -m pip install -r requirements.txt --editable .
```

Start the user interface from the terminal:
```bash
python3 -m ftir.main
```

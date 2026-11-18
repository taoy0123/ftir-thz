# FTIR-THz
FTIR-THz is a graphical user interface that calculates and displays FTIR spectra. The features were developed with aqueous solution samples in mind, although it may be applied to other types of samples consisting of solvent and solutes. THz signifies that the spectra processing techniques apply to frequency range of 30 to 650 cm-1. The default input format: wavenumbers in cm-1 on the left column, and intensities in arbitrary units on the right column, saved as .txt. The default input format suits exports from OPUS - the software used with Bruker VERTEX series FTIR spectrocmeters.

## Main features
- Calculates the difference in absorbance for a given sample and a reference (i.e., A_sample - A_reference).
- Calculates the empty cell thickness for a given empty cell and a background.
- Minimises water vapour absorption lines by subtracting an automatically scaled water vapur spectrum. Specify the water vapour lines by selecting _WaterVapor.txt_.
- Minimises fringes caused by etalons within the transmission windows using the half period averaging method. It is recommended to perform water vapour lines minimisation beforehand for easier detection of the fringe period.
- Performs principal component analysis by decomposing the spectra into the singular components and their scores.
    - Specify a subset of frequency range and click _Run_. 
    - By default, the first principal component represents the mean values of all spectra. Toggle _Subtract mean spectrum_ and re-click _Run_ to see the decomposition result without it.
    - Change the sign of the principal component and its score by ticking the corresponding boxes near the bottom.
- Calculates the density-corrected mole fraction contribution from all components except the bulk water/solvent component.
    - requires a bulk water/solvent absorption coefficient spectrum
    - requires a bulk water/solvent molar absorptivity (i.e., absorption eofficient per mole) spectrum
    - requires a total concentration value for each spectrum. Calculate the total concentration by summing of the number of moles of all the components in the solution, multiplying it by the density of the solution, and then divide it by the mass of the solution.
    - requires a mole fraction value for each spectrum. The mole fraction is the number of moles of the bulk water/solvent in the solution divided by the sum of the number of moles of all the components in the solution.
- Performs multivariate curve resolution with one or more known spectral components. 
    - The known spectral components can be in .csv or .txt format.
    - Non-negativity can be enforced on scores and/or unknown components for physically meaningful results.

### How to read the residual plots
The residual is measured minus fitted, so it shows what your components fail to explain. Ideally it is small, noise-like and has no consistent features.

1. Residual spectra: look for peaks or humps that repeat across spectra. A repeated feature means a missing component.
2. Residual norm per spectrum: it should be roughly flat. If it rises with concentration, as it does in your NaOH set, something changes with concentration that the model doesn’t capture.
3. Heat map: vertical stripes mean a missing feature at that frequency. Horizontal bands mean particular spectra fit poorly. A pattern that drifts down the rows means something is evolving that the model misses.
4. Mean residual: it should sit at zero. If it doesn’t, you are consistently under- or over-fitting at those frequencies.

## Prerequisites
Please refer to _requirements.txt_ for required packages. If using venv, import the required packages by:
```bash
python -m pip install -r requirements.txt --editable .
```

## Quick start guide
Start the program from the terminal:
```bash
python3 -m ftir.main
```

## License
This project is released under the MIT License. You are free to use, modify and share it, including in your own projects, as long as the copyright and licence notice stay with the code.

## How to cite
The licence does not require it, but if you use this software in work that you publish (a paper, thesis or report), please cite it. GitHub's "Cite this repository" button gives the reference from CITATION.cff. In short:

> Tao, Y. H. ftir-thz: Process and analyse FTIR spectra (version 0.0.0). ORCID: 0000-0001-8028-5023.

Please also name the version you used, because the methods (fringe removal, water vapour correction, MCR) may change between versions.
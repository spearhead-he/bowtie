[![DOI](https://zenodo.org/badge/901850215.svg)](https://doi.org/10.5281/zenodo.14505386)
[![pytest](https://github.com/spearhead-he/bowtie/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/spearhead-he/bowtie/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/github/spearhead-he/bowtie/graph/badge.svg?token=JZ0QJ7PUU6)](https://codecov.io/github/spearhead-he/bowtie)

# SPEARHEAD bow-tie analysis tool

- [About](#about)
- [How to install](#how-to-install)
- [How to use](#how-to-use)
- [Contributing](#contributing)
- [Acknowledgement](#acknowledgement)

## About

This analysis tool runs a bow-tie analysis ([Van Allen et al. 1974](https://doi.org/10.1029/JA079i025p03559)) for the energy channels of a generic particle instrument. The input is a CSV table of channel responses indexed by the incident energy. The analysis results are the geometric factor (with errors) and the effective energy of the channel. The analysis description may be found here ([A description of the BepiColombo/SIXS-P cruise phase data product](https://doi.org/10.5281/zenodo.13692883)). The instrument response may be obtained by simulation, analytical model, or measurement and presented as a discrete function on an energy grid. The only requirement is that the grid must be monotonously ascending.

*Tested in Ubuntu 20.04.6 LTS with Python version 3.12.8.*

## How to install

1. This tool requires a recent Python (>=3.10) installation. [Following SunPy's approach, we recommend installing Python via miniforge (click for instructions).](https://docs.sunpy.org/en/stable/tutorial/installation.html#installing-python)
2. [Download this file](https://github.com/spearhead-he/bowtie/archive/refs/heads/main.zip) and extract to a folder of your choice (or clone the repository [https://github.com/spearhead-he/bowtie](https://github.com/spearhead-he/bowtie) if you know how to use `git`).
3. Open a terminal or the miniforge prompt and move to the directory where the code is.
4. *(Strongly recommended)* Create a new virtual environment (e.g., `conda create --name bowtie` or `python -m venv venv_bowtie_tool` if you don't use miniforge/conda) and activate it (e.g., `conda activate bowtie`, or `source venv_bowtie_tool/bin/activate` if you don't use miniforge/conda).
5. Install the Python dependencies from the *requirements.txt* file with `pip install -r requirements.txt` within the virtual environment. You can also pass "--user --break-system-packages" options **if you know well** what you are doing to install in your ~/.local/.
6. Open the Jupyter Notebook by running `jupyter-lab bowtie_example.ipynb`

## How to use

The Notebook is a simple example that also acts as a tutorial to teach the user how to run the bow-tie analysis with this tool.

The instrument response function(s) must be stored in a CSV file. The first column is the midpoint of the energy bin. The next columns are the response functions of the particle instrument. 

The tool operates with two main classes, which are called `Bowtie` and `Spectra`. `Bowtie` stores response functions and contains the methods to run bow-tie analysis, while `Spectra` contains information on the spectral indices and the amount of different spectra that are used in the bow-tie calculation.

### Bowtie
---
The `Bowtie` class contains the data that the bow-tie analysis is applied to and the energy range to be considered in the calculations. Its methods make running analysis easy and straightforward.

Methods:
#
	set_energy_range(energy_min, energy_max):
 		energy_min : {float} The minimum energy in MeV to consider
   		energy_max : {float} See energy_min.
#
 	bowtie_analysis(channel, spectra, plot):
  		channel : {str} The channel name as it appears in the csv table.
		spectra : {Spectra} The Spectra class object, introduced in this package. Contains the 
  							spectral indices and the power law spectra used in the bow-tie analysis.
  		plot : {bool} A boolean switch to produce a plot visualizing the analysis.
#
	bowtie_analysis_full_stack(spectra, plot):
 		A wrapper for bowtie_analysis(). Runs the analysis on all channels that appear in the input file.
 		spectra : {Spectra} See bowtie_analysis().
   		plot : {bool} See bowtie_analysis()

### Spectra
---
The `Spectra` class contains the range of spectra that are applied on the response function to run bow-tie analysis.

Methods:
#
	set_spectral_indices(gamma_min, gamma_max):
 		gamma_min : {float} The minimum spectral index to consider in the calculation.
   		gamma_max : {float} See gamma_min.
#
 	produce_power_law_spectra(response_df):
  		response_df : {pandas.DataFrame} The input csv table read in to a pandas DataFrame. Contains
										 the channel responses as a function of incident energy.

## Contributing

Contributions to this tool are very much welcome and encouraged! Contributions can take the form of [issues](https://github.com/spearhead-he/bowtie/issues) to report bugs and request new features or [pull requests](https://github.com/spearhead-he/bowtie/pulls) to submit new code. 

If you don't have a GitHub account, you can [sign-up for free here](https://github.com/signup), or you can also reach out to us with feedback by sending an email to jan.gieseler@utu.fi.

## Acknowledgement

<img align="right" height="80px" src="https://github.com/user-attachments/assets/28c60e00-85b4-4cf3-a422-6f0524c42234"> 
<img align="right" height="80px" src="https://github.com/user-attachments/assets/854d45ef-8b25-4a7b-9521-bf8bc364246e"> 

This tool is developed within the SPEARHEAD (*SPEcification, Analysis & Re-calibration of High Energy pArticle Data*) project. SPEARHEAD has received funding from the European Union’s Horizon Europe programme under grant agreement No 101135044. 

The tool reflects only the authors’ view and the European Commission is not responsible for any use that may be made of the information it contains.

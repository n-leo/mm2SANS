# ```mm2SANS```

Calculation of SANS scattering patterns from micromagnetic simulations.

## Disclaimer

This code is a heavy work in progress, so use it with caution, and please let me know if you find any bugs ;-).

## Installation

Easiest used is probably in conjuction with [Anaconda](https://www.anaconda.com/products/individual) which bundles python libraries for scientific computing, including Ipython/Jupyter notebooks.

To install the ```mm2SANS``` package, download the source code. Browse (within an Anaconda prompt) to the folder containing the unpacked code and run `python .\setup.py install`.

## Usage

See example IPython notebooks (*.ipynb*) in the `examples` folder, detailing the following features:

- Example 1: Overview of the `Sample` class (properties of the sample).
- Example 2: Overview of the `Probe` class (settings of beamline)
- Example 3: Overview of the `Experiment` class (calculation of scattering patterns).
- Example 4: Plot magnetic scattering pattern for neutron polarisation in differnt directions.

Each function in the code features a help text accessible via `command?` or via the `Shift-Tab` tool-tip shortcut from the IPython Notebook interface.


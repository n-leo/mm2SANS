# ```mm2SANS```

Calculation of SANS scattering patterns from micromagnetic simulations.

## Disclaimer

This code is a heavy work in progress, so use it with caution, and please let me know if you find any bugs ;-).

Relevant functionalities are now also implemented in SasView (https://www.sasview.org/).

## Installation

`mm2SANS` probably is easiest used in conjuction with [Anaconda](https://www.anaconda.com/products/individual), which bundles python libraries for scientific computing.

To install the `mm2SANS` package, download the source code. Browse (within an Anaconda prompt) to the folder containing the unpacked code and run `python .\setup.py install`.

## Usage

Each function in the code features a help text accessible via `command?` or via the `Shift-Tab` tool-tip shortcut from the IPython Notebook interface.

The example IPython notebooks (*\*.ipynb*) in the `examples` folder detail the following features:

- Example 1: Overview of the `Sample` class (properties of the sample).
- Example 2: Overview of the `Probe` class (settings of beamline and Q map).
- Example 3: Overview of the `Experiment` class (calculation of scattering patterns).
- Example 4: Plot magnetic scattering pattern for neutron polarisation in differnt directions.
- Example 5: Overview of the three coordinate systems used to describe a SANS experiment.
- Example 6: Calculation and visualisation of a rocking curve.

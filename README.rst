Pennylane-Perceval Plugin
#########################

.. header-start-inclusion-marker-do-not-remove

The PennyLane-Perceval plugin integrates the Quandela Perceval quantum computing framework with PennyLane's
quantum machine learning capabilities.

`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`Perceval <https://perceval.quandela.net/docs/index.html/>`_ provides tools for composing photonic circuits from linear optical components like beamsplitters and phase shifters, defining single-photon sources, manipulating Fock states, and running simulations.

.. header-end-inclusion-marker-do-not-remove

# Installation

## GitHub
```bash
git clone https://github.com/oradomskyi/pennylane-perceval
```
then to install Perceval:
```bash
pip install .
```

# Running tests and benchmarks

Unit tests files are part of the repository in `tests/` and can be run with:

```
pip install -r tests/requirements.txt
pytest
```
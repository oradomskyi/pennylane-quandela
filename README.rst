PennyLane-Quandela Plugin
#########################

.. image:: https://www.codefactor.io/repository/github/oradomskyi/pennylane-quandela/badge
   :target: https://www.codefactor.io/repository/github/oradomskyi/pennylane-quandela
   :alt: CodeFactor

.. header-start-inclusion-marker-do-not-remove

The PennyLane-Quandela plugin integrates the Quandela Perceval quantum computing framework with PennyLane's
quantum machine learning capabilities.

`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`Perceval <https://perceval.quandela.net/docs/index.html/>`_ provides tools for composing photonic circuits from linear optical components like beamsplitters and phase shifters, defining single-photon sources, manipulating Fock states, and running simulations.

.. header-end-inclusion-marker-do-not-remove

.. installation-start-inclusion-marker-do-not-remove

Installation
============

This plugin requires Python version 3.9 and above, as well as PennyLane and Quandela Perceval.
Installation of this plugin, as well as all dependencies could be done manually:

.. code-block:: bash

    git clone https://github.com/oradomskyi/pennylane-quandela

then to install plugin:

.. code-block:: bash

    pip install .

.. installation-end-inclusion-marker-do-not-remove

.. running-tests-start-inclusion-marker-do-not-remove
Running tests
=============

Unit tests files are part of the repository in `tests/` and can be run with:

.. code-block:: bash

    pip install -r tests/requirements.txt
    pytest

.. running-tests-end-inclusion-marker-do-not-remove

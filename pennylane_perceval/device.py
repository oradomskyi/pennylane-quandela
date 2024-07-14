# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Perceval device class
========================================

This module contains an abstract base class for constructing Perceval devices for PennyLane.

"""

from pennylane import QubitDevice

import perceval as pcvl

from ._version import __version__

class PercevalDevice(QubitDevice):
    r"""Perceval device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables

        provider (Provider | None): The Perceval backend provider.

        backend_name (str): the desired Perceval backend name.
    """
    name = "Perceval PennyLane plugin"
    pennylane_requires = ">=0.37.0"
    version = __version__
    plugin_version = __version__
    author = "Quandela"

    short_name = "perceval.base_device" # is this a correct name?

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
    }

    #TODO : Can we construct this dynamically?
    _operation_map = {
        # native PennyLane operations also native to Perceval
        "PauliX": "X",
        "PauliY": "Y",
        "PauliZ": "Z",
        "Hadamard": "H",
        "RX": "RX",
        "RY": "RY",
        "RZ": "RZ",
        "Toffoli" : "Toffoli",
        # operations not natively implemented in Perceval
        # additional operations not native to PennyLane but present in Perceval
    }

    @property
    def operations(self) -> set[str]:
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    @property
    def backend(self) -> pcvl.ABackend:
        """The Perceval backend object.

        Returns:
            perceval.backends.ABackend: Perceval backend object.
        """
        return self._backend

    @staticmethod
    def backends() -> set[str]:
        """List of names of available Perceval backends

        Returns:
            perceval.backends.BACKEND_LIST: list of string names of available backends
        """
        return set(pcvl.BACKEND_LIST.keys())

    # -- QubitDevice Interface implementation
    def apply(self, operations, **kwargs):
        """Append circuit operations, compile the circuit (if applicable),
        and perform the quantum computation.
        """
        raise NotImplementedError

    def generate_samples(self):
        """Generate samples from the device from the
        exact or approximate probability distribution.
        """
        raise NotImplementedError
    # ----------------------------- #

    def __init__(self, wires , shots: int, provider, backend_name: str, **kwargs):

        super().__init__(wires=wires, shots=shots)

        # This will fall back on SLOS when no backend found
        self._backend = pcvl.BackendFactory.get_backend(backend_name)

        # need to verify provider type
        self._provider = provider

        self.reset()

        self.process_kwargs(kwargs)

    def process_kwargs(self, kwargs):
        """Processing the keyword arguments that were provided upon device initialization.

        Args:
            kwargs (dict): keyword arguments to be set for the device
        """
        pass # for testing purposes
        # raise NotImplementedError

    def apply_operation(self, operation):
        """
        Add the specified operation to ``self.circuit`` with the native Perceval op name.

        Args:
            operation[pennylane.operation.Operation]: the operation instance to be applied
        """
        raise NotImplementedError

    def reset(self):
        """Reset the Perceval backend device
        
        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        # Reset only internal data, not the options that are determined on
        # device creation

        self._circuit = None
        self._processor = None
        self._source = None

    def create_circuit(self):
        """Compose a quantum circuit
        """
        raise NotImplementedError

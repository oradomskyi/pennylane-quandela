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

from pennylane import QubitDevice, DeviceError

from perceval import backends, providers

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

        backend (str | Backend): the desired backend. If a string, a provider must be given.
    """
    name = "Perceval PennyLane plugin"
    pennylane_requires = ">=0.37.0"
    version = __version__
    plugin_version = __version__
    author = "Quandela"

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
    }

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
        #return set(self._operation_map.keys())
        raise NotImplementedError

    @property
    def backend(self):
        """The Perceval backend object.

        Returns:
            perceval.backends.backend: Perceval backend object.
        """
        return self._backend

    def apply(self, operations, **kwargs):
        """Build the circuit object and apply the operations
        """
        raise NotImplementedError

    def apply_operation(self, operation):
        """
        Add the specified operation to ``self.circuit`` with the native Perceval op name.

        Args:
            operation[pennylane.operation.Operation]: the operation instance to be applied
        """
        raise NotImplementedError

    def __init__(self, wires , shots: int, provider, backend, **kwargs):

        super().__init__(wires=wires, shots=shots)

        available_backends = backends.BACKEND_LIST
        if backend.name() in available_backends:
            self._backend = backend
        else:
            raise ValueError(
                    f"Backend '{backend}' does not exist. Available backends "
                    f"are:\n {available_backends}"
            )

        # need to verify provider type
        # self.provider = provider

        self.reset()

        self.process_kwargs(kwargs)

    def reset(self):
        """Reset the Perceval backend device
        
        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        # Reset only internal data, not the options that are determined on
        # device creation

        raise NotImplementedError

    def process_kwargs(self, kwargs):
        """Processing the keyword arguments that were provided upon device initialization.

        Args:
            kwargs (dict): keyword arguments to be set for the device
        """
        raise NotImplementedError

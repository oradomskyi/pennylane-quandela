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

This module contains a base class for constructing Perceval devices for PennyLane.

"""
# pylint: disable=too-many-instance-attributes,broad-exception-raised

from typing import Union, Iterable, Optional

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

    Keyword Args:
        backend (str | None): the desired Perceval backend name.
            If not provided, will fallback on 'SLOS'
            For more see Computing Backends section:
                https://perceval.quandela.net/docs/

        provider (str | None): the Perceval provider name.
            If not provided, computation will run locally.
            For more see Providers section:
                https://perceval.quandela.net/docs/

        platform (str | None): the name of cloud computing platform.
            For more info see compatible cloud platforms in the Providers section:
                https://perceval.quandela.net/docs/

        api_token (str | None): the API token obtained from Quandela Cloud Provider 
            if token is not provided, computation will run locally.

    Raises:
        Exception: when either is invalid :attr:`provider`, :attr:`platform` or :attr:`api_token`

    .. note::
        We strongly recommend you to keep secrets(like tokens and keys) in the
        separate files and load environment, instead of hardcoding these in your scripts.

        Here is example how to obtain a variable named 'PERCEVAL_TOKEN'
        stored in text file called '.env_perceval' using dotenv and os packages:

    .. code-block:: python
        >>> import os
        >>> from dotenv import load_dotenv
        >>> load_dotenv('.env_perceval')
        >>> my_token = os.environ.get('PERCEVAL_TOKEN')
    """
    name = "Perceval PennyLane plugin"
    short_name = "perceval.device"
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
        "CNOT" : "CNOT",
        # operations not natively implemented in Perceval
        # additional operations not native to PennyLane but present in Perceval
    }

    # public
    processor = None
    circuit = None

    # protected
    _backend = None
    _provider = None
    _api_token = None
    _backend_name = None
    _provider_name = None
    _platform_name = None

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

    @property
    def provider(self) -> pcvl.ISession:
        """The Perceval provider object.

        Returns:
            perceval.providers.ISession
        """
        return self._provider

    # -- Interface of pennylane.QubitDevice and its parent
    def apply(self, operations):
        """Create circuit from operations, compile the circuit (if applicable),
        and perform the quantum computation.
        """
        self._create_circuit(operations)
        self._create_processor()

    def generate_samples(self):
        """Generate samples from the device from the
        exact or approximate probability distribution.
        """
        return self._samples_as_pennylane()

    def reset(self) -> None:
        """Reset the Perceval backend device
        
        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        # Reset only internal data, not the options that are determined on
        # device creation

        self.circuit = None
        self.processor = None
    # ----------------------------- #

    def __init__(self, wires: Union[int, Iterable[Union[int, str]]], shots: int, **kwargs):
        super().__init__(wires=wires, shots=shots)
        self._read_kwargs(**kwargs)

        # This will fall back on SLOS when no backend found
        self._backend = pcvl.BackendFactory.get_backend(self._backend_name)

        if (self._provider_name is not None and
            self._platform_name is not None and
            self._api_token          is not None ):
            try:
                self._provider = pcvl.ProviderFactory.get_provider(self._provider_name,
                    platform_name=self._platform_name,
                    token=self._api_token)
            except Exception as e:
                raise Exception(
                    f"Cannot connect to provider {self._provider_name} platform {self._platform_name} with token {self._api_token}"
                ) from e

        # Set default inner state
        self.reset()

    def create_circuit(self, operations) -> None:
        """Compose a quantum circuit
        """
        raise NotImplementedError

    def create_processor(self) -> None:
        """Create Perceval Processor
        """
        raise NotImplementedError

    def samples_as_pennylane(self):
        """View of samples in PennyLane-compatible format
        """
        raise NotImplementedError

    def _read_kwargs(self, **kwargs) -> None:
        """
        Keyword Args:
            backend (str | None): the desired Perceval backend name.
                If not provided, will fallback on 'SLOS'
                For more see Computing Backends section:
                    https://perceval.quandela.net/docs/

            provider (str | None): the Perceval provider name.
                If not provided, computation will run locally.
                For more see Providers section:
                    https://perceval.quandela.net/docs/

            platform (str | None): the name of cloud computing platform.
                For more info see compatible cloud platforms in the Providers section:
                    https://perceval.quandela.net/docs/

            api_token (str | None): the API token obtained from Quandela Cloud Provider 
                if token is not provided, computation will run locally.
        """
        if 'backend' in kwargs:
            self._backend_name = kwargs['backend']
        if 'provider' in kwargs:
            self._provider_name = kwargs['provider']
        if 'platform' in kwargs:
            self._platform_name = kwargs['platform']
        if 'api_token' in kwargs:
            self._api_token = kwargs['api_token']

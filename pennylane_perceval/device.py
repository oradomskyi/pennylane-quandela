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

from typing import Union, Iterable

from pennylane import QubitDevice
from pennylane.operation import Operation

from perceval import (
    ISession,
    ABackend,
    BackendFactory,
    ProviderFactory,
    Processor
)

from ._version import __version__
from .pennylane_converter import PennylaneConverter


class PercevalDevice(QubitDevice):
    r"""Perceval device for PennyLane.

    Args:
        :param wires: (int or Iterable[Number,str]]) Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).

        :param shots: (int) number of circuit evaluations/random samples used
            to estimate expectation values of observables

    Keyword Args:
        :param backend: (str | None) desired Perceval backend name.
            If not provided, will fallback on 'SLOS'
            For more see `Computing Backends` section:
                https://perceval.quandela.net/docs/

        :param provider: (str | None) Perceval provider name.
            If not provided, computation will run locally.
            For more see `Providers` section:
                https://perceval.quandela.net/docs/

        :param platform: (str | None) name of cloud computing platform.
            For more info see compatible cloud platforms in the `Providers` section:
                https://perceval.quandela.net/docs/

        :param api_token: (str | None) API token obtained from Quandela Cloud Provider 
            if token is not provided, computation will run locally.
        
        :param source: (Source | None) single photon source, defined by specifying parameters
            such as `brightness`, `purity` or `indistinguishability`.
            For more infor see `Sources` sction:
            https://perceval.quandela.net/docs/

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

    # protected
    _pennylane_converter = None
    _backend = None
    _provider = None
    _processor = None

    @property
    def operations(self) -> set[str]:
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    @property
    def backend(self) -> ABackend:
        """Perceval backend object.

        Returns:
            perceval.backends.ABackend: Perceval backend object.
        """
        return self._backend

    @property
    def provider(self) -> ISession:
        """Perceval provider object.

        Returns:
            perceval.providers.ISession
        """
        return self._provider

    @property
    def processor(self) -> Processor:
        """Perceval Processor object
        
        Returns:
            perceval.components.processor.Processor
        """
        return self._processor

    def __init__(self, wires: Union[int, Iterable[Union[int, str]]], shots: int, **kwargs):
        QubitDevice.__init__(self, wires=wires, shots=shots)

        # This will fall back on SLOS when no backend found
        self._backend = BackendFactory.get_backend(kwargs.get('backend', None))

        provider = kwargs.get('provider', None)
        platform = kwargs.get('platform', None)
        api_token = kwargs.get('api_token', None)
        if (provider is not None and
            platform is not None and
            api_token is not None):
            try:
                self._provider = ProviderFactory.get_provider(provider_name=provider,
                    platform_name=platform,
                    token=api_token)
            except Exception as e:
                raise Exception(
                    f"Cannot connect to provider {provider}" +
                    f"platform {platform}" +
                    f"with token {api_token}") from e

        self._pennylane_converter = PennylaneConverter(
                    catalog = kwargs.get('catalog', None),
                    backend_name = kwargs.get('backend', None),
                    source = kwargs.get('source', None))

        # Set default inner state
        self.reset()

    # -- Interface of pennylane.QubitDevice and its parent
    def apply(self, operations: list[Operation], **kwargs):
        """Create circuit from operations, compile the circuit (if applicable),
        and perform the quantum computation.

        Args:
            :param operations: list[pennylane.operations.Operation]
                list of PennyLane objects representing a quantum circuit.
        
        Keyword Args:
            Not implemented
        
        Returns:
            None
        """
        self.processor = self._pennylane_converter.convert(operations)

    def generate_samples(self):
        """Generate samples from the device from the
        exact or approximate probability distribution.
        """
        return self.samples_as_pennylane()

    def reset(self) -> None:
        """Reset the Perceval backend device
        
        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        # Reset only internal data, not the options that are determined on
        # device creation
        self.processor = None
    # ----------------------------- #

    def samples_as_pennylane(self):
        """View of samples in PennyLane-compatible format
        """
        raise NotImplementedError

# MIT License
#
# Copyright (c) 2024 Oleksandr Radomskyi
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
from time import sleep
from warnings import warn

from pennylane import QubitDevice
from pennylane.operation import Operation

from perceval import (
    ISession,
    ABackend,
    BackendFactory,
    ProviderFactory,
    Processor,
    Circuit,
    NoiseModel,
    LocalJob,
    RemoteJob
)

from perceval.algorithm import Sampler

from ._version import __version__
from .converter_pennylane import PennylaneConverter

############################################################################
############################################################################
from qiskit import *
from qiskit.primitives import BackendSampler
from qiskit_aer import Aer
from qiskit.circuit import QuantumRegister, QuantumCircuit
from pennylane import QubitDevice, device, matrix as qml_matrix

from perceval import (pdisplay, Format)
############################################################################
############################################################################

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
    _circuit = None
    _input_state = None
    _min_detected_photons = None
    _noise_model = None
    _job_name = None
    _job = None

    DEFAULT_MIN_DETECTED_PHOTONS = 1
    DEFAULT_NOISE_MODEL_INDISTINGUISHABILITY = .95
    DEFAULT_NOISE_MODEL_TRANSMITTANCE = .1
    DEFAULT_NOISE_MODEL_G2 = .01
    DEFAULT_JOB_NAME = "My job"

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

    @property
    def circuit(self) -> Circuit:
        """Perceval Circuit object."""
        return self._circuit

    @property
    def input_state(self):
        """Input State object."""
        return self._input_state

    @input_state.setter
    def input_state(self, new_state):
        """Set the input state for the photonic circuit"""
        self._input_state = new_state

    @property
    def min_detected_photons(self):
        """Min detected photons of the Sampler"""
        return self._min_detected_photons

    @min_detected_photons.setter
    def min_detected_photons(self, new_min_detected_photons):
        """Set the min detected photons for the circuit Sampler"""
        self._min_detected_photons = new_min_detected_photons

    @property
    def noise_model(self):
        """Noise model of photon source"""
        return self._noise_model

    @noise_model.setter
    def noise_model(self, new_noise_model):
        """Set the noise model for the photon source"""
        self._noise_model = new_noise_model

    @property
    def job_name(self):
        """Name of the computation job you are running,
        this will be displayed in your Cloud Console"""
        return self._job_name

    @job_name.setter
    def job_name(self, new_job_name):
        """Set the new name for your computation job"""
        self._job_name = new_job_name

    @property
    def job(self):
        """Handle to the ongoing job"""
        return self._job

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
                    source = kwargs.get('source', None),
                    num_qubits = len(self.wires))

        # Set default inner state
        self.reset()

    # -- Interface of pennylane.QubitDevice and its parent class -- #
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
        self._process_apply_kwargs(kwargs)
        processor = self._pennylane_converter.convert(operations)
        self._circuit = processor.linear_circuit()

        if self.provider is not None:
            # Setup remote processor
            self._processor = self.provider.build_remote_processor()
            self.processor.set_circuit(self._circuit)
        else:
            # Setup local processor
            self._processor = processor

        #pdisplay(processor, recursive=True, output_format=Format.TEXT)
        self._submit_job()
   
        ################################################################
        ################################################################
        self._wait_for_job_to_complete()
        results = self.job.get_results()
        warn(f"Perceval job results: {results['results']}")
        print(results)
        dist = self.processor.probs()["results"]
        print(self.processor.probs())
        print(dist)

        print('create_circuit')
        qiskit_device = device('qiskit.aer', wires=super().wires)

        qiskit_device.create_circuit_object(operations)
        qiskit_circuit = qiskit_device._circuit

        print("qiskit_circuit")
        print(qiskit_circuit)
        print(type(qiskit_circuit), type(qiskit_circuit.data))
        
        #for i, instruction in enumerate(qiskit_circuit.data):
        #    print(instruction[0].name)
        #    if instruction[0].name not in ["h", "cx"]:
        #del qiskit_circuit.data[3]
        #del qiskit_circuit.data[3]
        #del qiskit_circuit.data[3]

        #qiskit_compiled_circuit = qiskit_device.compile()

        #print("qiskit_compiled_circuit\n", qiskit_compiled_circuit)
        #qiskit_compiled_circuit.draw()
        qiskit_circuit.draw()
        
        #Changing the simulator 
        backend = Aer.get_backend('aer_simulator')#('unitary_simulator')
        print(backend)
        sampler = BackendSampler(backend)
        print(sampler)
        qiskit_circuit = transpile(qiskit_circuit, backend=backend)
        print(qiskit_circuit)
        qiskit_job = sampler.run(qiskit_circuit)
        
        print("qiskit_job", qiskit_job)
        qiskit_job_result = qiskit_job.result()
        print("qiskit_job_result", qiskit_job_result)
        qiskit_counts = qiskit_job_result.get_counts(qiskit_circuit)
        print("qiskit_counts", qiskit_counts)
        ################################################################
        ################################################################

    def generate_samples(self):
        """Generate samples from the device from the
        exact or approximate probability distribution.
        """

        self._wait_for_job_to_complete()
        results = self.job.get_results()
        warn(f"Perceval job results: {results['results']}")

        return self._format_results()

    def reset(self) -> None:
        """Reset the Perceval backend device
        
        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        # Reset only internal data, not the options that are determined on
        # device creation
        self._processor = None
        self._min_detected_photons = None
        self._noise_model = None
        self._job_name = None
        self._job = None
    # ------------------------------------------------------------- #

    def _format_results(self):
        """Format Perceval results to a PennyLane-compatible format
        """
        return self.processor.probs()["results"]

    def _process_apply_kwargs(self, kwargs) -> None:
        """Processing the keyword arguments that were provided by PennyLane
        when calling an apply() function.

        Args:
            kwargs (dict): keyword arguments to be set for the device
        """

    def _submit_job(self):
        """Prepare and submit the computation job"""

        if not self.min_detected_photons:
            self.min_detected_photons = self.DEFAULT_MIN_DETECTED_PHOTONS

        # Output state filering on the basis of detected photons
        self.processor.min_detected_photons_filter(self.min_detected_photons)

        # User is obligated to provide input state
        if not self.input_state:
            raise ValueError(
                "Please set the input state. "+
                "See `Remote Computing with Quandela Cloud` "+
                "https://perceval.quandela.net/docs/v0.11/notebooks/Remote_computing.html")

        self.processor.with_input(self._input_state)

        if not self.noise_model:
            self.noise_model = NoiseModel(
                indistinguishability = self.DEFAULT_NOISE_MODEL_INDISTINGUISHABILITY,
                transmittance = self.DEFAULT_NOISE_MODEL_TRANSMITTANCE,
                g2 = self.DEFAULT_NOISE_MODEL_G2)

        self.processor.noise = self.noise_model

        # You have to set a 'max_shots_per_call' named parameter
        # Here, with `min_detected_photons_filter` set to 1, all shots
        # are de facto samples of interest. Thus, in this particular case,
        # the expected sample number can be used as the shots threshold.
        sampler = Sampler(self.processor, max_shots_per_call=self.shots)

        # All jobs created by this sampler instance will have this custom name on the cloud
        if not self.job_name:
            self.job_name = self.DEFAULT_JOB_NAME

        sampler.default_job_name = self.job_name

        try:
            self._job = sampler.sample_count.execute_async(self.shots)  # Create a job
            # Once created, the job was assigned a unique id
            if isinstance(self.job, LocalJob):
                warn(f"Job submitted: {self._job}")
            elif isinstance(self.job, RemoteJob):
                warn(f"Job submitted: {self._job.id}")
        except Exception as e:
            raise Exception("Cannot submit the Job.") from e

    def _wait_for_job_to_complete(self):
        """Poll job status"""
        while not self.job.is_complete:
            sleep(1)

        if isinstance(self.job, LocalJob):
            warn(f"Job: {self._job}\n Status: {self.job.status()}\n")
        elif isinstance(self.job, RemoteJob):
            warn(f"Job: {self._job.id}\n Status: {self.job.status()}\n")

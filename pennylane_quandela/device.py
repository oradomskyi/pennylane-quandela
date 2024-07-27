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
Quandela device class
========================================

This module contains a base class for constructing Quandela devices for PennyLane.

"""
# pylint: disable=too-many-instance-attributes,broad-exception-raised

from typing import Union, Iterable
from time import sleep

from numpy import vstack, array as np_array

from pennylane import QubitDevice
from pennylane.operation import Operation

from perceval import (
    ISession,
    ABackend,
    BackendFactory,
    ProviderFactory,
    Processor,
    Circuit,
    NoiseModel
)

from perceval.algorithm import Sampler

from ._version import __version__
from .converter_pennylane import PennylaneConverter


class QuandelaDevice(QubitDevice):
    r"""Quandela device for PennyLane.

    Quandela photonic circuits use Fock states to represent a quantum system.
    One natural way to encode qubits is the path encoding. A qubit is a two-level quantum state,
    so we will use two spatial modes to encode it: this is the dual-rail or path encoding.
    
    The logical qubit state |0> will correspond to a photon in the upper mode,
    as in the Fock state |1,0>, while logical qubit state |1> will be encoded as |0,1> Fock state.

    Read more:
        https://perceval.quandela.net/docs/

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

        :param source: (Source | None) single photon source, defined by specifying parameters
            such as `brightness`, `purity` or `indistinguishability`.
            For more infor see `Sources` sction:
            https://perceval.quandela.net/docs/

        ``Quandela Cloud`` related arguments

            https://cloud.quandela.com/

        :param platform: (str | None) name of cloud computing platform.
            For more info see compatible cloud platforms in the `Providers` section:
                https://perceval.quandela.net/docs/

        :param api_token: (str | None) API token obtained from Quandela Cloud Provider 
            if token is not provided, computation will run locally.
        
        ``Scaleway Cloud`` related arguments
            https://console.scaleway.com/

        :param platform: (str | None) name of the computing platform
            for example "sim:sampling:p100" or "sim:sampling:h100"

        :param project_id: (str | None) your Scaleway project ID
            https://console.scaleway.com/organization/projects

        :param token: (str | None) your API key
            https://console.scaleway.com/iam/api-keys

        :param deduplication_id: (str | None) default ""
 
        :param max_idle_duration_s: (int | None) default 1200, number of seconds a Scaleway session can idle

        :param max_duration_s: (int | None) default 3600, duration of your Scaleway session, for pricing info visit
            https://console.scaleway.com/qaas

    Raises:
        Exception: when any of cloud provider related parameters is invalid
            so the Device is unable to establish a session with the remote cloud service.

    .. note::
        We strongly recommend you to keep secrets(like tokens and keys) in the
        separate files and load environment, instead of hardcoding these in your scripts.

        Here is example how to obtain a variable named 'QUANDELA_TOKEN'
        stored in text file called '.env_quandela' using dotenv and os packages:

    .. code-block:: python
        >>> import os
        >>> from dotenv import load_dotenv
        >>> load_dotenv('.env_quandela')
        >>> my_token = os.environ.get('QUANDELA_TOKEN')
    """
    name = "Quandela PennyLane plugin"
    short_name = "quandela.device"
    pennylane_requires = ">=0.37.0"
    version = __version__
    plugin_version = __version__
    author = "Oleksandr Radomskyi"

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

    DEFAULT_MIN_DETECTED_PHOTONS = 0
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
        """Set the min detected photons for the circuit.
        
        Read more about photon detection, in real devices it is sometimes necessary
        to use detected-photons threshold to filter out the results.

        https://perceval.quandela.net/
        """
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

        if kwargs.get('provider_name', None) is not None:
            try:
                print('kwargs', kwargs)
                self._provider = ProviderFactory.get_provider(**kwargs)
            except Exception as e:
                raise Exception(
                    "Cannot connect to a cloud provider, " +
                    f"settings are:{[ f'{k}:{v}' for k,v in kwargs.items()]}\n"
                ) from e

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
        self._process_kwargs(kwargs)

        #TODO: Do we want to expose use_postselection param to user ?
        processor = self._pennylane_converter.convert(operations)

        self._circuit = processor.linear_circuit()

        if self.provider is not None:
            # Setup remote processor
            self._processor = self.provider.build_remote_processor()
            self.processor.set_circuit(self._circuit)
        else:
            # Setup local processor
            self._processor = processor

        self._submit_job()
        self._wait_for_job_to_complete()

    def generate_samples(self):
        """Generate samples from the device from the
        exact or approximate probability distribution.
        """
        # Consider example 3-wire, 10 shot logical qubit system:
        #
        # Perceval sample counts are a dict
        # {
        #   |1,0,1,0,1,0>: 1,
        #   |1,0,0,1,1,0>: 4,
        #   |0,1,1,0,0,1>: 3,
        #   |0,1,0,1,0,1>: 2
        # }
        #
        # Qiskit samples for this system would be list of strings
        # [
        #   '010', '101', '010', '000', '101', '111', '111', '101', '010', '010'
        # ]
        #
        # PennyLane expect samples as list of lists of integers
        # np.vstack( [ np.array([int ...]) np.array([int ...]) ...] )
        #
        # [
        #  [0,1,0], [1,0,1], [0,1,0], [0,0,0], [1,0,1], [1,1,1], [1,1,1], [1,0,1], [0,1,0], [0,1,0]
        # ]
        results = self.job.get_results()['results']
        samples = [ [0] * self.num_wires ] * self.shots
        idx = 0
        n_non_convertible_states = 0
        for state in results:
            q_state_key = self._state_to_list_int(state)
            if q_state_key is None:
                n_non_convertible_states += 1
                # do not move samples index forward
                continue

            # state is converted, copy it to output list
            for _ in range(0, results[state]):
                samples[idx] = q_state_key # copy instead of assignment
                idx += 1

        # would reduce the number of samples,
        # does this function properly with PennyLane?
        if 0 < n_non_convertible_states:
            samples = samples[0:-n_non_convertible_states]

        samples  = vstack(samples)
        return samples

    def reset(self) -> None:
        """Reset the Perceval backend device
        
        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        # Reset only internal data, not the options that are determined on
        # device creation
        self._processor = None
        self._circuit = None
        self._min_detected_photons = None
        self._noise_model = None
        self._job_name = None
        self._job = None
    # ------------------------------------------------------------- #

    def _process_kwargs(self, kwargs) -> None:
        """Processing the keyword arguments that were provided by PennyLane.

        Args:
            kwargs (dict): keyword arguments to be set for the device
        """

    def _submit_job(self):
        """Prepare and submit the computation job"""

        if not self.min_detected_photons:
            self._min_detected_photons = self.DEFAULT_MIN_DETECTED_PHOTONS

        # Output state filering on the basis of detected photons
        self.processor.min_detected_photons_filter(self.min_detected_photons)

        # User is obligated to provide input state
        if not self.input_state:
            raise ValueError(
                "Please set the input state. "+
                "See `Remote Computing with Quandela Cloud` "+
                "https://perceval.quandela.net/docs/v0.11/notebooks/Remote_computing.html")

        # exqalibur.StateVector might not have has_polarization property
        if hasattr(self.input_state, 'has_polarization'):
            self.processor.with_polarized_input(self.input_state)
        else:
            self.processor.with_input(self.input_state)

        if not self.noise_model:
            self._noise_model = NoiseModel(
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
            self._job_name = self.DEFAULT_JOB_NAME

        sampler.default_job_name = self.job_name

        try:
            self._job = sampler.sample_count.execute_async(self.shots)  # Create a job
        except Exception as e:
            raise Exception("Cannot submit the Job.") from e

    def _wait_for_job_to_complete(self):
        """Poll job status"""
        while not self.job.is_complete:
            sleep(1)

    def _state_to_list_int(self, state) -> list[int]:
        """Transforms Fock state into list of integers

        In spatial encoding, some Fock states don’t correspond to any qubit state.
        An example of such a Fock state |2,0> is where two photons are sent 
        in path 0 and no photon in path 1.

        More generally, any state which is not a superposition of Fock states
        |1,0> and |0,1> cannot be associated with a qubit state.

        Args:
            :param state:  resresentation os a Fock state

        Returns:
            int representation of a Fock state or None
            when state has no photons for a qubit like |0,0>
            or when there are several photons in one state line |2,0>

        Note that PennyLane uses the convention :math:`|q_0,q_1, ... ,q_{N-1}>` where
        :math:`q_0` is the most significant bit.

        Quandela Ports are using 0-based numbering - so port 0 is corresponding 
        to the first line, port (m-1) is corresponding to the m-th line.

        Examples with 2-bit states:
            |1,0,1,0> -> 0 -> [0 0]
            |1,0,0,1> -> 1 -> [0 1]
            |0,1,1,0> -> 2 -> [1 0]
            |0,1,0,1> -> 3 -> [1 1]
        """
        f_state = str(state).replace(",", "")[1:-1]

        f_state_list = np_array([int(i) for i in f_state])

        q_state_list = [0] * self.num_wires
        for i in range(0, self.num_wires):
            if f_state_list[i*2] == 1 and f_state_list[i*2 + 1] == 0:
                q_state_list[i] = 0
                continue

            if f_state_list[i*2] == 0 and f_state_list[i*2 + 1] == 1:
                q_state_list[i] = 1
                continue

            return None

        return q_state_list

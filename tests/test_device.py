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
"""Tests for the QuandelaDevice class"""

# pylint: disable=protected-access

import pytest

from numpy import array as np_array
import pennylane as qml

from perceval.backends import BackendFactory
from perceval import ABackend, BasicState

from pennylane_quandela import QuandelaDevice


class TestQuandelaDevice:
    """Tests for the QuandelaDevice base class."""

    @pytest.mark.parametrize("wires", [1, 3])
    @pytest.mark.parametrize("shots", [1, 100])
    def test_init_no_kwargs(self, wires , shots):
        """Tests that the device is properly initialized."""

        device = QuandelaDevice(wires=wires, shots=shots)

        assert device.num_wires == wires
        assert device.shots == shots
        assert device.backend.name in 'SLOS'

        assert device.provider is None

    @pytest.mark.parametrize("wires", [1, 3])
    @pytest.mark.parametrize("shots", [1, 100])
    @pytest.mark.parametrize("backend_name", BackendFactory.list())
    def test_init_with_backend(self, wires, shots, backend_name):
        """Tests that the device is properly initialized."""

        device = QuandelaDevice(wires=wires, shots=shots, backend=backend_name)

        assert device.num_wires == wires
        assert device.shots == shots
        assert isinstance(device.backend, ABackend)
        assert device.backend.name in backend_name
        assert device.processor is None

    def test_reset(self):
        """Tests that device is properly reset"""

        device = QuandelaDevice(1, 1)
        device.reset()

        assert device.processor is None

    def test_state_to_list_valid_inputs_1_wire(self):
        """Test if device can convert specific Fock states
        into quantum states.
        """
        device = QuandelaDevice(wires=1, shots=1)
        valid_states = [
            BasicState([1,0]),
            BasicState([0,1]),
        ]

        expected_valid_stated = [
            [0],
            [1],
        ]

        for a, b in zip(valid_states, expected_valid_stated):
            state_list = device._state_to_list_int(a)
            assert isinstance(state_list, list)
            for x, y in zip(state_list, b):
                assert x == y

    def test_state_to_list_valid_inputs_2_wires(self):
        """Test if device can convert specific Fock states
        into quantum states.
        """
        device = QuandelaDevice(wires=2, shots=1)
        valid_states = [
            BasicState([1,0,1,0]),
            BasicState([1,0,0,1]),
            BasicState([0,1,1,0]),
            BasicState([0,1,0,1])
        ]

        expected_valid_stated = [
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ]

        for a, b in zip(valid_states, expected_valid_stated):
            state_list = device._state_to_list_int(a)
            assert isinstance(state_list, list)
            for x, y in zip(state_list, b):
                assert x == y

    def test_state_to_list_invalid_inputs_1_wires(self):
        """Test that device cannot convert specific Fock states
        into quantum states and raises ValueError.
        """
        device = QuandelaDevice(wires=1, shots=1)

        invalid_q_states = [
            BasicState([0,0]),
            BasicState([1,1]),
            BasicState([2,0]),
            BasicState([0,3]),
            BasicState([1,2])
        ]

        for state in invalid_q_states:
            with pytest.raises(ValueError):
                device._state_to_list_int(state)

    def test_state_to_list_invalid_inputs_2_wires(self):
        """Test that device cannot convert specific Fock states
        into quantum states and raises ValueError.
        """
        device = QuandelaDevice(wires=2, shots=1)

        invalid_q_states = [
            BasicState([0,1,1,1]),
            BasicState([2,0,0,1]),
            BasicState([0,0,0,0]),
            BasicState([9,4,2,3])
        ]

        for state in invalid_q_states:
            with pytest.raises(ValueError):
                device._state_to_list_int(state)

    def test_qnode_probs_1(self):
        """Test correctness of circuit execution 
        Perceval vs PennyLane
        """
        wires=3
        weights = np_array([0.1, 0.2, 0.7])

        dev_pennylane = qml.device("default.qubit", wires=wires)

        test_input = BasicState([1,0,1,0,1,0])
        dev_perceval = QuandelaDevice(wires=wires, shots=1024, backend='SLOS')
        dev_perceval.input_state = test_input

        @qml.qnode(dev_pennylane)
        def circuit_pennylane(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(weights[2], wires=1)
            return qml.probs(wires=1)

        @qml.qnode(dev_perceval)
        def circuit_perceval(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(weights[2], wires=1)
            return qml.probs(wires=1)

        results_pennylane = circuit_pennylane(weights)
        results_perceval = circuit_perceval(weights)

        eps = 2e-2
        for x,y in zip(results_pennylane, results_perceval):
            assert abs(x - y) < eps

    def test_qnode_probs_2(self):
        """Test correctness of circuit execution 
        Perceval vs PennyLane
        """
        wires=2

        dev_pennylane = qml.device("default.qubit", wires=wires)

        test_input = BasicState([1,0,1,0])
        dev_perceval = QuandelaDevice(wires=wires, shots=1024, backend='SLOS')
        dev_perceval.input_state = test_input

        @qml.qnode(dev_pennylane)
        def circuit_pennylane():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=1)

        @qml.qnode(dev_perceval)
        def circuit_perceval():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=1)

        results_pennylane = circuit_pennylane()
        results_perceval = circuit_perceval()

        eps = 2.5e-2
        for x,y in zip(results_pennylane, results_perceval):
            assert abs(x - y) < eps

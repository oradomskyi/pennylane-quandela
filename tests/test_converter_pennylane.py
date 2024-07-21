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

# pylint: disable=protected-access,missing-function-docstring,missing-module-docstring

import pytest

try:
    import pennylane as qml
except ModuleNotFoundError as e:
    assert e.name == "pennylane"
    pytest.skip("need `pennylane` module", allow_module_level=True)

try:
    from pennylane_quandela import PennylaneConverter
except ModuleNotFoundError as e:
    assert e.name == "pennylane_quandela"
    pytest.skip("need `pennylane_quandela` module", allow_module_level=True)

from perceval import BasicState, StateVector, Circuit
import perceval.components.unitary_components as comp


def test_basic_circuit_h():
    converter = PennylaneConverter(num_qubits=1)
    qc = [qml.Hadamard(wires=[0])]
    pc = converter.convert(qc)
    c = pc.linear_circuit()
    sd = pc.source_distribution
    assert len(sd) == 1
    assert sd[StateVector('|1,0>')] == 1
    assert len(c._components) == 1
    assert isinstance(c._components[0][1], Circuit) and len(c._components[0][1]._components) == 1
    c0 = c._components[0][1]._components[0][1]
    assert isinstance(c0, comp.BS)
    assert c0._convention == comp.BSConvention.H

def test_basic_circuit_double_h():
    converter = PennylaneConverter(num_qubits=1)
    qc = [
        qml.Hadamard(wires=[0]),
        qml.Hadamard(wires=[0])
    ]
    pc = converter.convert(qc)
    assert pc.source_distribution[StateVector('|1,0>')] == 1
    assert len(pc._components) == 2

def test_basic_circuit_s():
    converter = PennylaneConverter(num_qubits=1)
    qc = [
        qml.S(wires=[0])
    ]
    pc = converter.convert(qc)
    assert pc.source_distribution[StateVector('|1,0>')] == 1
    assert len(pc._components) == 1
    assert isinstance(pc._components[0][1], Circuit) and len(pc._components[0][1]._components) == 1
    r0 = pc._components[0][1]._components[0][0]
    c0 = pc._components[0][1]._components[0][1]
    assert r0 == (1,)
    assert isinstance(c0, comp.PS)

def test_basic_circuit_swap_direct():
    converter = PennylaneConverter(num_qubits=2)
    qc = [
        qml.SWAP(wires=[0, 1])
    ]
    pc = converter.convert(qc)
    assert pc.source_distribution[StateVector('|1,0,1,0>')] == 1
    assert len(pc._components) == 1
    r0, c0 = pc._components[0]
    assert r0 == [0, 1, 2, 3]
    assert isinstance(c0, comp.PERM)
    assert c0.perm_vector == [2, 3, 0, 1]

def test_basic_circuit_swap_indirect():
    converter = PennylaneConverter(num_qubits=2)
    qc = [
        qml.SWAP(wires=[1, 0])
    ]
    pc = converter.convert(qc)
    assert pc.source_distribution[StateVector('|1,0,1,0>')] == 1
    assert len(pc._components) == 1
    r0, c0 = pc._components[0]
    assert r0 == [0, 1, 2, 3]
    assert isinstance(c0, comp.PERM)
    assert c0.perm_vector == [2, 3, 0, 1]

def test_cnot_1_heralded():
    converter = PennylaneConverter(num_qubits=2)
    qc = [
        qml.Hadamard(wires=[0]),
        qml.CNOT(wires=[0, 1])
    ]
    pc = converter.convert(qc, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,1,1>')] == 1
    assert len(pc._components) == 2  # should be BS.H//CNOT

def test_cnot_1_inverse_heralded():
    converter = PennylaneConverter(num_qubits=2)
    qc = [
        qml.Hadamard(wires=[0]),
        qml.CNOT(wires=[1, 0])
    ]
    pc = converter.convert(qc, use_postselection=False)
    assert pc.circuit_size == 6
    assert pc.m == 4
    assert pc.source_distribution[StateVector('|1,0,1,0,1,1>')] == 1
    assert len(pc._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = pc._components[1][1]
    assert isinstance(perm1, comp.PERM)
    perm2 = pc._components[3][1]
    assert isinstance(perm2, comp.PERM)
    # check that ports are correctly connected
    assert perm1.perm_vector == [2, 3, 0 ,1]
    assert perm2.perm_vector == [2, 3, 0, 1]

def test_cnot_2_heralded():
    converter = PennylaneConverter(num_qubits=3)
    qc = [
        qml.Hadamard(wires=[0]),
        qml.CNOT(wires=[0, 2])
    ]
    pc = converter.convert(qc, use_postselection=False)
    assert pc.circuit_size == 8
    assert pc.m == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,1,0,1,1>')] == 1
    assert len(pc._components) == 4
    # should be BS//PERM//CNOT//PERM
    perm1 = pc._components[1][1]
    assert isinstance(perm1, comp.PERM)
    perm2 = pc._components[3][1]
    assert isinstance(perm2, comp.PERM)
    # check that ports are correctly connected
    assert perm1.perm_vector == [4, 5, 0, 1, 2, 3]
    assert perm2.perm_vector == [2, 3, 4, 5, 0, 1]

def test_cnot_1_postprocessed():
    converter = PennylaneConverter(num_qubits=2)
    qc = [
        qml.Hadamard(wires=[0]),
        qml.CNOT(wires=[0, 1])
    ]
    pc = converter.convert(qc, use_postselection=True)
    assert pc.circuit_size == 6
    assert pc.source_distribution[StateVector('|1,0,1,0,0,0>')] == 1
    assert len(pc._components) == 2  # No permutation needed, only H and CNOT components exist in the Processor
    # should be BS//CNOT

def test_cnot_postprocess():
    converter = PennylaneConverter(num_qubits=2)
    qc = [
        qml.Hadamard(wires=[0]),
        qml.CNOT(wires=[0, 1])
    ]
    pc = converter.convert(qc, use_postselection=True)
    bsd_out = pc.probs()['results']
    assert len(bsd_out) == 2

    # We should be able to continue the circuit with 1-qubit gates even with a post-selected CNOT
    qc.append(qml.Hadamard(wires=[0]))

    # Derivation of number of qbits directly from PennyLane quantum
    # circuit is not implemented yet, we need to update it manually
    converter.num_qubits = converter.num_qubits + 1

    pc = converter.convert(qc, use_postselection=True)
    assert isinstance(pc._components[-1][1]._components[0][1], comp.BS)

def test_cnot_herald():
    converter = PennylaneConverter(num_qubits=2)
    qc = [
        qml.Hadamard(wires=[0]),
        qml.CNOT(wires=[0, 1])
    ]
    pc = converter.convert(qc, True)
    bsd_out = pc.probs()['results']
    assert bsd_out[BasicState("|1,0,0,1>")] + bsd_out[BasicState("|0,1,1,0>")] < 2e-5
    assert bsd_out[BasicState("|1,0,1,0>")] + bsd_out[BasicState("|0,1,0,1>")] > 0.99
    assert len(bsd_out) == 4

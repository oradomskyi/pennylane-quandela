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
PennyLane-Quandela quantum circuit converter
========================================

This module contains a conveter class for constructing Quandela Perceval Processor using
PennyLane desciption of a quantum circuit.

"""
from perceval.components import Processor, Source
from perceval.converters.abstract_converter import AGateConverter
from perceval import Catalog, catalog as default_catalog

from pennylane.operation import Operation
from pennylane import matrix as to_matrix

NAME_PENNYLANE_CNOT = 'CNOT'
NAME_PENNYLANE_BARRIER = 'barrier'
NAME_PENNYLANE_MEASURE = 'measure'


class PennylaneConverter(AGateConverter):
    r"""PennyLane quantum circuit to perceval processor converter.

    :param catalog: a component library to use for the conversion. It must contain CNOT gates.
    :param backend_name: backend name used in the converted processor (default SLOS)
    :param source: the source used as input for the converted processor (default perfect source).
    :param num_qubits: the number of qubits in the circuit to convert, automatic inference of this
        number from list[pennylane.Operation] is not implemented yet.
    """

    # Derivation of the number of qubits from input circuit, list[pennylane.Operation],
    # is not implemented yet.
    _num_qubits = None

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the input circuit"""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, new_num_qubits) -> None:
        """Set the number of qubits in the circuit to convert"""
        self._num_qubits = new_num_qubits

    def __init__(self, catalog: Catalog = None,
        backend_name: str = None, # it is Backend responsibility to handle None
        source: Source = None,
        num_qubits: int = None):
        """Initializes PennyLane to Perceval gate converter"""

        if catalog and source:
            super().__init__(catalog, backend_name, source)
        elif catalog and not source:
            super().__init__(catalog, backend_name, Source())
        elif not catalog and source:
            super().__init__(default_catalog, backend_name, source)
        else:
            # not catalog and not source:
            super().__init__(default_catalog, backend_name, Source())

        self._num_qubits = num_qubits

    def count_qubits(self, gate_circuit: list[Operation] = None) -> int:
        """Implementation necessary for the converter base class"""
        if self.num_qubits is None:
            raise ValueError(
                "Please set the number of qubits in the circuit"+
                " you want to convert. Automatic inference of num_qubits"+
                " directly from the PennyLane circuit described as"+
                " list[pennylane.Operation] is not implemented yet."+
                "\n\nAdd argument num_qubits to the PennylaneConverter"+
                " like this `converter = PennylaneConverter(num_qubits=2)`."+
                "\nYou also can set it directly `converter.num_qubits = 2`\n")

        if gate_circuit is None:
            raise ValueError(f"Cannot derive number of qubits from a crcuit which is {gate_circuit}.")

        return self.num_qubits

    def convert(self, gate_circuit: list[Operation], use_postselection: bool = True) -> Processor:
        r"""Convert a PennyLane quantum circuit into a `Processor`.

        :param gate_circuit: quantum-based PennyLane circuit
        :type gate_circuit: list[pennylane.Operation]
        :param use_postselection: when True, uses a `postprocessed CNOT` as the last gate. 
            Otherwise, uses only `heralded CNOT`
        :return: the converted processor
        """
        n_cnot = 0  # count the number of CNOT gates in circuit - needed to find the num. heralds
        for instruction in gate_circuit:
            if instruction.name in NAME_PENNYLANE_CNOT:
                n_cnot += 1

        self._configure_processor(gate_circuit=gate_circuit, qname='q')

        for instruction in gate_circuit:
            # barrier has no effect
            if instruction.name in NAME_PENNYLANE_BARRIER:
                continue

            # some limitation in the conversion, in particular measure
            if instruction.name in NAME_PENNYLANE_MEASURE:
                raise ValueError(f'Cannot convert {NAME_PENNYLANE_MEASURE} gate')

            num_gate_qubits = len(instruction.wires)
            if num_gate_qubits == 1:
                # one mode gate
                instruction_mat = to_matrix(instruction) # np.ndarray
                gate = self._create_generic_1_qubit_gate(instruction_mat)
                gate.name = instruction.name
                # pennylane.wires.Wires is an info about qubit positions
                self._converted_processor.add(instruction.wires[0] * 2, gate.copy())
            else:
                if num_gate_qubits > 2:
                    # only 2 qubit gates
                    raise ValueError("Gates with number of Qubits higher than 2 not implemented")

                c_idx = instruction.wires[0] * 2
                c_data = instruction.wires[1] * 2

                self._create_2_qubit_gates_from_catalog(instruction.name,
                    n_cnot,
                    c_idx,
                    c_data,
                    use_postselection)

        self.apply_input_state()
        return self._converted_processor

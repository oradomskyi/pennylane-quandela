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

"""Tests for the PercevalDevice class"""

import pytest

from pennylane_perceval import PercevalDevice

from perceval.backends import BackendFactory
from perceval.providers import ProviderFactory

class TestPercevalDevice:
    """Tests for the PercevalDevice base class."""

    @pytest.mark.parametrize("wires", [1, 3])
    @pytest.mark.parametrize("shots", [1, 100])
    def test_init_no_kwargs(self, wires , shots):
        """Tests that the device is properly initialized."""

        device = PercevalDevice(wires=wires, shots=shots)

        assert device.num_wires == wires
        assert device.shots == shots
        assert device.backend.name in 'SLOS'

        assert device.provider is None
        assert device._platform_name is None
        assert device._backend_name is None
        assert device._provider_name is None
        assert device._api_token is None

    @pytest.mark.parametrize("wires", [1, 3])
    @pytest.mark.parametrize("shots", [1, 100])
    @pytest.mark.parametrize("backend_name", BackendFactory.list())
    def test_init_with_backend(self, wires, shots, backend_name):
        """Tests that the device is properly initialized."""

        device = PercevalDevice(wires=wires, shots=shots, backend=backend_name)

        assert device.num_wires == wires
        assert device.shots == shots
        assert device.backend.name in backend_name
        assert device._backend_name in backend_name

    def test_reset(self):
        """Tests that device is properly reset"""

        device = PercevalDevice(1, 1)
        device.reset()

        assert device._circuit is None
        assert device._processor is None

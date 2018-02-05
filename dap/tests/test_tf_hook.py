# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the tensorflow HookeanSpring module

pydoc:dap.tf.hooke
"""

import numpy as np
import tensorflow as tf
from ase.build import molecule
from dap.tf.hooke import HookeanSpring

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestHookeanSpring(tf.test.TestCase):
  """Tests for the model HookeanSpring Calculator."""

  def test_default(self):
    """Test energy and forces with default settings."""
    atoms = molecule('H2')
    atoms.set_calculator(HookeanSpring())
    e0 = 0.5 * 1.0 * (1 - atoms.get_distance(0, 1))**2
    e1 = atoms.get_potential_energy()
    self.assertTrue(e0 == e1)
    fmag = np.abs(-1.0 * (1 - atoms.get_distance(0, 1)))
    f = atoms.get_forces()
    self.assertTrue(fmag == np.linalg.norm(f[0]))
    self.assertTrue(fmag == np.linalg.norm(f[1]))

  def test_custom(self):
    """Test energy and forces with custom settings."""
    atoms = molecule('H2')
    k, x0 = 1.5, 0.9
    atoms.set_calculator(HookeanSpring(k=k, x0=x0))
    e0 = 0.5 * k * (x0 - atoms.get_distance(0, 1))**2
    e1 = atoms.get_potential_energy()
    self.assertTrue(e0 == e1)
    fmag = np.abs(-k * (x0 - atoms.get_distance(0, 1)))
    f = atoms.get_forces()
    self.assertTrue(fmag == np.linalg.norm(f[0]))
    self.assertTrue(fmag == np.linalg.norm(f[1]))

  def test_custom(self):
    """Test energy and forces with custom settings."""
    atoms = molecule('H2O')
    atoms.set_calculator(HookeanSpring())
    with self.assertRaisesRegexp(Exception,
                                 'You can only use a two atom systems'):
      e = atoms.get_potential_energy()

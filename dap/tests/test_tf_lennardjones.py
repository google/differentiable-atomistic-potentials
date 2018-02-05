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
"""Tests for the tensorflow LennardJones module.

pydoc:dap.tf.lennardjones.
"""

import tensorflow as tf
from ase.build import bulk
from ase.calculators.lj import LennardJones
from dap.tf.lennardjones import (energy, forces, stress, energy_1way,
                                 forces_1way)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestLJ(tf.test.TestCase):
  """Tests comparing the TF version and ASE LennardJones implementations.

  Tests the energy, forces and stress in different crystal structures with
  different symmetries and repeats. The atoms are rattled in each structure to
  further break symmetry.

  """

  def test_energy(self):
    for structure in ('fcc', 'bcc', 'hcp', 'diamond', 'sc'):
      for repeat in ((1, 1, 1), (1, 2, 3)):
        for a in [3.0, 4.0]:
          atoms = bulk('Ar', structure, a=a).repeat(repeat)
          atoms.rattle()
          atoms.set_calculator(LennardJones())
          ase_energy = atoms.get_potential_energy()

          lj_energy = energy(atoms.positions, atoms.cell)
          init = tf.global_variables_initializer()
          with self.test_session() as sess:
            sess.run(init)
            self.assertAllClose(ase_energy, lj_energy.eval())

  def test_forces(self):
    for structure in ('fcc', 'bcc', 'hcp', 'diamond', 'sc'):
      for repeat in ((1, 1, 1), (1, 2, 3)):
        for a in [3.0, 4.0]:
          atoms = bulk('Ar', structure, a=a).repeat(repeat)
          atoms.rattle()
          atoms.set_calculator(LennardJones())
          ase_forces = atoms.get_forces()

          lj_forces = forces(atoms.positions, atoms.cell)
          init = tf.global_variables_initializer()
          with self.test_session() as sess:
            sess.run(init)
            self.assertAllClose(ase_forces, lj_forces.eval())

  def test_stress(self):
    for structure in ('fcc', 'bcc', 'hcp', 'diamond', 'sc'):
      for repeat in ((1, 1, 1), (1, 2, 3)):
        for a in [3.0, 4.0]:
          atoms = bulk('Ar', structure, a=a).repeat(repeat)
          atoms.rattle()
          atoms.set_calculator(LennardJones())
          ase_stress = atoms.get_stress()

          lj_stress = stress(atoms.positions, atoms.cell)
          init = tf.global_variables_initializer()
          with self.test_session() as sess:
            sess.run(init)
            self.assertAllClose(ase_stress, lj_stress.eval())


class TestLJ_1way(tf.test.TestCase):
  """Tests comparing the TF version and ASE LennardJones implementations.

  Tests the energy, forces and stress in different crystal structures with
  different symmetries and repeats. The atoms are rattled in each structure to
  further break symmetry.

  """

  def test_energy_1way(self):
    """Test oneway list"""
    import numpy as np
    import warnings

    warnings.filterwarnings('ignore')    
    for structure in ('hcp', 'fcc', 'bcc', 'hcp', 'diamond', 'sc'):
      for repeat in ((2, 1, 1), (1, 1, 1), (2, 2, 2), (1, 2, 3)):
        for a in [2.0, 3.0]:
          print(f'{structure} {repeat} {a}')
          atoms = bulk('Ar', structure, a=a).repeat(repeat)
          atoms.rattle()
          atoms.set_calculator(LennardJones())
          ase_energy = atoms.get_potential_energy()

          lj_energy = energy_1way(atoms.positions, atoms.cell)
          init = tf.global_variables_initializer()
          with self.test_session() as sess:
            sess.run(init)
            if not np.isclose(ase_energy, lj_energy.eval(), 1e-3):
              print(f'NOT CLOSE: {structure} {repeat} {a}',
                    np.isclose(ase_energy, lj_energy.eval(), 1e-3),
                    ase_energy, lj_energy.eval())

            self.assertAllClose(ase_energy, lj_energy.eval())

  def test_forces_1way(self):
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    for structure in ('fcc', 'bcc', 'hcp', 'diamond', 'sc'):
      for repeat in ((1, 1, 1), (2, 2, 2), (2, 1, 1), (1, 2, 3)):
        for a in [2.0, 3.0]:
          print(f'{structure} {repeat} {a}')
          atoms = bulk('Ar', structure, a=a).repeat(repeat)
          atoms.rattle()
          atoms.set_calculator(LennardJones())
          ase_forces = atoms.get_forces()

          lj_forces = forces_1way(atoms.positions, atoms.cell)
          init = tf.global_variables_initializer()
          with self.test_session() as sess:
            sess.run(init)
            self.assertAllClose(ase_forces, lj_forces.eval())

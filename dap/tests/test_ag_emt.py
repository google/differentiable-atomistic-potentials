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
"""Tests for the autograd EMT module.

pydoc:dap.ag.emt.
"""

import unittest
import autograd.numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from dap.ag.emt import (parameters, energy, forces, stress)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestAgEmt(unittest.TestCase):
  def test0(self):
    atoms = bulk('Cu', 'fcc', a=3.61).repeat((2, 2, 2))
    atoms.rattle(0.2)
    atoms.set_calculator(EMT())

    ase_e = atoms.get_potential_energy()
    age = energy(parameters, atoms.positions, atoms.numbers, atoms.cell)
    self.assertEqual(ase_e, age)

    ase_f = atoms.get_forces()
    ag_f = forces(atoms.positions, atoms.numbers, atoms.cell)
    self.assertTrue(np.allclose(ase_f, ag_f))

  def test_energy_forces(self):
    for structure in ('fcc', 'bcc', 'hcp', 'diamond', 'sc'):
      for repeat in ((1, 1, 1), (1, 2, 3)):
        for a in [3.0, 4.0]:
          atoms = bulk('Cu', structure, a=a).repeat(repeat)
          atoms.rattle()
          atoms.set_calculator(EMT())
          ase_energy = atoms.get_potential_energy()
          emt_energy = energy(parameters, atoms.positions, atoms.numbers, atoms.cell)
          self.assertEqual(ase_energy, emt_energy)

          ase_f = atoms.get_forces()
          ag_f = forces(atoms.positions, atoms.numbers, atoms.cell)
          self.assertTrue(np.allclose(ase_f, ag_f))

  def test_stress(self):
    for structure in ('fcc', 'bcc', 'hcp', 'diamond', 'sc'):
      for repeat in ((1, 1, 1), (1, 2, 3)):
        for a in [3.0, 4.0]:
          atoms = bulk('Cu', structure, a=a).repeat(repeat)
          atoms.rattle()
          atoms.set_calculator(EMT())

          # Numerically calculate the ase stress
          d = 1e-9  # a delta strain

          ase_stress = np.empty((3, 3)).flatten()
          cell0 = atoms.cell

          # Use a finite difference approach that is centered.
          for i in [0, 4, 8, 5, 2, 1]:
            strain_tensor = np.zeros((3, 3))
            strain_tensor = strain_tensor.flatten()
            strain_tensor[i] = d
            strain_tensor = strain_tensor.reshape((3, 3))
            strain_tensor += strain_tensor.T
            strain_tensor /= 2
            strain_tensor += np.eye(3, 3)

            cell = np.dot(strain_tensor, cell0.T).T
            positions = np.dot(strain_tensor, atoms.positions.T).T
            atoms.cell = cell
            atoms.positions = positions
            ep = atoms.get_potential_energy()

            strain_tensor = np.zeros((3, 3))
            strain_tensor = strain_tensor.flatten()
            strain_tensor[i] = -d
            strain_tensor = strain_tensor.reshape((3, 3))
            strain_tensor += strain_tensor.T
            strain_tensor /= 2
            strain_tensor += np.eye(3, 3)

            cell = np.dot(strain_tensor, cell0.T).T
            positions = np.dot(strain_tensor, atoms.positions.T).T
            atoms.cell = cell
            atoms.positions = positions
            em = atoms.get_potential_energy()

            ase_stress[i] = (ep - em) / (2 * d) / atoms.get_volume()

          ase_stress = np.take(ase_stress.reshape((3, 3)), [0, 4, 8, 5, 2, 1])

          ag_stress = stress(parameters, atoms.positions, atoms.numbers, atoms.cell)

          # I picked the 0.03 tolerance here. I thought it should be closer, but
          # it is a simple numerical difference I am using for the derivative,
          # and I am not sure it is totally correct.
          self.assertTrue(np.all(np.abs(ase_stress - ag_stress) <= 0.03),
                          f'''
ase: {ase_stress}
ag : {ag_stress}')
diff {ase_stress - ag_stress}
''')

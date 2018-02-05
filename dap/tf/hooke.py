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
"""A prototype ASE calculator using Tensorflow."""

import numpy as np
import tensorflow as tf
from ase.calculators.calculator import Calculator, all_changes


class HookeanSpring(Calculator):
  """A simple Hookean spring."""
  implemented_properties = ['energy', 'forces']
  default_parameters = {'k': 1.0, 'x0': 1.0}

  def __init__(self, **kwargs):
    Calculator.__init__(self, **kwargs)
    print(kwargs)
    self.sess = tf.Session()

  def __del__(self):
    self.sess.close()

  def _energy(self, positions):
    """Compute the energy of the spring.
    
    Parameters
    ----------
    positions: an array (2, 3) of positions.
    
    Returns
    -------
    a tensor containing the energy.
    """

    k = self.parameters.k
    x0 = self.parameters.x0
    positions = tf.convert_to_tensor(positions)
    x = tf.norm(positions[1] - positions[0])
    e = 0.5 * k * (x - x0)**2
    # We return a tensor here and eval it later.
    return e

  def _forces(self, positions):
    """Compute the forces on the atoms.
    
    Parameters
    ----------
    positions: an array (2, 3) of positions.
    
    Returns
    -------
    a tensor containing the forces
    """
    positions = tf.convert_to_tensor(positions)
    f = tf.gradients(-self._energy(positions), positions)[0]
    # We return a tensor here and eval it later.
    return f

  def calculate(self,
                atoms=None,
                properties=['energy'],
                system_changes=all_changes):
    """Run the calculator.
    You don't usually call this, it is usually called by methods on the Atoms.
    """

    if len(atoms) != 2:
      raise Exception('You can only use a two atom systems.')
    Calculator.calculate(self, atoms, properties, system_changes)

    self.results['energy'] = self._energy(
        atoms.positions).eval(session=self.sess)
    self.results['forces'] = self._forces(
        atoms.positions).eval(session=self.sess)

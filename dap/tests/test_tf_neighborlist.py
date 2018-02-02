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
"""Tests for the tensorflow neighborlist module.

pydoc:dap.tf.neighborlist
"""

import numpy as np
import tensorflow as tf
from ase.build import bulk
from ase.neighborlist import NeighborList
from dap.tf.neighborlist import (get_distances, get_neighbors_oneway)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestNeighborlist(tf.test.TestCase):
  """Tests comparing the TF version and ASE neighborlist implementations.

  """

  def test_basic(self):
    """Basic neighborlist test in TF"""
    a = 3.6
    Rc = a / np.sqrt(2) / 2
    atoms = bulk('Cu', 'fcc', a=a)

    nl = NeighborList(
        [Rc] * len(atoms), skin=0.01, self_interaction=False, bothways=True)
    nl.update(atoms)

    distances = get_distances({
        'cutoff_radius': 2 * Rc
    }, atoms.positions, atoms.cell, np.ones((len(atoms), 1)))

    mask = (distances <= 2 * Rc) & (distances > 0)
    tf_nneighbors = tf.reduce_sum(tf.cast(mask, tf.int32), axis=[1, 2])

    with self.test_session():
      for i, atom in enumerate(atoms):
        inds, disps = nl.get_neighbors(i)
        ase_nneighbors = len(inds)
        self.assertEqual(ase_nneighbors, tf_nneighbors.eval()[i])

  def test_structure_repeats(self):
    'Check several structures and repeats for consistency with ase.'
    for repeat in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 3)):
      for structure in ('fcc', 'bcc', 'sc', 'hcp', 'diamond'):
        a = 3.6
        # Float tolerances are tricky. The 0.01 in the next line is important.
        # This test fails without it due to subtle differences in computed
        # positions.
        Rc = 2 * a + 0.01
        atoms = bulk('Cu', structure, a=a).repeat(repeat)
        nl = NeighborList(
            [Rc] * len(atoms), skin=0.0, self_interaction=False, bothways=True)
        nl.update(atoms)
        distances = get_distances({
            'cutoff_radius': 2 * Rc
        }, atoms.positions, atoms.cell, np.ones((len(atoms), 1)))

        mask = (distances <= 2 * Rc) & (distances > 0)
        tf_nneighbors = tf.reduce_sum(tf.cast(mask, tf.int32), axis=[1, 2])

        with self.test_session():
          for i, atom in enumerate(atoms):
            inds, disps = nl.get_neighbors(i)
            ase_nneighbors = len(inds)
            self.assertEqual(ase_nneighbors, tf_nneighbors.eval()[i])

            # These are the indices of each neighbor in the atom list.
            tf_inds = tf.where(mask[i])[:, 0].eval()
            self.assertCountEqual(inds, tf_inds)

  def test_atom_types(self):
    """Tests if the neighbor indices agree with ase.

    This is important to find the
    chemical element associated with a specific neighbor.

    """
    a = 3.6
    Rc = a / np.sqrt(2) / 2 + 0.01

    atoms = bulk('Cu', 'fcc', a=a).repeat((3, 1, 1))
    atoms[1].symbol = 'Au'

    nl = NeighborList(
        [Rc] * len(atoms), skin=0.01, self_interaction=False, bothways=True)
    nl.update(atoms)
    nns = [nl.get_neighbors(i) for i in range(len(atoms))]
    ase_nau = [np.sum(atoms.numbers[inds] == 79) for inds, offs in nns]

    au_mask = tf.convert_to_tensor(atoms.numbers == 79, tf.int32)

    distances = get_distances({
        'cutoff_radius': 2 * Rc
    }, atoms.positions, atoms.cell)
    mask = (distances <= (2 * Rc)) & (distances > 0)

    nau = tf.reduce_sum(tf.cast(mask, tf.int32) * au_mask[:, None], [1, 2])

    with self.test_session():
      self.assertTrue(np.all(ase_nau == nau.eval()))


class TestOneWayNeighborlist(tf.test.TestCase):

  def test0(self):
    a = 3.6
    Rc = 5
    atoms = bulk('Cu', 'bcc', a=a).repeat((1, 1, 1))
    atoms.rattle(0.02)
    nl = NeighborList(
        [Rc] * len(atoms), skin=0.0, self_interaction=False, bothways=False)
    nl.update(atoms)

    inds, N = get_neighbors_oneway(
        atoms.positions, atoms.cell, 2 * Rc, skin=0.0)

    with self.test_session() as sess:
      inds, N = sess.run([inds, N])

      for i in range(len(atoms)):
        ase_inds, ase_offs = nl.get_neighbors(i)

        these_inds = np.array([x[1] for x in inds if x[0] == i])
        these_offs = np.array([N[x[2]] for x in inds if x[0] == i])

        self.assertAllClose(ase_inds, these_inds)
        self.assertAllClose(ase_offs, these_offs)

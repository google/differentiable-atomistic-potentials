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

import os
import unittest
import numpy as np
import numpy.testing as npt
from ase.build import molecule
from amp.descriptor.gaussian import *
from amp.utilities import hash_images

from dap.ag.neighborlist import get_distances
from dap.py.bpnn import pad, cosine_cutoff, G2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestPadding(unittest.TestCase):

  def test0(self):
    a = np.ones((2, 3))

    padded_array = pad(a, (3, 3))

    res = np.ones((3, 3))
    res[2, :] = 0

    self.assertTrue(np.all(padded_array == res))


class TestG2(unittest.TestCase):
  """Run comparisons of the python G2 functions against Amp."""

  def setUp(self):
    # amp saves these and we want to eliminate them
    os.system('rm -fr amp-data*')

  def tearDown(self):
    os.system('rm -fr amp-data*')

  def test_radial_G2(self):
    atoms = molecule('H2O')
    atoms.cell = 100 * np.eye(3)

    # Amp setup

    sf = {
        'H': make_symmetry_functions(['H', 'O'], 'G2', [0.05, 1.0]),
        'O': make_symmetry_functions(['H', 'O'], 'G2', [0.05, 1.0])
    }

    descriptor = Gaussian(Gs=sf)
    images = hash_images([atoms], ordered=True)
    descriptor.calculate_fingerprints(images)

    fparray = []
    for index, hash in enumerate(images.keys()):
      for fp in descriptor.fingerprints[hash]:
        fparray += [fp[1]]
    fparray = np.array(fparray)

    # This module setup
    positions = atoms.positions
    cell = atoms.cell
    atom_mask = [[1] for atom in atoms]

    numbers = list(np.unique(atoms.numbers))

    species_mask = np.stack(
        [[atom.number == el for atom in atoms] for el in numbers],
        axis=1).astype(int)

    config = {'cutoff_radius': 6.5}
    d, _ = get_distances(positions, cell, config['cutoff_radius'])

    g0 = G2(0, 0.05, 0.0)
    g1 = G2(1, 0.05, 0.0)
    g2 = G2(0, 1.0, 0.0)
    g3 = G2(1, 1.0, 0.0)

    # This builds the array of fingerprints
    this = np.concatenate(
        (g0(config, d, atom_mask, species_mask),
         g1(config, d, atom_mask, species_mask),
         g2(config, d, atom_mask, species_mask),
         g3(config, d, atom_mask, species_mask)),
        axis=1)

    npt.assert_almost_equal(fparray, this, 5)

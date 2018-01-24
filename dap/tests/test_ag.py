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

import unittest
import autograd.numpy as np
from ase.build import bulk
from ase.neighborlist import NeighborList
from ase.calculators.lj import LennardJones

from dap.ag.neighborlist import get_distances, get_neighbors_oneway
from dap.ag.lennardjones import energy, forces, stress


class TestNeighborList(unittest.TestCase):

    def test0(self):
        "Check a variety of repeats with different symmetries."
        a = 3.6
        for cutoff_radius in np.linspace(a / 2, 5 * a, 10):
            for rep in ((1, 1, 1),
                        (2, 1, 1),
                        (1, 2, 1),
                        (1, 1, 2),
                        (1, 2, 2),
                        (2, 1, 1),
                        (2, 2, 1),
                        (2, 2, 2),
                        (1, 2, 3)):
                atoms = bulk('Cu', 'fcc', a=a).repeat(rep)

                nl = NeighborList([cutoff_radius / 2] * len(atoms), skin=0.01,
                                  self_interaction=False, bothways=True)
                nl.update(atoms)
                nns_ase = [len(nl.get_neighbors(i)[0])
                           for i in range(len(atoms))]

                d = get_distances(atoms.positions, atoms.cell, cutoff_radius)
                inds = (d <= (cutoff_radius + 0.01)) & (d > 0.00)
                nns = inds.sum((1, 2))

                self.assertTrue(np.all(nns_ase == nns))


class TestNeighborListOneWay(unittest.TestCase):
    def test0(self):
        a = 3.6
        for rep in ((1, 1, 1),
                    (2, 1, 1),
                    (1, 2, 1),
                    (1, 1, 2),
                    (1, 2, 2),
                    (2, 1, 1),
                    (2, 2, 1),
                    (2, 2, 2),
                    (1, 2, 3)):
            for cutoff_radius in np.linspace(a / 2, 5 * a, 5):
                atoms = bulk('Cu', 'fcc', a=a).repeat(rep)

                nl = NeighborList([cutoff_radius / 2] * len(atoms), skin=0.0,
                                  self_interaction=False, bothways=False)
                nl.update(atoms)

                neighbors, displacements = get_neighbors_oneway(atoms.positions,
                                                                atoms.cell,
                                                                cutoff_radius)

                for i in range(len(atoms)):
                    an, ad = nl.get_neighbors(i)
                    # These are the indices
                    self.assertEqual(len(neighbors[i]), len(an))
                    self.assertCountEqual(neighbors[i], an)

                    # I am not sure how to test for the displacements.


class TestLennardJones(unittest.TestCase):
    def test_fcc(self):
        "Check structures and repeats with different symmetries."
        for struct in ['fcc', 'bcc', 'diamond']:
            for repeat in [(1, 1, 1),
                           (1, 1, 2),
                           (1, 2, 1),
                           (2, 1, 1),
                           (1, 2, 3),
                           (2, 2, 2)]:
                atoms = bulk('Cu', struct, a=3.7).repeat(repeat)
                atoms.rattle(0.02)
                atoms.set_calculator(LennardJones())

                ase_energy = atoms.get_potential_energy()
                lj_energy = energy({}, atoms.positions, atoms.cell)

                self.assertAlmostEqual(ase_energy, lj_energy)

                lj_forces = forces({}, atoms.positions, atoms.cell)
                self.assertTrue(np.allclose(atoms.get_forces(),
                                            lj_forces))

                lj_stress = stress({}, atoms.positions, atoms.cell)

                self.assertTrue(np.allclose(atoms.get_stress(),
                                            lj_stress))

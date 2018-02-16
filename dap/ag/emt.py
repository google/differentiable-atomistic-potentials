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
"""Differentiable effective medium theory potential

Adapted from https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/emt.html#EMT

"""

import autograd.numpy as np
from autograd import elementwise_grad
from ase.data import chemical_symbols
from ase.units import Bohr
from dap.ag.neighborlist import get_neighbors_oneway

parameters = {
    #      E0     s0    V0     eta2    kappa   lambda  n0
    #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
    'Al': (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),
    'Cu': (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),
    'Ag': (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),
    'Au': (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),
    'Ni': (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030),
    'Pd': (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),
    'Pt': (-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802),
    # extra parameters - just for fun ...
    'H': (-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547),
    'C': (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322),
    'N': (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222),
    'O': (-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850)
}

beta = 1.809  # (16 * pi / 3)**(1.0 / 3) / 2**0.5, preserve historical rounding


def energy(positions, numbers, cell, strain=np.zeros((3, 3))):
  """Compute the energy using Effective medium theory.

    Parameters
    ----------

    positions : array of floats. Shape = (natoms, 3)

    numbers : array of integers of atomic numbers (natoms,)

    cell: array of unit cell vectors. Shape = (3, 3)

    strain: array of strains to apply to cell. Shape = (3, 3)

    Returns
    -------
    energy : float
    """
  strain_tensor = np.eye(3) + strain
  cell = np.dot(strain_tensor, cell.T).T
  positions = np.dot(strain_tensor, positions.T).T

  par = {}
  rc = 0.0

  relevant_pars = parameters
  maxseq = max(par[1] for par in relevant_pars.values()) * Bohr
  rc = rc = beta * maxseq * 0.5 * (np.sqrt(3) + np.sqrt(4))
  rr = rc * 2 * np.sqrt(4) / (np.sqrt(3) + np.sqrt(4))
  acut = np.log(9999.0) / (rr - rc)

  rc_list = rc + 0.5
  for Z in numbers:
    if Z not in par:
      sym = chemical_symbols[Z]
      if sym not in parameters:
        raise NotImplementedError('No EMT-potential for {0}'.format(sym))
      p = parameters[sym]
      s0 = p[1] * Bohr
      eta2 = p[3] / Bohr
      kappa = p[4] / Bohr
      x = eta2 * beta * s0
      gamma1 = 0.0
      gamma2 = 0.0
      for i, n in enumerate([12, 6, 24]):
        r = s0 * beta * np.sqrt(i + 1)
        x = n / (12 * (1.0 + np.exp(acut * (r - rc))))
        gamma1 += x * np.exp(-eta2 * (r - beta * s0))
        gamma2 += x * np.exp(-kappa / beta * (r - beta * s0))

      par[Z] = {
          'E0': p[0],
          's0': s0,
          'V0': p[2],
          'eta2': eta2,
          'kappa': kappa,
          'lambda': p[5] / Bohr,
          'n0': p[6] / Bohr**3,
          'rc': rc,
          'gamma1': gamma1,
          'gamma2': gamma2
      }

  ksi = {}
  for s1, p1 in par.items():
    ksi[s1] = {}
    for s2, p2 in par.items():
      ksi[s1][s2] = p2['n0'] / p1['n0']

  natoms = len(positions)
  sigma1 = [0.0] * natoms

  all_neighbors, all_offsets = get_neighbors_oneway(
      positions, cell, rc_list, skin=0.0)

  # Calculate
  energy = 0.0

  for a1 in range(natoms):
    Z1 = numbers[a1]
    p1 = par[Z1]
    _ksi = ksi[Z1]
    neighbors, offsets = all_neighbors[a1], all_offsets[a1]
    offsets = np.dot(offsets, cell)
    for a2, offset in zip(neighbors, offsets):
      d = positions[a2] + offset - positions[a1]
      r = np.sqrt(np.dot(d, d))
      if r < rc_list:
        Z2 = numbers[a2]
        p2 = par[Z2]
        x = np.exp(acut * (r - rc))
        theta = 1.0 / (1.0 + x)
        y1 = (0.5 * p1['V0'] * np.exp(-p2['kappa'] * (
            r / beta - p2['s0'])) * _ksi[Z2] / p1['gamma2'] * theta)
        y2 = (0.5 * p2['V0'] * np.exp(-p1['kappa'] * (
            r / beta - p1['s0'])) / _ksi[Z2] / p2['gamma2'] * theta)
        energy = energy - (y1 + y2)

        sa = (
            np.exp(-p2['eta2'] *
                   (r - beta * p2['s0'])) * _ksi[Z2] * theta / p1['gamma1'])
        sigma1[a1] = sigma1[a1] + sa

        sa = (
            np.exp(-p1['eta2'] *
                   (r - beta * p1['s0'])) / _ksi[Z2] * theta / p2['gamma1'])

        sigma1[a2] = sigma1[a2] + sa

  for a in range(natoms):
    Z = numbers[a]
    p = par[Z]
    try:
      ds = -np.log(sigma1[a] / 12) / (beta * p['eta2'])
    except (OverflowError, ValueError):
      energy -= p['E0']
      continue
    x = p['lambda'] * ds
    y = np.exp(-x)
    z = 6 * p['V0'] * np.exp(-p['kappa'] * ds)
    energy += p['E0'] * ((1 + x) * y - 1) + z

  return energy


def forces(positions, numbers, cell):
  """Compute the forces of an EMT system.

    Parameters
    ----------

    positions : array of floats. Shape = (natoms, 3)

    numbers : array of integers of atomic numbers (natoms,)

    cell: array of unit cell vectors. Shape = (3, 3)

    Returns
    -------
    forces : an array of forces. Shape = (natoms, 3)

    """
  dEdR = elementwise_grad(energy, 0)
  return -dEdR(positions, numbers, cell)


def stress(positions, numbers, cell, strain=np.zeros((3, 3))):
  """Compute the stress on an EMT system.

    Parameters
    ----------

    positions : array of floats. Shape = (natoms, 3)

    numbers : array of integers of atomic numbers (natoms,)

    cell: array of unit cell vectors. Shape = (3, 3)

    Returns
    -------
    stress : an array of stress components. Shape = (6,)
    [sxx, syy, szz, syz, sxz, sxy]

    """
  dEdst = elementwise_grad(energy, 3)

  volume = np.abs(np.linalg.det(cell))

  der = dEdst(positions, numbers, cell, strain)
  result = (der + der.T) / 2 / volume
  return np.take(result, [0, 4, 8, 5, 2, 1])

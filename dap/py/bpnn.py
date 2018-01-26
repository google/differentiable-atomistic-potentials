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

import numpy as np


def pad(array, shape):
  """Returns a zero-padded array with shape."""
  array = np.array(array)
  p = np.zeros(shape)

  r, c = array.shape
  p[:r, :c] = array
  return p


def cosine_cutoff(config, distances, atom_mask=None):
  """Cosine cutoff function.

    Parameters
    ----------

    config: a dictionary containing 'cutoff_radius' as a float
    distances: the distance array from get_distances
    atom_mask: An array of ones for atoms, 0 for non-atoms. 
      Defaults to all atoms.

  """
  cutoff_radius = config.get('cutoff_radius', 6.0)
  distances = np.array(distances)

  if atom_mask is None:
    atom_mask = np.ones((len(distances), 1))
  else:
    atom_mask = np.array(atom_mask)

  cc = 0.5 * (np.cos(np.pi * (distances / cutoff_radius)) + 1.0)
  cc *= (distances <= cutoff_radius) & (distances > 0.0)
  cc *= atom_mask
  cc *= atom_mask[:, None]
  return cc


def G2(species_index, eta, Rs):
  """G2 function generator.

    This is a radial function between an atom and atoms with some chemical
    symbol. It is defined in cite:khorshidi-2016-amp, eq. 6. This version is
    scaled a little differently than the one Behler uses.

    Parameters
    ----------

    species_index : integer
      species index for this function. Elements that do not have this index will
      be masked out

    eta : float
      The gaussian width

    Rs : float
      The gaussian center or shift

    Returns
    -------
    The g2 function with the cosine_cutoff function integrated into it.

  """

  def g2(config, distances, atom_mask, species_masks):
    distances = np.array(distances)
    atom_mask = np.array(atom_mask)
    species_masks = np.array(species_masks)
    # Mask out non-species contributions
    smask = species_masks[:, species_index][:, None]
    distances *= smask
    distances *= atom_mask
    distances *= atom_mask[:, None]

    Rc = config.get('cutoff_radius', 6.5)
    result = np.where(distances > 0,
                      np.exp(-eta * ((distances - Rs)**2 / Rc**2)), 0.0)

    result *= cosine_cutoff(config, distances, atom_mask)
    gsum = np.sum(result, (1, 2))
    return gsum[:, None]

  g2.__desc__ = 'g2({species_index}, eta={eta}, Rs={Rs})'.format(**locals())
  return g2

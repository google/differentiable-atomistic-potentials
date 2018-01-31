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
"""Neighborlist functions for tensorflow."""

import os
import warnings
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_distances(config, positions, cell, atom_mask=None, bothways=True):
  """Get distances to neighboring atoms with periodic boundary conditions.

  The way this function works is it tiles a volume with unit cells to at least
  fill a sphere with a radius of cutoff_radius. That means some atoms will be
  outside the cutoff radius. Those are included in the results. Then we get
  distances to all atoms in the tiled volume. This is always the same number for
  every atom, so we have consistent sized arrays.

  Args:
    config: A dictionary containing 'cutoff_radius' with a float value.
    positions: array-like or Tensor shape=(numatoms, 3)
      Array of cartesian coordinates of atoms in a unit cell.
    cell: array-like shape=(3, 3)
      Array of unit cell vectors in cartesian basis. Each row is a unit cell
      vector.
    atom_mask: array-like (numatoms,)
      ones for atoms, zero for padded positions. If None, defaults to all ones
    cutoff_radius: float
      The cutoff_radius we want atoms within.

  Returns:
    distances: shape=(maxnatoms, maxnatoms, nunitcells) containing the distances
    between all pairs of atoms in the tiled volume.

  Related
  -------

  pydoc:pymatgen.core.lattice.Lattice.get_points_in_sphere

  """
  with tf.name_scope('get_distances'):
    positions = tf.convert_to_tensor(positions)
    cell = tf.convert_to_tensor(cell)

    if atom_mask is None:
      natoms = positions.get_shape()[0]
      atom_mask = tf.ones((natoms, 1), dtype=cell.dtype)
    else:
      atom_mask = tf.convert_to_tensor(atom_mask, dtype=cell.dtype)
    cutoff_radius = tf.convert_to_tensor(
        config['cutoff_radius'], dtype=cell.dtype)
    # Next we get the inverse unit cell, which will be used to compute the
    # unit cell offsets required to tile space inside the sphere.
    inverse_cell = tf.matrix_inverse(cell)

    fractional_coords = tf.mod(
        tf.matmul(positions, inverse_cell), tf.ones_like(positions))

    num_cell_repeats = cutoff_radius * tf.norm(inverse_cell, axis=0)

    mins = tf.reduce_min(tf.floor(fractional_coords - num_cell_repeats), axis=0)
    maxs = tf.reduce_max(tf.ceil(fractional_coords + num_cell_repeats), axis=0)

    # Now we generate a set of cell offsets. We start with the repeats in each
    # unit cell direction.
    if bothways:
      v0_start = mins[0]
    else:
      v0_start = 0
    v0_range = tf.range(v0_start, maxs[0])
    v1_range = tf.range(mins[1], maxs[1])
    v2_range = tf.range(mins[2], maxs[2])

    # Then we expand them in each dimension
    xhat = tf.constant([1.0, 0.0, 0.0], dtype=inverse_cell.dtype)
    yhat = tf.constant([0.0, 1.0, 0.0], dtype=inverse_cell.dtype)
    zhat = tf.constant([0.0, 0.0, 1.0], dtype=inverse_cell.dtype)

    v0_range = v0_range[:, None] * xhat[None, :]
    v1_range = v1_range[:, None] * yhat[None, :]
    v2_range = v2_range[:, None] * zhat[None, :]

    # And combine them to get an offset vector for each cell
    offsets = (
        v0_range[:, None, None] + v1_range[None, :, None] +
        v2_range[None, None, :])

    offsets = tf.reshape(offsets, (-1, 3))

    if not bothways:
      # pick out the ones that don't meet the one-way conditions
      n1 = offsets[:, 0]
      n2 = offsets[:, 1]
      n3 = offsets[:, 2]

      mask = tf.logical_not(
        tf.logical_and(
          tf.equal(n1, 0), (tf.logical_or(
            tf.less(n2, 0),
            (tf.logical_and(tf.equal(n2, 0), (tf.less(n3, 0))))))))

      offsets = tf.boolean_mask(offsets, mask)

    # Now we have a vector of unit cell offsets (offset_index, 3) in the inverse
    # unit cell basis. We convert that to cartesian coordinate offsets here.
    cart_offsets = tf.matmul(offsets, cell)

    # we need to offset each atom coordinate by each offset.
    # This array is (atom_index, offset, 3)
    shifted_cart_coords = positions[:, None] + cart_offsets[None, :]

    # Next, we subtract each position from the array of positions.
    # This leads to (atom_i, atom_j, positionvector, xhat)
    relative_positions = shifted_cart_coords - positions[:, None, None]

    # This is the distance squared. This leads to (atom_i, atom_j, distance2)
    distances2 = tf.reduce_sum(relative_positions**2, axis=3)

    # We zero out masked distances.
    distances2 *= atom_mask
    distances2 *= atom_mask[:, None]

    # We do not mask out the values greater than cutoff_radius here. That is
    # done later in the energy function. The zero masking here is due to the
    # fact that the gradient of the square_root at x=0 is nan, so we have to
    # avoid the zeros. Here we replace the zeros temporarily with ones, take the
    # sqrt, and then return the right parts.
    zeros = tf.equal(distances2, 0.0)
    adjusted = tf.where(zeros, tf.ones_like(distances2), distances2)
    distance = tf.sqrt(adjusted)
    return tf.where(zeros, tf.zeros_like(distance), distance)


def get_neighbors(config, atom_index, positions, cell,
                  atom_mask=None, bothways=True):
  pass

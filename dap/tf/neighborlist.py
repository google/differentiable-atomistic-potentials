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

import numpy as np
import tensorflow as tf


def get_distances(config, positions, cell, atom_mask=None):
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
    v0_range = tf.range(mins[0], maxs[0])
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


def get_neighbors_oneway(positions, cell, cutoff_radius, skin=0.01,
                         strain=None):
  """A one-way neighbor list.

    Parameters
    ----------

    positions: atomic positions. array-like (natoms, 3)
    cell: unit cell. array-like (3, 3)
    cutoff_radius: Maximum radius to get neighbor distances for. float
    skin: A tolerance for the cutoff_radius. float
    strain: array-like (3, 3)

    Returns
    -------
    indices, offsets

    """
  positions = tf.convert_to_tensor(positions)
  cell = tf.convert_to_tensor(cell)
  if strain is None:
    strain = tf.zeros_like(cell)
  else:
    strain = tf.convert_to_tensor(strain)

  strain_tensor = tf.eye(3, dtype=cell.dtype) + strain
  cell = tf.transpose(tf.matmul(strain_tensor, tf.transpose(cell)))
  positions = tf.transpose(tf.matmul(strain_tensor, tf.transpose(positions)))
  inverse_cell = tf.matrix_inverse(cell)
  h = 1 / tf.norm(inverse_cell, axis=0)
  N = tf.floor(2 * cutoff_radius / h) + 1

  scaled = tf.matmul(positions, inverse_cell)
  scaled0 = tf.matmul(positions, inverse_cell) % 1.0

  offsets = tf.cast(tf.round(scaled - scaled0), tf.int32)

  positions0 = positions + tf.matmul(tf.cast(offsets, cell.dtype), cell)
  natoms = positions.get_shape().as_list()[0]
  indices = tf.range(natoms)

  v0_range = tf.range(0, N[0] + 1)
  v1_range = tf.range(-N[1], N[1] + 1)
  v2_range = tf.range(-N[2], N[2] + 1)

  xhat = tf.constant([1, 0, 0], dtype=cell.dtype)
  yhat = tf.constant([0, 1, 0], dtype=cell.dtype)
  zhat = tf.constant([0, 0, 1], dtype=cell.dtype)

  v0_range = v0_range[:, None] * xhat[None, :]
  v1_range = v1_range[:, None] * yhat[None, :]
  v2_range = v2_range[:, None] * zhat[None, :]

  N = (
      v0_range[:, None, None] + v1_range[None, :, None] +
      v2_range[None, None, :])

  N = tf.cast(tf.reshape(N, (-1, 3)), tf.int32)

  # pick out the ones that don't meet the one-way conditions
  n1 = N[:, 0]
  n2 = N[:, 1]
  n3 = N[:, 2]

  mask = tf.logical_not(
      tf.logical_and(
          tf.equal(n1, 0), (tf.logical_or(
              tf.less(n2, 0),
              (tf.logical_and(tf.equal(n2, 0), (tf.less(n3, 0))))))))

  N = tf.boolean_mask(N, mask)
  noffsets = tf.shape(N)[0]

  cartesian_displacements = tf.matmul(tf.cast(N, dtype=cell.dtype), cell)

  def get_ath_neighbors(a):
    with tf.variable_scope('neighbors', reuse=tf.AUTO_REUSE):
      var = tf.get_variable('neighbor_{}'.format(a), (1,), dtype=tf.int32)
      return var

  neighbors = [get_ath_neighbors(i) for i in range(natoms)]

  n = tf.constant(0)

  def offset_cond(n):
    return n < noffsets

  def offset_body(n):
    displacement = cartesian_displacements[n]

    for a in range(natoms):
      d = positions0 + displacement - positions0[a]
      mask = tf.reduce_sum(d**2, 1) < (cutoff_radius + skin)**2
      inds = tf.boolean_mask(indices, mask)

      n1, n2, n3 = tf.unstack(N[n])
      inds = tf.cond(
          tf.reduce_all([tf.equal(n1, 0),
                         tf.equal(n2, 0),
                         tf.equal(n3, 0)]),
          true_fn=lambda: tf.boolean_mask(inds, inds > a),
          false_fn=lambda: inds)

      vara = get_ath_neighbors(a)
      ath_neighbors = tf.assign(vara,
                                tf.concat([vara, inds], axis=0),
                                validate_shape=False)

    with tf.control_dependencies([ath_neighbors]):
      return tf.add(n, 1)

  n = tf.while_loop(offset_cond, offset_body, loop_vars=[n])

  with tf.control_dependencies([n]):
    return neighbors

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

from collections import namedtuple
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


import numpy as np


def get_neighbors_oneway(positions,
                         cell,
                         cutoff_distance,
                         skin=0.01,
                         strain=np.zeros((3, 3))):
  """Oneway neighborlist.


  Returns
  -------

  indices_tuples: a list of tuples (atom_index, neighbor_index, offset_index),
  i.e. the atom at neighbor_index is a neighbor of the atom at atom_index and it
  is located at the offset in offset_index.



  Adapted from
  https://wiki.fysik.dtu.dk/ase/_modules/ase/neighborlist.html#NeighborList.

  """
  positions = tf.convert_to_tensor(positions)
  cell = tf.convert_to_tensor(cell)
  strain = tf.convert_to_tensor(strain, dtype=cell.dtype)

  strain_tensor = tf.eye(3, dtype=cell.dtype) + strain
  positions = tf.transpose(tf.matmul(strain_tensor, tf.transpose(positions)))
  cell = tf.transpose(tf.matmul(strain_tensor, tf.transpose(cell)))

  inverse_cell = tf.matrix_inverse(cell)
  h = 1 / tf.norm(inverse_cell, axis=0)
  N = tf.floor(cutoff_distance / h) + 1

  #  N = tf.Print(N, [N], ' tf  N: ')

  scaled = tf.matmul(positions, inverse_cell)
  scaled0 = tf.matmul(positions, inverse_cell) % 1.0

  offsets = tf.round(scaled0 - scaled)
  #  offsets = tf.Print(offsets, [offsets], ' tf offsets:', summarize=100)

  positions0 = positions + tf.matmul(offsets, cell)
  # positions0 = tf.Print(
  #     positions0, [positions0], ' tf positions: ', summarize=100)

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

  N = tf.reshape(N, (-1, 3))

  n1 = N[:, 0]
  n2 = N[:, 1]
  n3 = N[:, 2]

  mask = tf.logical_not(
      tf.logical_and(
          tf.equal(n1, 0.0),
          tf.logical_or(
              tf.less(n2, 0.0),
              tf.logical_and(tf.equal(n2, 0.0), tf.less(n3, 0.0)))))
  N = tf.boolean_mask(N, mask)
  #  N = tf.Print(N, [N], 'tf offsets', summarize=20)
  noffsets = tf.shape(N)[0]
  natoms = positions.get_shape().as_list()[0]

  indices = tf.range(natoms)
  # Finally, we have to run two loops, one over the offsets, and one over the
  # positions. We will accumulate the neighbors as we go. I like to save all the
  # loop vars in one place.
  # n is a counter for offsets
  # a is a counter for atom index
  # k is a counter for neighbors
  # indices contains a list of (a, index): the index of the neighbor of atom a.
  # displacements is a list of (n1, n2, n3) corresponding to displacements for
  # each neighbor.
  LV = namedtuple('LoopVariables', 'n, a, k, indices, distances, displacements')

  lv0 = LV(
      tf.constant(0, dtype=tf.int32),  # n counter
      tf.constant(0, dtype=tf.int32),  # a counter
      tf.constant(0, dtype=tf.int32),  # offset counter
      tf.Variable(tf.zeros((0, 2), dtype=tf.int32), dtype=tf.int32),  # indices
      # distances
      tf.Variable(tf.zeros((0,), dtype=positions.dtype), dtype=positions.dtype),
      tf.Variable(tf.zeros((0, 3), dtype=tf.int32),
                  dtype=tf.int32)  # displacements
  )

  shiv = LV(
      tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None),
      tf.TensorShape([None, 2]), tf.TensorShape(None), tf.TensorShape([None,
                                                                       3]))

  def outer_cond(nt):
    return tf.less(nt.n, noffsets)

  def outer_body(nt):
    """This is the loop over the offsets."""
    n = nt.n

    n1, n2, n3 = tf.unstack(N[n])

    displacement = tf.matmul(tf.cast(N[n][None, :], dtype=cell.dtype), cell)

    # displacement = tf.Print(displacement, [n, displacement],
    # 'tf displacement: ')

    # Now we loop over each atom
    def inner_cond(nt):
      return tf.less(nt.a, natoms)

    def inner_body(nt):
      """This is a loop over each atom."""
      _p = positions0 + displacement - positions0[nt.a]
      _p2 = tf.reduce_sum(_p**2, axis=1)
      _m0 = tf.equal(_p2, 0.0)
      _mp = tf.where(_m0, tf.ones_like(_p2), _p2)
      _d = tf.sqrt(_mp)
      d = tf.where(_m0, tf.zeros_like(_p2), _d)

      #d = tf.norm(positions0 + displacement - positions0[nt.a], axis=1)
      i = tf.boolean_mask(indices,
                          tf.logical_and(d > 0.0, d < (cutoff_distance + skin)))

      # ug. you have to specify the shape here since i, and hence m is not know
      # in advance. Without it you get:

      # "Number of mask dimensions must be specified, even if some
      # dimensions" ValueError: Number of mask dimensions must be specified,
      # even if some dimensions are None. E.g. shape=[None] is ok, but
      # shape=None is not.

      def self_interaction():
        m = tf.greater(i, nt.a)
        m.set_shape([None])
        return tf.boolean_mask(i, m)

      i = tf.cond(
          tf.reduce_all([tf.equal(n1, 0),
                         tf.equal(n2, 0),
                         tf.equal(n3, 0)]),
          true_fn=self_interaction,
          false_fn=lambda: i)

      # Now we need to add tuples of (nt.a, ind) for ind in i if there is
      # anything in i, and also the index of the offset.

      n_inds = tf.shape(i)[0]

      disp = N[n][None, :]
      disp += tf.gather(offsets, i)
      disp -= offsets[nt.a]

      def nind_cond(nt):
        return tf.less(nt.k, n_inds)

      def nind_body(nt):
        tups = tf.concat(
            [
                nt.indices,
                [(
                    nt.a,  # atom to get neighbors for
                    i[nt.k],  # index of neighbor equivalent atom.
                )]
            ],
            axis=0)

        dists = tf.concat([nt.distances, [d[nt.k]]], axis=0)

        disps = tf.concat(
            [nt.displacements, [tf.cast(disp[nt.k], tf.int32)]], axis=0)

        return LV(nt.n, nt.a, nt.k + 1, tups, dists, disps),

      nt, = tf.while_loop(nind_cond, nind_body, [nt], [shiv])
      return LV(nt.n, nt.a + 1, 0, nt.indices, nt.distances, nt.displacements),

    nt, = tf.while_loop(inner_cond, inner_body, [nt], [shiv])

    return LV(nt.n + 1, 0, 0, nt.indices, nt.distances, nt.displacements),

  lv1, = tf.while_loop(outer_cond, outer_body, [lv0], [shiv])

  return lv1.indices, lv1.distances, lv1.displacements

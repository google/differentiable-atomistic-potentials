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
"""A Tensorflow implementation of a LennardJones Potential for a single element.

This is a standalone module by design.
"""

import numpy as np
import tensorflow as tf
from dap.tf.neighborlist import get_neighbors_oneway
from ase.calculators.calculator import Calculator, all_changes


def get_Rij(positions, cell, mask, cutoff_radius):
  """Get distances to neighboring atoms with periodic boundary conditions.

  The way this function works is it tiles space with unit cells to at least fill
  a sphere with a radius of cutoff_radius. That means some atoms will be outside
  the cutoff radius. Those are included in the results. Then we get distances to
  all atoms in the tiled space. This is always the same number for every atom,
  so we have consistent sized arrays.

  This function is specific to the Lennard Jones potential as noted in the
  comments below.

  Args:
    positions: array-like shape=(numatoms, 3)
      Array of cartesian coordinates of atoms in a unit cell.
    cell: array-like shape=(3, 3)
      Array of unit cell vectors in cartesian basis. Each row is a unit cell
      vector.
    mask: array-like (numatoms,)
      ones for atoms, zero for padded positions
    cutoff_radius: float
      The cutoff_radius we want atoms within.

  Returns:
    A flattened array of distances to all the neighbors.

  Notes:

  One of the distances is equal to 0.0, which corresponds to Rii. This distance
  is problematic for the gradients, which are undefined for these points. I have
  not found a masking strategy to eliminate these points while keeping the
  gradients besides the one used here. This is not an issue with other
  potentials that don't have a 1/r form like this one does.

  This code was adapted from:
  Related: pydoc:pymatgen.core.lattice.Lattice.get_points_in_sphere

  """
  with tf.name_scope("get_Rij"):
    positions = tf.convert_to_tensor(positions)
    cell = tf.convert_to_tensor(cell)
    mask = tf.convert_to_tensor(mask, dtype=cell.dtype)

    with tf.name_scope("get_offsets"):
      # Next we get the reciprocal unit cell, which will be used to compute the
      # unit cell offsets required to tile space inside the sphere.
      inverse_cell = tf.matrix_inverse(cell)

      fcoords = tf.mod(
          tf.matmul(positions, inverse_cell), tf.ones_like(positions))

      recp_len = tf.norm(inverse_cell, axis=0)

      nmax = cutoff_radius * recp_len

      mins = tf.reduce_min(tf.floor(fcoords - nmax), axis=0)
      maxs = tf.reduce_max(tf.ceil(fcoords + nmax), axis=0)

      # Now we generate a set of cell offsets. We start with the repeats in each
      # unit cell direction.
      arange = tf.range(mins[0], maxs[0])
      brange = tf.range(mins[1], maxs[1])
      crange = tf.range(mins[2], maxs[2])

      # Then we expand them in each dimension
      xhat = tf.constant([1.0, 0.0, 0.0], dtype=inverse_cell.dtype)
      yhat = tf.constant([0.0, 1.0, 0.0], dtype=inverse_cell.dtype)
      zhat = tf.constant([0.0, 0.0, 1.0], dtype=inverse_cell.dtype)

      arange = arange[:, None] * xhat[None, :]
      brange = brange[:, None] * yhat[None, :]
      crange = crange[:, None] * zhat[None, :]

      # And combine them to get an offset vector for each cell
      offsets = (
          arange[:, None, None] + brange[None, :, None] + crange[None, None, :])

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
    Rij2 = tf.reduce_sum(relative_positions**2, axis=3)

    # We zero out masked distances. This is subtle. We have to zero out parts of
    # two dimensions. First, all the entries in the first dimension which are
    # not atoms must be zeroed, and then, all the entries in the second
    # dimension which aren't atoms have to be zeroed.
    Rij2 *= mask[:, None] * mask[:, None, None]
    # Since we assume the atoms are all the same we can flatten it. It turns out
    # that the array will get flattened anyway because of the boolean mask in
    # the return. This effectively removes elements in some of the subarrays so
    # the shape is no longer constant, causing the array to be flattened.
    Rij2 = tf.reshape(Rij2, [-1])

    # We exclude the self-interaction by only considering atoms with a distance
    # greater than 0. For this potential, it is necessary to do this here to
    # avoid nan's in the gradients.
    #
    # It is not necessary to take the square root here, since we later compute
    # 1/Rij^6. But, this function was originally intended to be used for other
    # potentials where Rij is used directly, so we do that here.
    #
    # We do not mask out the values greater than cutoff_radius here. That is
    # done later in the energy function.
    return tf.sqrt(tf.boolean_mask(Rij2, Rij2 > 0.0))


def energy(positions, cell, mask=None, strain=None):
  """Compute the energy of a Lennard-Jones system.

  Args:
    positions: array-like shape=(numatoms, 3)
      Array of cartesian coordinates of atoms in a unit cell.
    cell: array-like shape=(3, 3)
      Array of unit cell vectors in cartesian basis. Each row is a unit cell
      vector.
    mask: array-like (numatoms,)
      ones for atoms, zero for padded positions.
    strain: array-like shape=(3, 3)
      Array of strains to compute the energy at.

  Returns: float
    The total energy from the Lennard Jones potential.
  """

  with tf.name_scope("LennardJones"):
    with tf.name_scope("setup"):
      positions = tf.convert_to_tensor(positions)
      cell = tf.convert_to_tensor(cell)
      if mask is None:
        mask = tf.ones_like(positions[:, 0])
      mask = tf.convert_to_tensor(mask)
      if strain is None:
        strain = tf.zeros_like(cell)

      strain = tf.convert_to_tensor(strain)

      strained_cell = tf.matmul(cell, tf.eye(3, dtype=cell.dtype) + strain)
      strained_positions = tf.matmul(positions,
                                     tf.eye(3, dtype=cell.dtype) + strain)

      with tf.variable_scope("sigma", reuse=tf.AUTO_REUSE):
        sigma = tf.get_variable(
            "sigma",
            dtype=cell.dtype,
            initializer=tf.constant(1.0, dtype=cell.dtype))

      with tf.variable_scope("epsilon", reuse=tf.AUTO_REUSE):
        epsilon = tf.get_variable(
            "epsilon",
            dtype=cell.dtype,
            initializer=tf.constant(1.0, dtype=cell.dtype))

      rc = 3 * sigma

    with tf.name_scope("calculate_energy"):
      e0 = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
      energy = 0.0

      d = get_Rij(strained_positions, strained_cell, mask, rc)

      neighbor_mask = tf.less_equal(d, tf.ones_like(d) * rc)
      energy -= e0 * tf.reduce_sum(tf.cast(neighbor_mask, e0.dtype))
      c6 = (sigma**2 / tf.boolean_mask(d, neighbor_mask)**2)**3
      c12 = c6**2
      energy += tf.reduce_sum(4 * epsilon * (c12 - c6))

      return energy / 2.0


def forces(positions, cell, mask=None, strain=None):
  """Compute the forces.

  Args:
    positions: array-like shape=(numatoms, 3)
      Array of cartesian coordinates of atoms in a unit cell.
    cell: array-like shape=(3, 3)
      Array of unit cell vectors in cartesian basis. Each row is a unit cell 
      vector.
    mask: array-like (numatoms,)
      ones for atoms, zero for padded positions.
    strain: array-like shape=(3, 3)
      Array of strains to compute the energy at.

  Returns:
    array: shape=(natoms, 3)
  """
  with tf.name_scope("forces"):
    positions = tf.convert_to_tensor(positions)
    cell = tf.convert_to_tensor(cell)
    if mask is None:
      mask = tf.ones_like(positions[:, 0])
    mask = tf.convert_to_tensor(mask)
    if strain is None:
      strain = tf.zeros_like(cell)
    return tf.gradients(-energy(positions, cell, mask, strain), positions)[0]


def stress(positions, cell, mask=None, strain=None):
  """Compute the stress.

  Args:
    positions: array-like shape=(numatoms, 3)
      Array of cartesian coordinates of atoms in a unit cell.
    cell: array-like shape=(3, 3)
      Array of unit cell vectors in cartesian basis. Each row is a unit cell 
      vector.
    mask: array-like (numatoms,)
      ones for atoms, zero for padded positions
    strain: array-like shape=(3, 3)
      Array of strains to compute the stress at.

  Returns:
    The stress components [sxx, syy, szz, syz, sxz, sxy]
    array: shape=(6,)
  """
  with tf.name_scope("stress"):
    with tf.name_scope("setup"):
      positions = tf.convert_to_tensor(positions)
      cell = tf.convert_to_tensor(cell)
      if mask is None:
        mask = tf.ones_like(positions[:, 0])
      mask = tf.convert_to_tensor(mask)
      if strain is None:
        strain = tf.zeros_like(cell)

    with tf.name_scope("get_stress"):
      volume = tf.abs(tf.matrix_determinant(cell))
      stress = tf.gradients(energy(positions, cell, mask, strain), strain)[0]
      stress /= volume
      return tf.gather(tf.reshape(stress, (9,)), [0, 4, 8, 5, 2, 1])


def energy_batch(POSITIONS,
                 CELLS,
                 MASKS,
                 strain=np.zeros((3, 3), dtype=np.float64)):
  """A batched version of `energy'.

  Args:
    POSITIONS: array-like shape=(batch, maxnumatoms, 3)
      batched array of position arrays. Each position array should be padded 
      if there are fewer atoms than maxnatoms.
    CELLS: array-like shape=(batch, 3, 3)
    MASKS: array-like shape=(batch, maxnatoms)
    strain: array-like shape=(3, 3)
      Array of strains to compute the stress at.

  Returns:
    energies: array-like shape=(batch,)
  """
  return tf.convert_to_tensor([
      energy(positions, cell, mask, strain)
      for positions, cell, mask in zip(POSITIONS, CELLS, MASKS)
  ])


def forces_batch(POSITIONS,
                 CELLS,
                 MASKS,
                 strain=np.zeros((3, 3), dtype=np.float64)):
  """A batched version of `forces'.
 
  Args:
    POSITIONS: array-like shape=(batch, maxnumatoms, 3)
      batched array of position arrays. Each position array should be padded
      if there are fewer atoms than maxnatoms.
    CELLS: array-like shape=(batch, 3, 3)
    MASKS: array-like shape=(batch, maxnatoms)
    strain: array-like shape=(3, 3)
      Array of strains to compute the stress at.

  Returns:
    forces: array-like shape=(batch, maxnatoms, 3)
  """
  return tf.convert_to_tensor([
      forces(positions, cell, mask, strain)
      for positions, cell, mask in zip(POSITIONS, CELLS, MASKS)
  ])


def stress_batch(POSITIONS,
                 CELLS,
                 MASKS,
                 strain=np.zeros((3, 3), dtype=np.float64)):
  """A batched version of `stress'.

  Args:
    POSITIONS: array-like shape=(batch, maxnumatoms, 3)
      batched array of position arrays. Each position array should be padded
      if there are fewer atoms than maxnatoms.
    CELLS: array-like shape=(batch, 3, 3)
    MASKS: array-like shape=(batch, maxnatoms)
    strain: array-like shape=(3, 3)
      Array of strains to compute the stress at.

  Returns:
    stresses: array-like shape=(batch, 6)"""
  return tf.convert_to_tensor([
      stress(positions, cell, mask, strain)
      for positions, cell, mask in zip(POSITIONS, CELLS, MASKS)
  ])


# * One way list class


class LennardJones(Calculator):
  implemented_properties = ["energy", "forces", "stress"]

  default_parameters = {"sigma": 1.0, "epsilon": 1.0}

  def __init__(self, **kwargs):
    Calculator.__init__(self, **kwargs)
    self.sess = tf.Session()

    with tf.variable_scope("sigma", reuse=tf.AUTO_REUSE):
      sigma = tf.get_variable(
          "sigma",
          dtype=tf.float64,
          initializer=tf.constant(self.parameters.sigma, dtype=tf.float64))

    with tf.variable_scope("epsilon", reuse=tf.AUTO_REUSE):
      epsilon = tf.get_variable(
          "epsilon",
          dtype=tf.float64,
          initializer=tf.constant(self.parameters.epsilon, dtype=tf.float64))

    self.sigma = sigma
    self.epsilon = epsilon

    self._positions = tf.placeholder(dtype=tf.float64, shape=(None, 3))
    self._cell = tf.placeholder(dtype=tf.float64, shape=(3, 3))
    self._strain = tf.placeholder(dtype=tf.float64, shape=(3, 3))

    with tf.name_scope("LennardJones"):
      with tf.name_scope("setup"):
        strain_tensor = tf.eye(3, dtype=self._cell.dtype) + self._strain
        strained_cell = tf.matmul(self._cell, strain_tensor)
        strained_positions = tf.matmul(self._positions, strain_tensor)

        sigma = self.sigma
        epsilon = self.epsilon
        rc = 3 * sigma

        with tf.name_scope("calculate_energy"):
          e0 = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
          _energy = 0.0

          inds, dists, displacements = get_neighbors_oneway(
              strained_positions, strained_cell, rc)

          m = dists < rc
          m.set_shape([None])
          r2 = tf.boolean_mask(dists, m)**2
          c6 = (sigma**2 / r2)**3
          c12 = c6**2
          n = tf.ones_like(r2)

          _energy -= tf.reduce_sum(e0 * n)
          _energy += tf.reduce_sum(4 * epsilon * (c12 - c6))

          self._energy = tf.identity(_energy, name='_energy')

    with tf.name_scope("forces"):
      f = tf.gradients(-self._energy, self._positions)[0]
      self._forces = tf.identity(tf.convert_to_tensor(f), name='_forces')
          

    with tf.name_scope("stress"):
      with tf.name_scope("get_stress"):
        volume = tf.abs(tf.matrix_determinant(self._cell))
        g = tf.gradients(self._energy, self._strain)
        stress = tf.convert_to_tensor(g[0])
        stress /= volume
        stress = tf.gather(tf.reshape(stress, (9,)), [0, 4, 8, 5, 2, 1])
        self._stress = tf.identity(stress, name='_stress')
      
  def calculate(self,
                atoms=None,
                properties=["energy"],
                system_changes=all_changes):
    """Run the calculator.
    You don't usually call this, it is usually called by methods on the Atoms.
    """
    Calculator.calculate(self, atoms, properties, system_changes)
    self.sess.run(tf.global_variables_initializer())
    self.results["energy"] = self.sess.run(self._energy,
                                      feed_dict={self._positions: atoms.positions,
                                                 self._cell: atoms.cell,
                                                 self._strain: np.zeros_like(atoms.cell)})

    self.results["forces"] = self.sess.run(self._forces,
                                           feed_dict={self._positions: atoms.positions,
                                                      self._cell: atoms.cell,
                                                      self._strain: np.zeros_like(atoms.cell)})

    self.results["stress"] = self.sess.run(self._stress,
                                           feed_dict={self._positions: atoms.positions,
                                                      self._cell: atoms.cell,
                                                      self._strain: np.zeros_like(atoms.cell)})


  def save(self, label):
    "Save the graph and variables."
    saver = tf.train.Saver()
    saver.save(self.sess, label)

  def load(self, label):
    "Load variables from label."
    saver = tf.train.import_meta_graph(label + ".meta")
    self.sess.run(saver.restore(self.sess, label + ".meta"))
    g = tf.get_default_graph()
    self.sigma = g.get_tensor_by_name("sigma:0")
    self.epsilon = g.get_tensor_by_name("epsilon:0")
    print(f'Loaded {self.sigma} and {self.epsilon}')

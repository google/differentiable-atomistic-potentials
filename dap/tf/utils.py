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
"""Tensorflow utilities.
These are mostly functions that bring additional numpy functionality to
Tensorflow.
"""
import itertools
import numpy as np
import tensorflow as tf


def swish(x, beta=1.0):
  """Swish activation function. See https://arxiv.org/abs/1710.05941"""
  with tf.name_scope("swish"):
    return x * tf.nn.sigmoid(beta * x)


def tri(N, M=None, k=0, dtype=tf.float64):
  """
    An array with ones at and below the given diagonal and zeros elsewhere.
    Parameters
    ----------
    N : int
        Number of rows in the array
    M : int, optional
        Number of columns in the array. Defaults to number of rows.
    k : The subdiagonal at and below which the array is filled, optional
    Returns
    -------
    out : tensor of shape (N, M)
    Modeled after pydoc:numpy.tri.
    """

  if M is None:
    M = N

  r1 = tf.range(N)
  r2 = tf.range(-k, M - k)
  return tf.greater_equal(r1[:, None], r2[None, :])


def triu_indices(n, k=0, m=None):
  """Return indices for upper triangle of an (n, m) array.
    Parameters
    ----------
    n : int
      number of rows in the array.
    k : int, optional
      diagonal offset.
    m : int, optional
      number of columns in the array. Defaults to `n`.
    Returns
    -------
    inds : a tensor, shape = (None, 2)
      column 0 is one set of indices, column 1 is the other set.
    modeled after pydoc:numpy.triu_indices.
    """

  result = tf.where(tf.logical_not(tri(n, m, k=k - 1)))
  return result[:, 0], result[:, 1]


def triu_indices_from(arr, k=0):
  """Return the indices for the upper-triangle of arr.
    Parameters
    ----------
    arr : tensor or array.
    k : diagonal.
    see pydoc:numpy.triu_indices_from.
    """
  tensor = tf.convert_to_tensor(arr)
  shape = tensor.get_shape().as_list()
  if len(shape) != 2:
    raise ValueError("Tensor must be 2d")
  return triu_indices(shape[-2], k=k, m=shape[-1])


def tril_indices(n, k=0, m=None):
  """Return indices for lower triangle of an (n, m) array.
    Parameters
    ----------
    n : int
      number of rows in the array.
    k : int, optional
      diagonal offset.
    m : int, optional
      number of columns in the array. Defaults to `n`.
    Returns
    -------
    inds : a tensor, shape = (None, 2)
      column 0 is one set of indices, column 1 is the other set.
    modeled after pydoc:numpy.tril_indices.
    """

  result = tf.where(tri(n, m, k=k))
  return result[:, 0], result[:, 1]


def tril_indices_from(arr, k=0):
  """Return the indices for the lower-triangle of arr.
    Parameters
    ----------
    arr : tensor or array.
    k : diagonal.
    see pydoc:numpy.tril_indices_from.
    """
  tensor = tf.convert_to_tensor(arr)
  shape = tensor.get_shape().as_list()
  if len(shape) != 2:
    raise ValueError("Tensor must be 2d")
  return tril_indices(shape[-2], k=k, m=shape[-1])


def combinations(arr, k):
  """Return tensor of combinations of k elements.
    Parameters
    ----------
    arr : 1D array or tensor
    k : number of elements to make combinations of .
    Returns
    -------
    a 2D tensor of combinations. Each row is a combination, and each element of
    the combination is in the columns.
    Related: pydoc:itertools.combinations
    """
  tensor = tf.convert_to_tensor(arr)

  shape = tensor.get_shape().as_list()
  if len(shape) != 1:
    raise ValueError("Tensor must be 1d")

  N = shape[0]
  inds = np.arange(N)
  combination_indices = [
      combination for combination in itertools.combinations(inds, k)
  ]
  return tf.stack([tf.gather(arr, ind) for ind in combination_indices])


def slices_values_to_sparse_tensor(slices, values, dense_shape):
  """Convert a tensor of slices and corresponding values to a sparse tensor.

  Given a 2D tensor of slices, where each row corresponds to the row the slice
  is from in another tensor, and the columns are the indices in that row, and a
  tensor of corresponding values, create a tf.SparseTensor representation.

  For example, to create a sparse representation of the top_k results:
  >> arr = [[1, 2, 3], [3, 2, 1], [2, 1, 3]]
  >> kv, ki = tf.nn.top_k(arr, k)
  >> sparse_tensor = slices_values_to_sparse_tensor(kv, ki, arr.shape)

  This is useful to then make a dense tensor comprised of those values, with
  some other default value for the rest.

  Here the default other values are zero.

  >> dst = tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False)
  """

  slices = tf.cast(tf.convert_to_tensor(slices), dtype=tf.int64)
  values = tf.convert_to_tensor(values)

  shape = tf.shape(slices, out_type=tf.int64)

  nrows = shape[0]
  row_inds = tf.range(nrows)

  flattened_indices = tf.reshape(slices * nrows + row_inds[:, None], [-1])
  twod_inds = tf.stack(
      [flattened_indices % nrows, flattened_indices // nrows], axis=1)
  return tf.SparseTensor(
      twod_inds, values=tf.reshape(values, [-1]), dense_shape=dense_shape)

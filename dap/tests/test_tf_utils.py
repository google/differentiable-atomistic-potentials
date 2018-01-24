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

"""Tests for tensorflow module."""
import itertools
import numpy as np
import tensorflow as tf

from dap.tf.utils import (tri, triu_indices, tril_indices, triu_indices_from,
                          tril_indices_from, combinations,
                          slices_values_to_sparse_tensor)


class TestTFUtils_tri(tf.test.TestCase):

  def test_tri(self):
    npt = np.tri(3, dtype=np.bool)
    tft = tri(3)
    with self.test_session():
      self.assertTrue(np.all(npt == tft.eval()))

  def test_above(self):
    npt = np.tri(3, k=1, dtype=np.bool)
    tft = tri(3, k=1)
    with self.test_session():
      self.assertTrue(np.all(npt == tft.eval()))

  def test_below(self):
    npt = np.tri(3, k=-1, dtype=np.bool)
    tft = tri(3, k=-1)
    with self.test_session():
      self.assertTrue(np.all(npt == tft.eval()))

  def test_notsquare(self):
    npt = np.tri(3, 4, dtype=np.bool)
    tft = tri(3, 4)
    with self.test_session():
      self.assertTrue(np.all(npt == tft.eval()))

  def test_notsquare_above(self):
    npt = np.tri(3, 4, k=1, dtype=np.bool)
    tft = tri(3, 4, k=1)
    with self.test_session():
      self.assertTrue(np.all(npt == tft.eval()))

  def test_notsquare_below(self):
    npt = np.tri(3, 4, k=-1, dtype=np.bool)
    tft = tri(3, 4, k=-1)
    with self.test_session():
      self.assertTrue(np.all(npt == tft.eval()))


class TestTFUtils_triu(tf.test.TestCase):

  def test_triu(self):
    npu = np.triu_indices(3)
    r0, r1 = triu_indices(3)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))

  def test_triu_k_over(self):
    npu = np.triu_indices(3, k=1)
    r0, r1 = triu_indices(3, k=1)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))

  def test_triu_k_under(self):
    npu = np.triu_indices(3, k=-1)
    r0, r1 = triu_indices(3, k=-1)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))

  def test_triu_nonsquare(self):
    npu = np.triu_indices(3, m=4)
    r0, r1 = triu_indices(3, m=4)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))

  def test_triu_nonsquare_long(self):
    npu = np.triu_indices(3, m=2)
    r0, r1 = triu_indices(3, m=2)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))


class TestTFUtils_tril(tf.test.TestCase):

  def test_tril(self):
    npu = np.tril_indices(3)
    r0, r1 = tril_indices(3)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))

  def test_tril_k_over(self):
    npu = np.tril_indices(3, k=1)
    r0, r1 = tril_indices(3, k=1)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))

  def test_tril_k_under(self):
    npu = np.tril_indices(3, k=-1)
    r0, r1 = tril_indices(3, k=-1)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))

  def test_tril_nonsquare(self):
    npu = np.tril_indices(3, m=4)
    r0, r1 = tril_indices(3, m=4)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))

  def test_tril_nonsquare_long(self):
    npu = np.tril_indices(3, m=2)
    r0, r1 = tril_indices(3, m=2)
    with self.test_session():
      self.assertTrue(np.all(npu[0] == r0.eval()))
      self.assertTrue(np.all(npu[1] == r1.eval()))


class TestTFUtils_triu_indices_from(tf.test.TestCase):

  def test_triu_indices_from(self):

    a = np.zeros((3, 3))
    ref1, ref2 = np.triu_indices_from(a)

    tref1, tref2 = triu_indices_from(a)
    with self.test_session():
      self.assertTrue(np.all(ref1 == tref1.eval()))
      self.assertTrue(np.all(ref2 == tref2.eval()))

  def test_triu_indices_from_kover(self):

    a = np.zeros((3, 3))
    ref1, ref2 = np.triu_indices_from(a, k=1)

    tref1, tref2 = triu_indices_from(a, k=1)
    with self.test_session():
      self.assertTrue(np.all(ref1 == tref1.eval()))
      self.assertTrue(np.all(ref2 == tref2.eval()))

  def test_triu_indices_from_kunder(self):

    a = np.zeros((3, 3))
    ref1, ref2 = np.triu_indices_from(a, k=-1)
    tref1, tref2 = triu_indices_from(a, k=-1)
    with self.test_session():
      self.assertTrue(np.all(ref1 == tref1.eval()))
      self.assertTrue(np.all(ref2 == tref2.eval()))

  def test_triu_indices_from_non2d(self):
    a = np.zeros((3, 3, 3))
    with self.test_session():
      with self.assertRaises(ValueError):
        triu_indices_from(a)


class TestTFUtils_tril_indices_from(tf.test.TestCase):

  def test_tril_indices_from(self):

    a = np.zeros((3, 3))
    ref1, ref2 = np.tril_indices_from(a)

    tref1, tref2 = tril_indices_from(a)
    with self.test_session():
      self.assertTrue(np.all(ref1 == tref1.eval()))
      self.assertTrue(np.all(ref2 == tref2.eval()))

  def test_tril_indices_from_kover(self):

    a = np.zeros((3, 3))
    ref1, ref2 = np.tril_indices_from(a, k=1)

    tref1, tref2 = tril_indices_from(a, k=1)
    with self.test_session():
      self.assertTrue(np.all(ref1 == tref1.eval()))
      self.assertTrue(np.all(ref2 == tref2.eval()))

  def test_tril_indices_from_kunder(self):

    a = np.zeros((3, 3))
    ref1, ref2 = np.tril_indices_from(a, k=-1)
    tref1, tref2 = tril_indices_from(a, k=-1)
    with self.test_session():
      self.assertTrue(np.all(ref1 == tref1.eval()))
      self.assertTrue(np.all(ref2 == tref2.eval()))

  def test_tril_indices_from_non2d(self):
    a = np.zeros((3, 3, 3))
    with self.test_session():
      with self.assertRaises(ValueError):
        tril_indices_from(a)


class TestTFUtils_combinations(tf.test.TestCase):

  def test_combinations_2(self):
    a = [0, 1, 2, 3, 4]
    for k in [2, 3]:
      combs = np.array(list(itertools.combinations(a, k)))
      with self.test_session():
        tf_combs = combinations(a, k).eval()
        self.assertTrue(np.all(combs == tf_combs))

  def test_combinations_non1d(self):
    a = [[0, 1, 2, 3, 4]]
    with self.assertRaises(ValueError):
      with self.test_session():
        combinations(a, 2).eval()


class TestTFUtils_slices(tf.test.TestCase):

  def test(self):
    arr = [[1, 2, 3], [3, 2, 1], [2, 1, 3]]
    k = 2
    kv, ki = tf.nn.top_k(arr, k)
    st = slices_values_to_sparse_tensor(ki, kv, (3, 3))

    ref = tf.SparseTensor([[0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 2]],
                          [2, 3, 3, 2, 2, 3], (3, 3))

    dst = tf.sparse_tensor_to_dense(st, validate_indices=False)
    dref = tf.sparse_tensor_to_dense(
        ref,
        validate_indices=False,
    )

    with self.test_session():
      self.assertTrue(np.all((tf.equal(dst, dref).eval())))

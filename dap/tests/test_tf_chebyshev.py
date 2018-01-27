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
import tensorflow as tf

from dap.tf.chebyshev import (chebvander_py, chebvander)


class TestTFChebyshev(tf.test.TestCase):

  def test_chebvander_py(self):
    """Test the python implementation."""
    x = np.linspace(0, 1, 11)

    for deg in range(1, 49):
      ref = np.polynomial.chebyshev.chebvander(x, deg)
      Tn = chebvander_py(x, deg)
      self.assertTrue(np.allclose(ref, Tn))

  def test_chebvander_tf(self):
    """Test the Tensorflow implementation."""
    x = np.linspace(0, 1, 11)

    with self.test_session():
      for deg in range(1, 49):
        ref = np.polynomial.chebyshev.chebvander(x, deg)
        Tn = chebvander(x, deg)
        self.assertTrue(np.allclose(ref, Tn.eval()))

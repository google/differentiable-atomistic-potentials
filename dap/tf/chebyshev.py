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

"""Chebyshev module for tensorflow.

This implementation was inspired by https://arxiv.org/pdf/1511.09199v1.pdf.


"""

import numpy as np
import tensorflow as tf


def chebvander_py(x, deg):
    """python implementation of pydoc:numpy.polynomial.chebyshev.chebvander.

    This is not intended for usage. It is here to document a transition to the
    tensorflow function below.

    """
    x = np.array(x)
    v = [x * 0 + 1]
    # Use forward recursion to generate the entries.

    x2 = 2 * x
    v += [x]
    for i in range(2, deg + 1):
      v += [v[i - 1] * x2 - v[i - 2]]
    v = np.stack(v)

    roll = list(np.arange(v.ndim))[1:] + [0]
    return v.transpose(roll)


def chebvander(x, deg):
    """TF implementation of pydoc:numpy.polynomial.chebyshev.chebvander."""
    x = tf.convert_to_tensor(x)

    v = [x * 0 + 1]

    x2 = 2 * x
    v += [x]

    for i in range(2, deg + 1):
      v += [v[i - 1] * x2 - v[i - 2]]

    v = tf.stack(v)

    roll = tf.unstack(tf.range(len(v.get_shape().as_list())))
    roll = tf.stack([roll[1:], [roll[0]]], axis=1)
    roll = tf.squeeze(roll)
    v = tf.transpose(v, roll)
    return v

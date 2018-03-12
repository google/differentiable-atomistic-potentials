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
"""Visualization utilities for tensorflow graphs."""

import tensorflow as tf
from graphviz import Digraph


def tf_graph_to_dot(graph):
  ('Adapted from '
   'https://blog.jakuba.net/2017/05/30/tensorflow-visualization.html')
  dot = Digraph()

  for n in g.as_graph_def().node:
    dot.node(n.name, label=n.name)

    for i in n.input:
      dot.edge(i, n.name)
  dot.format = 'svg'
  return dot.pipe().decode('utf-8')


ip = get_ipython()
svg_f = ip.display_formatter.formatters['image/svg+xml']
svg_f.for_type_by_name('tensorflow.python.framework.ops', 'Graph',
                       tf_graph_to_dot)

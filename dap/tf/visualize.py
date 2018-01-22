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
import tempfile
import hashlib
import numpy as np
import os
import webbrowser
from IPython.display import clear_output, Image, display, HTML
import time


def tf_to_dot(graph=None, fname=None, format=None):
    """
    Create an image from a tensorflow graph.

    graph: The tensorflow graph to visualize. Defaults to tf.get_default_graph()
    fname: Filename to save the graph image in
    format: Optional image extension. If you do not use this, the extension is
      derived from the fname.

    Returns an org-mode link to the path where the image is.

    Adapted from https://blog.jakuba.net/2017/05/30/tensorflow-visualization.html

    Note: This can make very large images for complex graphs.
    """

    dot = Digraph()

    if graph is None:
        graph = tf.get_default_graph()

    shapes = {'Const': 'circle',
              'Placeholder': 'oval'}

    for n in graph.as_graph_def().node:
        shape = tuple([dim.size for dim
                       in n.attr['value'].tensor.tensor_shape.dim])
        dot.node(n.name, label=f'{n.name} {shape}',
                 shape=shapes.get(n.op, None))

        for i in n.input:
            dot.edge(i, n.name)

    m = hashlib.md5()
    m.update(str(dot).encode('utf-8'))

    if fname is None:
        fname = 'tf-graph-' + m.hexdigest()

    if format is None:
        base, ext = os.path.splitext(fname)
        fname = base
        format = ext[1:] or 'png'

    dot.format = format
    dot.render(fname)
    os.unlink(fname)
    print(f'{fname}, {format}')
    return f'[[./{fname}.{format}]]'


# Tensorboard visualizations
# Adapted from https://gist.githubusercontent.com/yaroslavvb/97504b8221a8529e7a51a50915206d68/raw/f1473d2873676c0e885b9fbd363c882a7a83b28a/show_graph


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = f"<stripped {size} bytes>".encode('utf-8')
    return strip_def


def show_graph(graph_def=None, browser=True,
               width=1200, height=800,
               max_const_size=32, ungroup_gradients=False):
    """Open a graph in Tensorboard. By default this is done in a browser. If you set
    browser to False, then html will be emitted that shows up in a Jupyter
    notebook.

    """
    if not graph_def:
        graph_def = tf.get_default_graph().as_graph_def()

    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    data = str(strip_def)
    if ungroup_gradients:
        data = data.replace('"gradients/', '"b_')
        #print(data)
    code = """<style>.container {{ width:100% !important; }}</style>
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(data), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:100%;height:100%;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    if browser:

        fh, tmpf = tempfile.mkstemp(prefix='tf-graph-', suffix='.html')
        os.close(fh)
        with open(tmpf, 'w') as f:
            f.write(iframe)
        webbrowser.open('file://' + tmpf)
    else:
        display(HTML(iframe))

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
"""XSF file utilities.
"""
import numpy as np
import re
import ase.io


def read_xsf(xsfile):
  """Return an atoms with energy and forces for the aenet xsfile."""
  atoms = ase.io.read(xsfile)
  calc = atoms.get_calculator()

  with open(xsfile, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
      if line.startswith('# total energy'):
        m = re.findall(r'[-+]?\d*\.\d+|\d+', line)
        energy = float(m[0])
        break
    calc.results['energy'] = energy

  forces = []
  with open(xsfile, 'r') as f:
    while True:
      l = f.readline()
      if not l:
        break
      if l.startswith('PRIMCOORD'):
        break
    count = int(f.readline().split()[0])
    for i in range(count):
      fields = f.readline().split()
      forces += [[float(x) for x in fields[4:]]]
    calc.results['forces'] = np.array(forces)

  return atoms


def write_xsf(xsfile, atoms):
  """Create an aenet compatible xsf file in FNAME for ATOMS.

  fname: a string for the filename.
  atoms: an ase atoms object with an attached calculator containing energy and
  forces.

  returns the string that is written to the file.
  """
  energy = atoms.get_potential_energy()
  forces = atoms.get_forces()

  xsf = ['# total energy = {} eV'.format(energy), '']

  if True in atoms.pbc:
    xsf += ['CRYSTAL', 'PRIMVEC']
    for v in atoms.get_cell():
      xsf += ['{} {} {}'.format(*v)]
    xsf += ['PRIMCOORD', '{} 1'.format(len(atoms))]

  else:
    xsf += ['ATOMS']

  S = ('{atom.symbol:<3s} {atom.x: .12f} {atom.y: .12f} {atom.z: .12f} {f[0]: '
       '.12f} {f[1]: .12f} {f[2]: .12f}')
  xsf += [S.format(atom=atom, f=forces[i]) for i, atom in enumerate(atoms)]

  output = '\n'.join(xsf)
  with open(xsfile, 'w') as f:
    f.write(output)

  return output

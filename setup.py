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

from setuptools import setup, find_packages

with open('README.org') as f:
  readme = f.read()

with open('LICENSE') as f:
  license = f.read()

setup(
    name='dap',
    version='0.0.1',
    description='Differentiable atomistic potentials',
    long_description=readme,
    author='John Kitchin',
    author_email='kitchin@google.com',
    license=license,
    setup_requires=['nose>=1.0'],
    data_files=['requirements.txt', 'LICENSE'],
    packages=find_packages(exclude=('tests', 'docs')))

# python setup.py register to setup user
# to push to pypi - (shell-command "python setup.py sdist upload")

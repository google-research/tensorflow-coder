# Copyright 2021 The TF-Coder Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installs the Colab logging portion of TF-Coder."""

import os
import setuptools

from tf_coder_colab_logging import __version__


def get_required_packages():
  """Returns a list of required packages."""
  return [
      'absl-py >= 0.6.1',  # Oct 26, 2018
      'numpy >= 1.15.4',  # Nov 4, 2018
      'mock >= 3.0.5',  # May 7, 2019
      'tensorflow >= 2.2.0',  # May 7, 2020
  ]


def run_setup():
  """Installs the Colab logging portion of TF-Coder."""

  logging_readme_path = os.path.join('tf_coder_colab_logging',
                                     'ReadmeLogging.md')
  with open(logging_readme_path, 'r') as fh:
    long_description = fh.read()

  setuptools.setup(
      name='tensorflow-coder-colab-logging',
      version=__version__,
      author='Google LLC',
      author_email='no-reply@google.com',
      description="Logging utilities for TensorFlow Coder's Colab interface",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/google-research/tensorflow-coder',
      packages=setuptools.find_packages(exclude=['tf_coder', 'tf_coder.*']),
      install_requires=get_required_packages(),
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      keywords='tensorflow coder tf-coder',
      python_requires='>=3',
  )


if __name__ == '__main__':
  run_setup()

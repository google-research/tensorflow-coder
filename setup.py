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

"""Installs TF-Coder."""

import setuptools

from tf_coder import __version__


def get_required_packages():
  """Returns a list of required packages."""
  return [
      'absl-py >= 0.6.1',  # Oct 26, 2018
      'funcsigs >= 1.0.2',  # Apr 25, 2016
      'numpy >= 1.15.4',  # Nov 4, 2018
      'scikit-learn >= 0.22.2',  # Feb 28, 2020
      'six >= 1.12.0',  # Dec 9, 2018
      'tensorflow >= 2.2.0',  # May 7, 2020
  ]


def run_setup():
  """Installs TF-Coder."""

  with open('README.md', 'r') as fh:
    long_description = fh.read()

  setuptools.setup(
      name='tensorflow-coder',
      version=__version__,
      author='Google LLC',
      author_email='no-reply@google.com',
      description=('TensorFlow Coder (TF-Coder): '
                   'A Program Synthesis Tool for TensorFlow'),
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/google-research/tensorflow-coder',
      packages=setuptools.find_packages(
          exclude=['tf_coder_colab_logging', 'tf_coder_colab_logging.*']),
      package_data={
          'tf_coder.datasets.github': [
              'data/tf_apis_full.tfrecord-00000-of-00001',
          ],
          'tf_coder.models': [
              'trained_model/ckpt-1172.data-00000-of-00001',
              'trained_model/config.json',
              'trained_model/ckpt-1172.index',
          ],
      },
      include_package_data=True,
      install_requires=get_required_packages(),
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development',
          'Topic :: Software Development :: Code Generators',
      ],
      keywords=('tensorflow coder tf-coder program synthesis tensor '
                'manipulation programming by example'),
      python_requires='>=3',
  )


if __name__ == '__main__':
  run_setup()

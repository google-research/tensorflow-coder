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

# Lint as: python3
"""Seed inputs for synthetic I/O example generation."""

import enum
import random
from typing import List, Optional

from absl import logging
import numpy as np
import tensorflow as tf
from tf_coder import tf_coder_utils


def _ints_from_nonnegative(ints: List[int]) -> List[int]:
  return [-x for x in ints][:0:-1] + ints


def _np_array_to_float_list(np_array) -> List[float]:
  return [round(float(f), 5) for f in np_array]

# Each element is a list of bools to choose from. For example, when populating a
# boolean tensor, all of the elements should come from the same list of choices.
BOOL_CHOICES = [
    # Different ratios of True and False.
    [False, True],
    [False] * 4 + [True],
    [False] * 9 + [True],
    [True] * 4 + [False],
    [True] * 9 + [False],
]

# Each element is a list of nonnegative ints to choose from.
UINT_CHOICES = [
    # 0s and 1s with different ratios.
    [0, 1],
    [0] * 4 + [1],
    [0] + [1] * 4,

    # Nonnegative ints. (Positive ints can be obtained by "getting lucky" and
    # not choosing 0, which should happen often.)
    list(range(6)),
    list(range(10)),
    list(range(20)),
    list(range(50)),
    list(range(100)),
]

# Each element is a list of ints to choose from.
INT_CHOICES = UINT_CHOICES + [
    # 0s, 1s, -1s.
    [-1, 0, 1],

    # Negative and nonnegative ints.
    list(range(-5, 6)),
    list(range(-9, 10)),
    list(range(-19, 20)),
    list(range(-49, 50)),
    list(range(-99, 100)),
    list(range(-1000, 1001)),
]

# Each element is a list of floats to choose from.
FLOAT_CHOICES = [
    # 0s and 1s with different ratios.
    [0.0, 1.0],
    [0.0] * 4 + [1.0],
    [0.0] + [1.0] * 4,

    # Ints as floats.
    [float(i) for i in range(10)],
    [float(i) for i in range(50)],
    [float(i) for i in range(-9, 10)],
    [float(i) for i in range(-99, 100)],

    # Probabilities with higher chance of exactly 0.0 or 1.0.
    [0.0] * 2 + _np_array_to_float_list(np.arange(0, 1.1, 0.1)) + [1.0] * 2,
    [0.0] * 50 + _np_array_to_float_list(np.arange(0, 1.01, 0.01)) + [1.0] * 50,

    # Nonnegative floats.
    _np_array_to_float_list(np.arange(0, 5, 0.1)),
    _np_array_to_float_list(np.arange(0, 10, 0.1)),
    _np_array_to_float_list(np.arange(0, 50, 0.1)),

    # Negative and nonnegative floats.
    _np_array_to_float_list(np.arange(-4.9, 5, 0.1)),
    _np_array_to_float_list(np.arange(-19.9, 20, 0.1)),
    _np_array_to_float_list(np.arange(-99, 100, 0.1)),
]


def generate_random_tensor(python_type, shape: Optional[List[int]] = None):
  """Returns a random tensor, optionally with the given shape.

  First choose the rank and shape. Depending on the python_type, choose a list
  of value choices from the lists above, decide whether to sample with or
  without replacement from that list of choices, and then randomly construct a
  tensor of the chosen shape using randomly selected values.

  Args:
    python_type: One of `int`, `float`, or `bool`.
    shape: Optional list of ints describing the desired shape, or None to have
      the shape be randomly chosen.
  """
  if shape is None:
    rank = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])
    # Randomly fill the shape, but make sure there aren't too many elements.
    while True:
      shape = []
      for _ in range(rank):
        length_distribution = 1 / (1 + np.arange(1, 21))
        length_distribution /= np.sum(length_distribution)
        shape.append(np.random.choice(range(1, 21), p=length_distribution))
      if np.prod(shape) <= 100:
        # We're assuming that no one will create a larger input tensor.
        break

  if python_type == bool:
    dtype = tf.bool
    choices_list = BOOL_CHOICES
    sort = False
  elif python_type == float:
    dtype = (
        tf.float32 if random.random() < 0.9 else random.choice(
            tf_coder_utils.FLOAT_DTYPES))
    choices_list = FLOAT_CHOICES
    sort = random.random() < 0.1
  elif python_type == int:
    dtype = (
        tf.int32
        if random.random() < 0.9 else random.choice(tf_coder_utils.INT_DTYPES))
    choices_list = (
        UINT_CHOICES if dtype in tf_coder_utils.UINT_DTYPES else INT_CHOICES)
    sort = random.random() < 0.1
  else:
    logging.fatal('Unhandled python_type: %s', python_type)

  choices = random.choice(choices_list)
  num_elements = np.prod(shape)
  replace = (True if num_elements > len(choices) else
             np.random.choice([False, True], p=[0.3, 0.7]))
  elements = np.random.choice(choices, size=shape, replace=replace)
  tensor = tf.constant(elements, dtype=dtype)

  # Only nonscalar tensors of certain dtypes can be sorted.
  if sort and len(shape) and (python_type == float
                              or dtype in [tf.int32, tf.int64]):
    tensor = tf.sort(tensor)
  return tensor


@enum.unique
class ValueKind(enum.Enum):
  PRIMITIVE = 1
  SEQUENCE = 2
  TENSOR = 3
  SPARSE_TENSOR = 4


def generate_random_input(random_seed: Optional[int] = None):
  """Create a random input."""
  if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)

  python_type = np.random.choice([int, float, bool], p=[0.5, 0.4, 0.1])
  kind = np.random.choice([ValueKind.PRIMITIVE, ValueKind.SEQUENCE,
                           ValueKind.TENSOR, ValueKind.SPARSE_TENSOR],
                          p=[0.05, 0.05, 0.8, 0.1])

  if kind == ValueKind.PRIMITIVE:
    scalar_tensor = generate_random_tensor(python_type=python_type, shape=[])
    return python_type(scalar_tensor.numpy())

  elif kind == ValueKind.SEQUENCE:
    # Generate a random tensor and turn it into a sequence along the first
    # dimension.
    tensor = generate_random_tensor(python_type=python_type)
    # An empty TensorShape, when converted to a bool, is True.
    while not len(tensor.shape):  # pylint: disable=g-explicit-length-test
      tensor = generate_random_tensor(python_type=python_type)
    return list(tensor)

  elif kind == ValueKind.TENSOR:
    return generate_random_tensor(python_type=python_type)

  elif kind == ValueKind.SPARSE_TENSOR:
    # 3 and 4 are the most likely, and the probability decreases as the number
    # gets farther from 3.5.
    num_elements_distribution = 1 / (1 + np.abs(np.arange(1, 21) - 3.5))
    num_elements_distribution /= np.sum(num_elements_distribution)
    num_elements = np.random.choice(range(1, 21), p=num_elements_distribution)
    values = generate_random_tensor(python_type=python_type,
                                    shape=[num_elements])

    rank = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.3, 0.1])
    while True:
      dense_shape = []
      for _ in range(rank):
        dense_shape.append(random.choice(
            list(range(1, 4)) * 5 +
            list(range(1, 10)) +
            list(range(10, 100, 10)) +
            list(range(100, 1001, 100))))
      if np.prod(dense_shape) >= 2 * num_elements:
        break

    indices_set = set()
    while len(indices_set) < num_elements:
      coordinates = tuple(random.choice(range(dense_shape[dimension]))
                          for dimension in range(rank))
      indices_set.add(coordinates)
    indices = list(indices_set)

    # TODO(b/140891723): tf.sparse.reorder doesn't work when the SparseTensor
    # has values of dtype tf.uint32 or tf.uint64. This workaround instead casts
    # the values to a workable dtype.
    if values.dtype in (tf.uint32, tf.uint64):
      values = tf.cast(values, tf.int32)

    sparse_tensor = tf.SparseTensor(values=values, indices=indices,
                                    dense_shape=dense_shape)
    return tf.sparse.reorder(sparse_tensor)

  else:
    logging.fatal('Unhandled kind: %s', kind)

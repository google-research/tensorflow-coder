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
"""Tests for value_search_utils.py."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from tf_coder import tensor_limits as limits
from tf_coder.value_search import value
from tf_coder.value_search import value_search_utils


class ValueSearchUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('acceptable_size', (1, 2, 3), True),
      ('too_many_elements', (10, 20, 30), False),
      ('one_dimension_too_long', (1, 1, 8000), False),
      ('too_many_dimensions', (1, 1, 1, 2, 3), False))
  def test_check_tensor_size(self, tensor_shape, expected_result):
    # check_tensor_size() expects a Value object, but is also called from
    # Value.__init__. We first create a Value object with an ok tensor size, and
    # then replace its wrapped value with the new tensor, so that we can check
    # the result of check_tensor_size() without it causing the Value constructor
    # to throw an exception.
    tensor = tf.zeros(tensor_shape)
    tensor_value = value.InputValue(tf.constant([1]), 'test')
    tensor_value.value = tensor
    tensor_value.cached_info = {}
    self.assertEqual(value_search_utils.check_tensor_size(tensor_value),
                     expected_result)

  def test_check_tensor_size_scalar_tensor(self):
    self.assertTrue(value_search_utils.check_tensor_size(
        value.ConstantValue(tf.constant(1))))

  @parameterized.named_parameters(
      ('acceptable_size',
       tf.SparseTensor(indices=[[0, 0, 1], [1, 2, 3]],
                       values=[8, 9],
                       dense_shape=[100, 200, 1000]),
       True),
      ('too_many_elements',
       tf.SparseTensor(
           indices=[[i] for i in range(limits.MAX_TENSOR_ELEMENTS + 1)],
           values=[1] * (limits.MAX_TENSOR_ELEMENTS + 1),
           dense_shape=[10000]),
       False),
      ('too_many_dimensions',
       tf.SparseTensor(indices=[[0] * (limits.MAX_NUM_DIMENSIONS + 1)],
                       values=[1],
                       dense_shape=[100] * (limits.MAX_NUM_DIMENSIONS + 1)),
       False))
  def test_check_sparse_tensor_size(self, sparse_tensor, expected_result):
    # check_tensor_size() expects a Value object, but is also called from
    # Value.__init__. We first create a Value object with an ok tensor size, and
    # then replace its wrapped value with the new tensor, so that we can check
    # the result of check_tensor_size() without it causing the Value constructor
    # to throw an exception.
    input_value = value.InputValue(
        tf.sparse.from_dense(tf.constant([[0, 1], [0, 0]])), 'test')
    input_value.value = sparse_tensor
    input_value.cached_info = {}
    self.assertEqual(value_search_utils.check_sparse_tensor_size(input_value),
                     expected_result)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

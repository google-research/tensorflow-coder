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
"""Tests for function_operation.py."""

from absl import logging
from absl.testing import absltest
import tensorflow as tf
from tf_coder import filter_group
from tf_coder import tensor_limits as limits
from tf_coder import tf_functions
from tf_coder.value_search import function_operation
from tf_coder.value_search import value
from tf_coder.value_search import value_search_settings as settings_module


class FunctionOperationTest(absltest.TestCase):

  def setUp(self):
    super(FunctionOperationTest, self).setUp()
    self.settings = settings_module.default_settings()

  def test_metadata(self):
    operation = function_operation.FunctionOperation(
        tf_functions.FunctionInfo(name='tf.reduce_sum(input_tensor, axis)',
                                  filter_group=filter_group.FilterGroup.NONE,
                                  weight=2))
    docstring = operation.metadata.docstring
    self.assertIn('Computes the sum of elements across dimensions of a tensor.',
                  docstring)
    self.assertIn('tf.reduce_sum(input_tensor, axis)', docstring)
    self.assertIn('reduce sum', docstring)
    self.assertIn('input tensor', docstring)

  def test_apply_succeeds(self):
    operation = function_operation.FunctionOperation(
        tf_functions.FunctionInfo(name='tf.reduce_sum(input_tensor, axis)',
                                  filter_group=filter_group.FilterGroup.NONE,
                                  weight=2))
    input_tensor = value.ConstantValue(tf.constant([[1, 3, 4],
                                                    [50, 20, 80]]))
    axis_0 = value.ConstantValue(0)
    axis_1 = value.ConstantValue(1)
    self.assertEqual(operation.apply([input_tensor, axis_0], self.settings),
                     value.ConstantValue(tf.constant([51, 23, 84])))
    self.assertEqual(operation.apply([input_tensor, axis_1], self.settings),
                     value.ConstantValue(tf.constant([8, 150])))

  def test_apply_returns_none_if_exception(self):
    operation = function_operation.FunctionOperation(
        tf_functions.FunctionInfo(name='tf.reduce_sum(input_tensor, axis)',
                                  filter_group=filter_group.FilterGroup.NONE,
                                  weight=2))
    input_tensor = value.ConstantValue(tf.constant([[1, 3, 4],
                                                    [50, 20, 80]]))
    axis_2 = value.ConstantValue(2)
    self.assertIsNone(operation.apply([input_tensor, axis_2], self.settings))

  def test_apply_returns_none_if_bad_value(self):
    operation = function_operation.FunctionOperation(
        tf_functions.FunctionInfo(name='tf.one_hot(indices, depth)',
                                  filter_group=filter_group.FilterGroup.NONE,
                                  weight=2))
    indices = value.ConstantValue(
        tf.ones([limits.MAX_DIMENSION_LENGTH]))
    depth = value.ConstantValue(limits.MAX_TENSOR_ELEMENTS)
    self.assertIsNone(operation.apply([indices, depth], self.settings))

  def test_reconstruct_expression(self):
    operation = function_operation.FunctionOperation(
        tf_functions.FunctionInfo(name='tf.reduce_sum(input_tensor, axis)',
                                  filter_group=filter_group.FilterGroup.NONE,
                                  weight=2))
    arg_1 = value.InputValue([[1, 3], [50, 20]], 'my_input')
    arg_2 = value.ConstantValue('tf-coder')
    self.assertEqual(operation.reconstruct_expression([arg_1, arg_2]),
                     "tf.reduce_sum(my_input, axis='tf-coder')")

if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

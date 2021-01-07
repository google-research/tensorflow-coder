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
"""Tests for operation_filtering.py."""

from absl import logging
from absl.testing import absltest
import tensorflow as tf
from tf_coder import filter_group
from tf_coder import tf_functions
from tf_coder.value_search import operation_base
from tf_coder.value_search import operation_filtering
from tf_coder.value_search import value


class DummyOperation(operation_base.Operation):
  """A dummy subclass of Operation used for testing."""

  def __init__(self, num_args, function_info):
    """Creates a DummyOperation with the given arity and FunctionInfo."""
    metadata = operation_base.OperationMetadata(docstring='dummy operation')
    super(DummyOperation, self).__init__(num_args, weight=1, metadata=metadata)
    self.function_info = function_info

  def _compute_name(self):
    return self.function_info.name

  def apply(self, arg_values):
    return None

  def reconstruct_expression_from_strings(self, arg_strings):
    return 'dummy operation'


class OperationFilteringTest(absltest.TestCase):

  def test_get_type_filter(self):
    int_type_filter = operation_filtering.get_type_filter(int)
    self.assertTrue(int_type_filter(value.ConstantValue(1)))
    self.assertFalse(int_type_filter(value.ConstantValue(1.0)))

  def test_get_dtype_filter(self):
    int32_dtype_filter = operation_filtering.get_dtype_filter(tf.int32)
    self.assertTrue(int32_dtype_filter(value.ConstantValue(tf.constant(1))))
    self.assertFalse(int32_dtype_filter(value.ConstantValue(tf.constant(1.0))))

  def test_get_dtype_filter_raises_for_non_dtype(self):
    with self.assertRaises(TypeError):
      operation_filtering.get_dtype_filter(int)

  def test_add_filters_to_function_operation_handles_all_filter_groups(self):
    for group in filter_group.FilterGroup:
      if group == filter_group.FilterGroup.NONE:
        arity = 1
      else:
        arity = int(group.name[group.name.rfind('_') + 1:])

      dummy_operation = DummyOperation(arity,
                                       tf_functions.FunctionInfo(
                                           name='tf.dummy()',
                                           filter_group=group,
                                           weight=1))
      operation_filtering.add_filters_to_function_operation(dummy_operation)

      if group is filter_group.FilterGroup.NONE:
        self.assertIsNone(dummy_operation._value_filters_list)
        self.assertIsNone(dummy_operation._apply_filter)
      else:
        self.assertTrue(dummy_operation._value_filters_list is not None or
                        dummy_operation._apply_filter is not None)

  def test_add_filters_to_function_operation_raises_for_unknown_group(self):
    dummy_operation = DummyOperation(1,
                                     tf_functions.FunctionInfo(
                                         name='tf.dummy()',
                                         filter_group='UNKNOWN',
                                         weight=1))
    with self.assertRaises(ValueError):
      operation_filtering.add_filters_to_function_operation(dummy_operation)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

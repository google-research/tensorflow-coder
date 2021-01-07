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
"""Tests for value.py."""

import collections
import itertools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from tf_coder import filter_group
from tf_coder import tf_functions
from tf_coder.value_search import function_operation
from tf_coder.value_search import value
from tf_coder.value_search import value_search_settings as settings_module


class DummyValue(value.Value):
  """A dummy implementation of the value.Value ABC for testing purposes."""

  def reconstruct_expression(self):
    return ''


class ValueTest(parameterized.TestCase):

  def test_init_with_primitive(self):
    constant_value = value.ConstantValue(12)
    self.assertEqual(constant_value.type, int)
    self.assertTrue(constant_value.is_primitive)

    self.assertIsNone(constant_value.elem_type)
    self.assertFalse(constant_value.elem_type_is_tensor)
    self.assertIsNone(constant_value.dtype)
    self.assertIsNone(constant_value.shape)

  def test_init_with_dtype(self):
    constant_value = value.ConstantValue(tf.int64)
    self.assertTrue(constant_value.is_dtype)

  @parameterized.named_parameters(
      {
          'testcase_name': 'sequence_of_primitives',
          'sequence': [1.23, 4.56, 7.89],
          'elem_type': float,
          'sequence_dtype': tf.float32,
          'sequence_shape': [3],
      }, {
          'testcase_name': 'sequence_of_tensors',
          'sequence': [tf.constant(1), tf.constant([2.3, 4.5])],
          'elem_type': type(tf.constant(1)),
          'elem_type_is_tensor': True,
      }, {
          'testcase_name': 'sequence_of_sparse_tensors',
          'sequence': [tf.sparse.from_dense(tf.eye(5))],
          'elem_type': tf.SparseTensor,
          'elem_type_is_sparse_tensor': True,
      }, {
          'testcase_name': 'nested_sequence',
          'sequence': [[1, 2], [3, 4], [5, 6]],
          'elem_type': list,
          'sequence_dtype': tf.int32,
          'sequence_shape': [3, 2],
      },
  )
  def test_init_with_sequence(self,
                              sequence,
                              elem_type=None,
                              elem_type_is_tensor=False,
                              elem_type_is_sparse_tensor=False,
                              sequence_dtype=None,
                              sequence_shape=None):
    dummy_value = DummyValue(sequence)
    self.assertTrue(dummy_value.is_sequence)
    self.assertEqual(dummy_value.elem_type, elem_type)
    self.assertEqual(dummy_value.elem_type_is_tensor, elem_type_is_tensor)
    self.assertEqual(dummy_value.elem_type_is_sparse_tensor,
                     elem_type_is_sparse_tensor)
    self.assertEqual(dummy_value.sequence_dtype, sequence_dtype)
    self.assertEqual(dummy_value.sequence_shape, sequence_shape)

  def test_init_with_tensor(self):
    input_value = value.InputValue(tf.constant([[11, 22, 33], [4, 5, 6]]), 'in')
    self.assertTrue(input_value.is_tensor)
    self.assertFalse(input_value.is_sparse_tensor)
    self.assertEqual(input_value.dtype, tf.int32)
    self.assertEqual(input_value.shape, [2, 3])
    # The shape should be a native Python list containing native Python ints.
    self.assertEqual(type(input_value.shape), list)
    self.assertEqual(type(input_value.shape[0]), int)

  def test_init_with_sparse_tensor(self):
    input_value = value.InputValue(
        tf.SparseTensor(
            indices=[[12, 34]],
            values=tf.constant([5.6], dtype=tf.float16),
            dense_shape=[50, 60]),
        'in')
    self.assertTrue(input_value.is_sparse_tensor)
    self.assertFalse(input_value.is_tensor)
    self.assertEqual(input_value.dtype, tf.float16)
    self.assertEqual(input_value.shape, [50, 60])
    # The shape should be a native Python list containing native Python ints.
    self.assertEqual(type(input_value.shape), list)
    self.assertEqual(type(input_value.shape[0]), int)

  @parameterized.named_parameters(
      ('empty_sequence', [], 'Sequences must be nonempty'),
      ('sequence_of_different_types', [1, False],
       'Sequences must contain elements of the same type'),
      ('sequence_of_bad_type', [lambda x: x + 1],
       'Sequence must contain Tensors, SparseTensors, primitives, or other '
       'sequences'),
      ('nested_sequence_not_rectangular', [[1], [2, 3]],
       'Sequence is not tensor-like'),
      ('nested_sequence_with_tensors', [[tf.constant(1)], [tf.constant(2)]],
       'Sequence is not tensor-like'),
      ('nested_sequence_empty_tensor', [[], [], []],
       'Sequence represents an empty tensor'),
      ('tensor_empty', tf.zeros((1, 2, 3))[:0], 'Tensor is empty'),
      ('tensor_too_large', tf.zeros((1, 2, 3, 4, 5, 6)),
       'Tensor value is too large'),
      ('sparse_index_out_of_bounds',
       tf.SparseTensor(indices=[[12, 34]], values=[3], dense_shape=[5, 6]),
       'SparseTensor has out-of-bounds index'),
      ('sparse_too_large',
       tf.SparseTensor(indices=[[0] * 10],
                       values=[3],
                       dense_shape=[5] * 10),
       'SparseTensor value is too large'),
      ('sparse_empty',
       tf.sparse.from_dense(tf.constant([])),
       'SparseTensor is empty'))
  def test_init_raises(self, wrapped_value, regexp):
    with self.assertRaisesRegex(ValueError, regexp):
      DummyValue(wrapped_value)

  def test_repr(self):
    self.assertEqual(repr(DummyValue(tf.constant([1, 2]))), 'tf.int32:[1, 2]')

  def test_hash(self):
    value_1a = value.InputValue(tf.constant([1, 2]), 'a')
    value_1b = value.OutputValue(tf.constant([1, 2]))
    value_2 = value.InputValue(tf.constant([1, 3]), 'a')
    self.assertEqual(hash(value_1a), hash(value_1b))
    self.assertNotEqual(hash(value_1a), hash(value_2))

  def test_eq(self):
    value_1a = value.InputValue(tf.constant([1, 2]), 'a')
    value_1b = value.OutputValue(tf.constant([1, 2]))
    value_2 = value.InputValue(tf.constant([1, 3]), 'a')
    # We want to explicitly test __eq__.
    self.assertTrue(value_1a == value_1b)  # pylint: disable=g-generic-assert
    self.assertFalse(value_1a == value_2)  # pylint: disable=g-generic-assert

  def test_eq_comparing_to_non_value(self):
    input_value = value.InputValue(tf.constant([1, 2]), 'a')
    not_comparable = [1, 2]
    # We want to explicitly test __eq__.
    self.assertFalse(input_value == not_comparable)  # pylint: disable=g-generic-assert

  def test_ne(self):
    value_1a = value.InputValue(tf.constant([1, 2]), 'a')
    value_1b = value.OutputValue(tf.constant([1, 2]))
    value_2 = value.InputValue(tf.constant([1, 3]), 'a')
    # We want to explicitly test __ne__.
    self.assertFalse(value_1a != value_1b)  # pylint: disable=g-generic-assert
    self.assertTrue(value_1a != value_2)  # pylint: disable=g-generic-assert


class OperationValueTest(absltest.TestCase):

  def setUp(self):
    super(OperationValueTest, self).setUp()
    function_info = tf_functions.FunctionInfo(
        name='tf.reduce_max(input_tensor, axis)',
        filter_group=filter_group.FilterGroup.NONE,
        weight=1)
    self.operation = function_operation.FunctionOperation(function_info)
    self.settings = settings_module.default_settings()

  def test_copy(self):
    tensor_value = value.InputValue([[1, 3, 2], [-3, 0, 4]], 'my_input')
    axis_value = value.ConstantValue(1)
    operation_value = self.operation.apply([tensor_value, axis_value],
                                           self.settings)

    copy_value = operation_value.copy()
    self.assertIsNot(operation_value, copy_value)
    self.assertTrue(operation_value.reconstruct_expression(use_cache=False)  # pylint: disable=g-generic-assert
                    == copy_value.reconstruct_expression(use_cache=False)
                    == 'tf.reduce_max(my_input, axis=1)')

    copy_value.operation_applications[0].arg_values[0].name = 'new_name'
    self.assertEqual(operation_value.reconstruct_expression(use_cache=False),
                     'tf.reduce_max(my_input, axis=1)')
    self.assertEqual(copy_value.reconstruct_expression(use_cache=False),
                     'tf.reduce_max(new_name, axis=1)')

  def test_reconstruct_expression(self):
    tensor_value = value.InputValue([[1, 3, 2], [-3, 0, 4]], 'my_input')
    axis_value = value.ConstantValue(1)

    operation_value = value.OperationValue(tf.constant([3, 4]),
                                           self.operation,
                                           [tensor_value, axis_value])
    expected_expression = 'tf.reduce_max(my_input, axis=1)'
    self.assertEqual(operation_value.reconstruct_expression(),
                     expected_expression)
    self.assertEqual(operation_value.reconstruct_expression(),
                     expected_expression)  # Cached.

    operation_value_apply = self.operation.apply([tensor_value, axis_value],
                                                 self.settings)
    self.assertEqual(operation_value, operation_value_apply)
    self.assertEqual(operation_value_apply.reconstruct_expression(),
                     expected_expression)

  def test_merge_reconstructions(self):
    tensor_value = value.InputValue([[1, 3, 2], [-3, 0, 4]], 'my_input')
    axis_value = value.ConstantValue(1)
    operation_value = self.operation.apply([tensor_value, axis_value],
                                           self.settings)
    self.assertLen(operation_value.operation_applications, 1)

    tensor_value_2 = value.InputValue([[1, 4], [3, 0]], 'my_input_2')
    axis_value_2 = value.ConstantValue(0)
    operation_value_2 = self.operation.apply([tensor_value_2, axis_value_2],
                                             self.settings)
    self.assertEqual(operation_value, operation_value_2)

    operation_value.merge_reconstructions(operation_value_2)
    self.assertLen(operation_value.operation_applications, 2)

  def test_reconstruct_all_expressions_with_input_names(self):
    input_0 = value.InputValue(0, 'in_0')
    constant_1 = value.ConstantValue(1)
    constant_2 = value.ConstantValue(2)

    my_input = value.InputValue([[1, 3, 2], [-3, 0, 4]], 'my_input')
    final_value = self.operation.apply([my_input, constant_1], self.settings)

    add_operation = function_operation.FunctionOperation(
        tf_functions.FunctionInfo(name='tf.add(x, y)',
                                  filter_group=filter_group.FilterGroup.NONE,
                                  weight=1))

    input_1321 = value.InputValue([[1, 3], [2, 1]], 'in_1321')
    value_2432 = add_operation.apply([input_1321, constant_1], self.settings)

    input_0210 = value.InputValue([[0, 2], [1, 0]], 'in_0210')
    value_2432_duplicate = add_operation.apply([input_0210, constant_2],
                                               self.settings)
    self.assertEqual(value_2432, value_2432_duplicate)
    value_2432.merge_reconstructions(value_2432_duplicate)

    final_value_duplicate = self.operation.apply([value_2432, input_0],
                                                 self.settings)
    self.assertEqual(final_value, final_value_duplicate)
    final_value.merge_reconstructions(final_value_duplicate)

    input_1430 = value.InputValue([[1, 4], [3, 0]], 'in_1430')
    final_value_duplicate = self.operation.apply([input_1430, input_0],
                                                 self.settings)

    self.assertEqual(final_value, final_value_duplicate)
    final_value.merge_reconstructions(final_value_duplicate)

    expected = [
        ('tf.reduce_max(my_input, axis=1)', {'my_input'}),
        ('tf.reduce_max(tf.add(in_1321, 1), axis=in_0)', {'in_1321', 'in_0'}),
        ('tf.reduce_max(tf.add(in_0210, 2), axis=in_0)', {'in_0210', 'in_0'}),
        ('tf.reduce_max(in_1430, axis=in_0)', {'in_1430', 'in_0'}),
    ]

    self.assertEqual(final_value.reconstruct_all_expressions_with_input_names(),
                     expected)

  def test_reconstruct_all_expressions_with_input_names_using_addition(self):
    constants = [value.ConstantValue(i) for i in range(10)]
    add_operation = function_operation.FunctionOperation(
        tf_functions.FunctionInfo(name='tf.add(x, y)',
                                  filter_group=filter_group.FilterGroup.NONE,
                                  weight=1))

    # The i-th element contains all unique Value objects of weight i, mapped to
    # themselves to allow retrieving the stored Value equal to some query Value.
    values_by_weight = [collections.OrderedDict()]  # Nothing of weight 0.
    # Add constants with weight 1.
    values_by_weight.append(collections.OrderedDict())
    for constant in constants:
      values_by_weight[1][constant] = constant

    for weight in range(2, 6):
      new_values = collections.OrderedDict()
      for arg_1_weight in range(1, weight):
        arg_2_weight = weight - arg_1_weight - 1
        for arg1, arg2 in itertools.product(values_by_weight[arg_1_weight],
                                            values_by_weight[arg_2_weight]):
          result = add_operation.apply([arg1, arg2], self.settings)
          if result not in new_values:
            new_values[result] = result
          else:
            new_values[result].merge_reconstructions(result)
      values_by_weight.append(new_values)

    query = value.OutputValue(9)

    # The form must be (a + b), where there are 10 choices for a, which then
    # determines b.
    reconstructions = (values_by_weight[3][query]
                       .reconstruct_all_expressions_with_input_names())
    self.assertLen(reconstructions, 10)
    # No expression uses input values.
    self.assertTrue(all(not bool(used_names)
                        for _, used_names in reconstructions))

    # No AST with only binary operators has weight 4.
    self.assertEmpty(values_by_weight[4])

    # The form is either (a + (b + c)) or ((a + b) + c). Each of the two forms
    # has 1 + 2 + ... + 9 = 45 options. Note that "a" in (a + (b + c)) cannot be
    # 0, or else (b + c) would have the same value as the entire expression.
    # Similarly, "c" in ((a + b) + c) cannot be 0.
    self.assertLen(values_by_weight[5][query]
                   .reconstruct_all_expressions_with_input_names(), 90)


class ConstantValueTest(absltest.TestCase):

  def test_init_does_not_convert_to_tensor(self):
    constant_value = value.ConstantValue(1)
    self.assertFalse(constant_value.is_tensor)
    self.assertTrue(constant_value.is_primitive)

  def test_copy(self):
    constant_value = value.ConstantValue(1)
    copy_value = constant_value.copy()
    self.assertIsNot(constant_value, copy_value)
    self.assertEqual(constant_value.reconstruct_expression(),
                     copy_value.reconstruct_expression())

  def test_reconstruct_expression(self):
    constant_value = value.ConstantValue('TF-Coder')
    self.assertEqual(constant_value.reconstruct_expression(), "'TF-Coder'")


class InputValueTest(absltest.TestCase):

  def test_init_converts_list_to_tensor(self):
    input_value = value.InputValue([1], 'descriptive_name')
    self.assertTrue(input_value.is_tensor)
    self.assertFalse(input_value.is_primitive)

  def test_init_does_not_convert_primitive_to_tensor(self):
    input_value = value.InputValue(1, 'descriptive_name')
    self.assertFalse(input_value.is_tensor)
    self.assertTrue(input_value.is_primitive)

  def test_copy(self):
    input_value = value.InputValue([1, 2], 'a')
    copy_value = input_value.copy()
    self.assertIsNot(input_value, copy_value)
    self.assertEqual(input_value.reconstruct_expression(use_cache=False),
                     copy_value.reconstruct_expression(use_cache=False))
    copy_value.name = 'b'
    self.assertNotEqual(input_value.reconstruct_expression(use_cache=False),
                        copy_value.reconstruct_expression(use_cache=False))

  def test_reconstruct_expression(self):
    input_value = value.InputValue([1], 'descriptive_name')
    self.assertEqual(input_value.reconstruct_expression(), 'descriptive_name')


class OutputValueTest(absltest.TestCase):

  def test_init_converts_to_tensor(self):
    output_value = value.OutputValue(1)
    self.assertTrue(output_value.is_tensor)

  def test_copy(self):
    with self.assertRaises(NotImplementedError):
      value.OutputValue(1).copy()

  def test_reconstruct_expression(self):
    output_value = value.OutputValue(1)
    with self.assertRaises(NotImplementedError):
      output_value.reconstruct_expression()


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

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
"""Tests for python_operations.py."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from tf_coder.value_search import python_operations
from tf_coder.value_search import value
from tf_coder.value_search import value_search_settings as settings_module


class IndexingOperationTest(parameterized.TestCase):

  def setUp(self):
    super(IndexingOperationTest, self).setUp()
    self.operation = python_operations.IndexingOperation()

  @parameterized.named_parameters(
      ('index_0', 0, 1.2),
      ('index_2', 2, 5.6),
      ('index_negative_2', -2, 3.4))
  def test_apply_for_list(self, index, expected):
    arg_values = [value.ConstantValue([1.2, 3.4, 5.6]),
                  value.ConstantValue(index)]
    result_value = self.operation.apply(arg_values,
                                        settings_module.default_settings())
    self.assertEqual(result_value, value.ConstantValue(expected))

  @parameterized.named_parameters(
      ('index_0', 0, [1.2, 3.4]),
      ('index_negative_1', -1, [5.6, 7.8]))
  def test_apply_for_tensor(self, index, expected):
    tensor = tf.constant([[1.2, 3.4], [5.6, 7.8]])
    arg_values = [value.ConstantValue(tensor),
                  value.ConstantValue(index)]
    result_value = self.operation.apply(arg_values,
                                        settings_module.default_settings())
    self.assertEqual(result_value, value.ConstantValue(tf.constant(expected)))

  def test_reconstruct_expression(self):
    to_index = value.InputValue([1.2, 3.4], 'my_list')
    index = value.ConstantValue(0)
    self.assertEqual(self.operation.reconstruct_expression([to_index, index]),
                     'my_list[0]')


class IndexingAxis1OperationTest(absltest.TestCase):

  def setUp(self):
    super(IndexingAxis1OperationTest, self).setUp()
    self.operation = python_operations.IndexingAxis1Operation()
    self.arg_values = [value.InputValue([[12, 34], [56, 78]], 'my_input'),
                       value.ConstantValue(1)]

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.OutputValue([34, 78]))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     'my_input[:, 1]')


class SingletonTupleCreationOperationTest(absltest.TestCase):

  def setUp(self):
    super(SingletonTupleCreationOperationTest, self).setUp()
    self.operation = python_operations.SingletonTupleCreationOperation()
    self.arg_values = [value.ConstantValue(12)]

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.ConstantValue((12,)))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     '(12,)')


class PairCreationOperationTest(absltest.TestCase):

  def setUp(self):
    super(PairCreationOperationTest, self).setUp()
    self.operation = python_operations.PairCreationOperation()
    self.arg_values = [value.ConstantValue(12), value.ConstantValue(34)]

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.ConstantValue((12, 34)))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     '(12, 34)')


class TripleCreationOperationTest(absltest.TestCase):

  def setUp(self):
    super(TripleCreationOperationTest, self).setUp()
    self.operation = python_operations.TripleCreationOperation()
    self.arg_values = [value.ConstantValue(12),
                       value.ConstantValue(34),
                       value.ConstantValue(56)]

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.ConstantValue((12, 34, 56)))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     '(12, 34, 56)')


class SlicingAxis0LeftOperationTest(absltest.TestCase):

  def setUp(self):
    super(SlicingAxis0LeftOperationTest, self).setUp()
    self.operation = python_operations.SlicingAxis0LeftOperation()
    self.arg_values = [value.InputValue([12, 34, 56], 'my_input'),
                       value.ConstantValue(1)]

  def test_get_docstring(self):
    self.assertIn('Selects a suffix of indices along axis 0 of the tensor',
                  self.operation.metadata.docstring)

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.OutputValue([34, 56]))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     'my_input[1:]')


class SlicingAxis0RightOperationTest(absltest.TestCase):

  def setUp(self):
    super(SlicingAxis0RightOperationTest, self).setUp()
    self.operation = python_operations.SlicingAxis0RightOperation()
    self.arg_values = [value.InputValue([12, 34, 56], 'my_input'),
                       value.ConstantValue(-1)]

  def test_get_docstring(self):
    self.assertIn('Selects a prefix of indices along axis 0 of the tensor',
                  self.operation.metadata.docstring)

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.OutputValue([12, 34]))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     'my_input[:-1]')


class SlicingAxis0BothOperationTest(absltest.TestCase):

  def setUp(self):
    super(SlicingAxis0BothOperationTest, self).setUp()
    self.operation = python_operations.SlicingAxis0BothOperation()
    self.arg_values = [value.InputValue([12, 34, 56, 78], 'my_input'),
                       value.ConstantValue(1),
                       value.ConstantValue(-1)]

  def test_get_docstring(self):
    self.assertIn('Selects a range of indices along axis 0 of the tensor',
                  self.operation.metadata.docstring)

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.OutputValue([34, 56]))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     'my_input[1:-1]')


class SlicingAxis1LeftOperationTest(absltest.TestCase):

  def setUp(self):
    super(SlicingAxis1LeftOperationTest, self).setUp()
    self.operation = python_operations.SlicingAxis1LeftOperation()
    self.arg_values = [value.InputValue([[12, 34, 56]], 'my_input'),
                       value.ConstantValue(1)]

  def test_get_docstring(self):
    self.assertIn('Selects a suffix of indices along axis 1 of the tensor',
                  self.operation.metadata.docstring)

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.OutputValue([[34, 56]]))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     'my_input[:, 1:]')


class SlicingAxis1RightOperationTest(absltest.TestCase):

  def setUp(self):
    super(SlicingAxis1RightOperationTest, self).setUp()
    self.operation = python_operations.SlicingAxis1RightOperation()
    self.arg_values = [value.InputValue([[12, 34, 56]], 'my_input'),
                       value.ConstantValue(-1)]

  def test_get_docstring(self):
    self.assertIn('Selects a prefix of indices along axis 1 of the tensor',
                  self.operation.metadata.docstring)

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.OutputValue([[12, 34]]))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     'my_input[:, :-1]')


class SlicingAxis1BothOperationTest(absltest.TestCase):

  def setUp(self):
    super(SlicingAxis1BothOperationTest, self).setUp()
    self.operation = python_operations.SlicingAxis1BothOperation()
    self.arg_values = [value.InputValue([[12, 34, 56, 78],
                                         [-1, -2, -3, -4]], 'my_input'),
                       value.ConstantValue(1),
                       value.ConstantValue(-1)]

  def test_get_docstring(self):
    self.assertIn('Selects a range of indices along axis 1 of the tensor',
                  self.operation.metadata.docstring)

  def test_apply(self):
    self.assertEqual(self.operation.apply(self.arg_values,
                                          settings_module.default_settings()),
                     value.OutputValue([[34, 56], [-2, -3]]))

  def test_reconstruct_expression(self):
    self.assertEqual(self.operation.reconstruct_expression(self.arg_values),
                     'my_input[:, 1:-1]')


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

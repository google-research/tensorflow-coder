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
"""Tests for all_operations.py."""

from absl.testing import absltest
from absl.testing import parameterized
from tf_coder import tf_functions
from tf_coder.value_search import all_operations
from tf_coder.value_search import operation_base


class AllOperationsTest(parameterized.TestCase):

  def test_get_python_operations(self):
    operations = all_operations.get_python_operations()
    self.assertTrue(all(isinstance(operation, operation_base.Operation)
                        for operation in operations))
    self.assertTrue(any(operation.name == 'IndexingOperation'
                        for operation in operations))

  def test_get_tf_operations(self):
    operations = all_operations.get_tf_operations()
    self.assertLen(operations, len(tf_functions.TF_FUNCTIONS))
    self.assertTrue(any(operation.name == 'tf.add(x, y)'
                        for operation in operations))
    self.assertFalse(any(operation.name == 'tf.sparse.add(a, b)'
                         for operation in operations))

  def test_get_sparse_operations(self):
    operations = all_operations.get_sparse_operations()
    self.assertLen(operations, len(tf_functions.SPARSE_FUNCTIONS))
    self.assertFalse(any(operation.name == 'tf.add(x, y)'
                         for operation in operations))
    self.assertTrue(any(operation.name == 'tf.sparse.add(a, b)'
                        for operation in operations))

  def test_get_operations_correct_type(self):
    operations = all_operations.get_operations(include_sparse_operations=True)
    self.assertTrue(all(isinstance(element, operation_base.Operation)
                        for element in operations))

  @parameterized.named_parameters(
      ('with_sparse', True,
       len(tf_functions.TF_FUNCTIONS) + len(tf_functions.SPARSE_FUNCTIONS) +
       len(all_operations.get_python_operations())),
      ('without_sparse', False,
       len(tf_functions.TF_FUNCTIONS) +
       len(all_operations.get_python_operations())))
  def test_get_operations_correct_cardinality(
      self, include_sparse_operations, expected_cardinality):
    operations = all_operations.get_operations(
        include_sparse_operations=include_sparse_operations)
    self.assertLen(operations, expected_cardinality)

  @parameterized.named_parameters(
      ('indexing', 'IndexingOperation', False),
      ('slicing_axis_0_both', 'SlicingAxis0BothOperation', False),
      ('tf_add', 'tf.add(x, y)', False),
      ('tf_cast', 'tf.cast(x, dtype)', False),
      ('tf_sparse_expand_dims', 'tf.sparse.expand_dims(sp_input, axis)', True))
  def test_get_operations_includes_expected(self, name, is_sparse):
    for include_sparse_operations in [True, False]:
      operations = all_operations.get_operations(
          include_sparse_operations=include_sparse_operations)
      should_be_included = include_sparse_operations or not is_sparse
      self.assertEqual(any(operation.name == name for operation in operations),
                       should_be_included)

  def test_get_operations_unique_names(self):
    operations = all_operations.get_operations(
        include_sparse_operations=True)
    names_set = {operation.name for operation in operations}
    self.assertLen(names_set, len(operations))

  def test_get_operations_all_have_docstrings(self):
    operations = all_operations.get_operations(include_sparse_operations=True)
    self.assertTrue(all(operation.metadata.docstring
                        for operation in operations))

  def test_get_operations_consistent_order(self):
    operations_1 = all_operations.get_operations(include_sparse_operations=True)
    operations_2 = all_operations.get_operations(include_sparse_operations=True)
    self.assertEqual([op.name for op in operations_1],
                     [op.name for op in operations_2])

  def test_find_operation_with_name(self):
    operation = all_operations.find_operation_with_name('tf.add(x, y)')
    self.assertEqual(operation.name, 'tf.add(x, y)')

    operation = all_operations.find_operation_with_name(
        'tf.add(x, y)', operation_list=all_operations.get_tf_operations())
    self.assertEqual(operation.name, 'tf.add(x, y)')

  def test_find_operation_with_name_raises_if_not_found(self):
    with self.assertRaises(ValueError):
      all_operations.find_operation_with_name('bad name')

    with self.assertRaises(ValueError):
      all_operations.find_operation_with_name(
          'tf.add(x, y)', operation_list=all_operations.get_python_operations())


if __name__ == '__main__':
  absltest.main()

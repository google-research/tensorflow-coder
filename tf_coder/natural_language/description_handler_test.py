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
"""Tests for description_handler.py."""

from absl.testing import absltest
from absl.testing import parameterized
from tf_coder.natural_language import description_handler
from tf_coder.value_search import python_operations
from tf_coder.value_search import value_search_settings as settings_module


class NoChangeDescriptionHandlerTest(absltest.TestCase):

  def setUp(self):
    super(NoChangeDescriptionHandlerTest, self).setUp()
    self.indexing_operation = python_operations.IndexingOperation()
    self.settings = settings_module.default_settings()

  def test_init_succeeds(self):
    description_handler.NoChangeDescriptionHandler()

  def test_init_raises_if_duplicate_name(self):
    with self.assertRaises(ValueError):
      description_handler.NoChangeDescriptionHandler(
          operations=[self.indexing_operation, self.indexing_operation])

  def test_attributes_providing_operations(self):
    handler = description_handler.NoChangeDescriptionHandler(
        operations=[self.indexing_operation])
    self.assertLen(handler.operations, 1)
    self.assertEqual(handler.all_names, [self.indexing_operation.name])

  def test_attributes_default_operations(self):
    handler = description_handler.NoChangeDescriptionHandler()
    self.assertGreater(len(handler.operations), 1)
    self.assertLen(handler.operations, len(handler.all_names))

  def test_get_operation_multipliers(self):
    handler = description_handler.NoChangeDescriptionHandler()
    multipliers = handler.get_operation_multipliers('dummy description',
                                                    self.settings)
    self.assertEmpty(multipliers)

  def test_repr(self):
    handler = description_handler.NoChangeDescriptionHandler()
    self.assertEqual(repr(handler), 'NoChangeDescriptionHandler')


class FunctionNameDescriptionHandlerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('exact_match', 'Use reduce_max', 'tf.reduce_max(input_tensor)', 0.75),
      ('parts_exist', 'Reduce to the max element',
       'tf.reduce_max(input_tensor)', 0.75),
      ('handle_dots', 'Add a sparse tensor', 'tf.sparse.add(a, b)', 0.75),
      ('partial_match', 'Reduce to the min element',
       'tf.reduce_max(input_tensor)', 1.0),
      ('unrelated', 'Reduce to the min element', 'tf.square(x)', 1.0),
      ('not_tf', 'Reduce to the max element', 'IndexingOperation', 1.0))
  def test_get_operation_multipliers(self, description, operation_name,
                                     expected_multiplier):
    handler = description_handler.FunctionNameDescriptionHandler()
    multipliers = handler.get_operation_multipliers(
        description, settings_module.default_settings())
    self.assertEqual(multipliers.get(operation_name, 1.0), expected_multiplier)
    # Nothing is deprioritized.
    self.assertTrue(all(0 < value < 1 for value in multipliers.values()))


if __name__ == '__main__':
  absltest.main()

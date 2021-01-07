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
"""Tests for tf_coder_utils.py."""

import collections

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from tf_coder import tf_coder_utils


class TfCoderUtilsTest(parameterized.TestCase):

  def test_int_dtype_min_max_has_correct_keys(self):
    self.assertEqual(
        set(tf_coder_utils.INT_DTYPES), set(tf_coder_utils.INT_DTYPE_MIN_MAX))

  @parameterized.named_parameters(
      ('correct_usage_add', 'tf.add', tf.add),
      ('correct_usage_matmul', 'tf.matmul', tf.matmul),
      ('nested_modules', 'tf.nn.softmax', tf.nn.softmax))
  def test_get_tf_function(self, function_name, expected_result):
    self.assertEqual(
        tf_coder_utils.get_tf_function(function_name), expected_result)

  @parameterized.named_parameters(
      ('not_tf', 'np.add'),
      ('no_module', 'add'),
      ('bad_nested_module', 'tf.bad.softmax'),
      ('bad_nested_module_function', 'tf.nn.bad'),
      ('not_function', 'tf.this_is_not_a_function'))
  def test_get_tf_function_raises(self, function_name):
    with self.assertRaises(ValueError):
      tf_coder_utils.get_tf_function(function_name)

  def test_convert_to_tensor_for_tensor(self):
    original_tensor = tf.constant([1, 2, 3])
    tensor = tf_coder_utils.convert_to_tensor(original_tensor)
    self.assertIsInstance(tensor, tf.Tensor)
    self.assertEqual(original_tensor.numpy().tolist(), tensor.numpy().tolist())

  def test_convert_to_tensor_for_list(self):
    list_2d = [[1, 2, 3], [44, 55, 66]]
    tensor = tf_coder_utils.convert_to_tensor(list_2d)
    self.assertIsInstance(tensor, tf.Tensor)
    self.assertEqual(list_2d, tensor.numpy().tolist())

  def test_convert_to_tensor_for_scalar(self):
    scalar = 1.23
    tensor = tf_coder_utils.convert_to_tensor(scalar)
    self.assertIsInstance(tensor, tf.Tensor)
    self.assertAlmostEqual(scalar, tensor.numpy().tolist())

  @parameterized.named_parameters(
      ('1_dimension', (3), 3),
      ('2_dimensions', (3, 4), 12),
      ('3_dimensions', (3, 4, 5), 60))
  def test_num_tensor_elements_using_shape(self, shape, expected_result):
    self.assertEqual(
        tf_coder_utils.num_tensor_elements(tf.ones(shape)), expected_result)

  @parameterized.named_parameters(
      ('empty_list', [], 0),
      ('filled_list', [11, 12, 13], 3),
      ('scalar', 123, 1))
  def test_num_tensor_elements_using_content(self, content, expected_result):
    self.assertEqual(
        tf_coder_utils.num_tensor_elements(tf.constant(content)),
        expected_result)

  @parameterized.named_parameters(
      ('empty', [], float('-inf')),
      ('single_element_list', [12.5], 12.5),
      ('2_dimensions', [[-100, -50], [-123, -45]], -45.0),
      ('scalar', -12.5, -12.5))
  def test_max_tensor_value(self, content, expected_result):
    self.assertAlmostEqual(
        tf_coder_utils.max_tensor_value(tf.constant(content)), expected_result)

  @parameterized.named_parameters(
      ('empty', [], float('inf')),
      ('single_element_list', [12.5], 12.5),
      ('2_dimensions', [[-100, -50], [-123, -45]], -123.0),
      ('scalar', -12.5, -12.5))
  def test_min_tensor_value(self, content, expected_result):
    self.assertAlmostEqual(
        tf_coder_utils.min_tensor_value(tf.constant(content)), expected_result)

  def test_tensor_to_string(self):
    tensor = tf.constant([[1, 2], [3, 4]])
    self.assertEqual(
        tf_coder_utils.tensor_to_string(tensor), 'tf.int32:[[1, 2], [3, 4]]')

  def test_tensor_to_string_ugly_floats(self):
    # We do not know how to print floating-point numpy arrays in a prettier way.
    tensor = tf.constant([0.4, 0.7])
    self.assertEqual(
        tf_coder_utils.tensor_to_string(tensor),
        'tf.float32:[0.4000000059604645, 0.699999988079071]')

  @parameterized.named_parameters(
      ('zero_places', 0.9, 0, 'tf.float32:[1.0]'),
      ('round_down', 1.549, 1, 'tf.float32:[1.5]'),
      ('round_up', 1.249, 2, 'tf.float32:[1.25]'),
      ('not_rounded', 1.03125, 6, 'tf.float32:[1.03125]'))
  def test_tensor_to_string_performs_rounding(self, float_value, decimals,
                                              expected_result):
    tensor = tf.constant([float_value])
    self.assertEqual(
        tf_coder_utils.tensor_to_string(tensor, decimals=decimals),
        expected_result)

  def test_object_to_string_tensor(self):
    tensor = tf.constant([[1, 2], [3, 4]])
    self.assertEqual(
        tf_coder_utils.object_to_string(tensor), 'tf.int32:[[1, 2], [3, 4]]')

  @parameterized.named_parameters(
      ('int', 123, '123'),
      ('float', 1.23, '1.23'),
      ('bool', False, 'False'),
      ('string', 'abc', "'abc'"))
  def test_object_to_string_primitive(self, primitive, expected_result):
    self.assertEqual(
        tf_coder_utils.object_to_string(primitive), expected_result)

  def test_object_to_string_dtype(self):
    self.assertEqual(tf_coder_utils.object_to_string(tf.int32), 'tf.int32')

  def test_object_to_string_sequence(self):
    # `Named` is a class and should be capitalized. It is only used in this test
    # so it is declared here, not in the global scope.
    Named = collections.namedtuple('Named', ('a', 'b'))  # pylint: disable=invalid-name
    sequence = [123, tf.constant([1, 2]), (), Named(a=False, b=1.5)]
    self.assertEqual(
        tf_coder_utils.object_to_string(sequence),
        'seq[123, tf.int32:[1, 2], seq[], seq[False, 1.5]]')

  def test_object_to_string_raises_if_unsupported(self):
    with self.assertRaises(ValueError):
      tf_coder_utils.object_to_string({'key': 'value'})

  @parameterized.named_parameters(
      ('0_elements_1_part', 0, 1, [[0]]),
      ('0_elements_2_parts', 0, 2, [[0, 0]]),
      ('1_element_1_part', 1, 1, [[1]]),
      ('1_element_2_parts', 1, 2, [[0, 1], [1, 0]]),
      ('3_element_3_parts', 3, 3, [
          [0, 0, 3], [0, 1, 2], [0, 2, 1], [0, 3, 0],
          [1, 0, 2], [1, 1, 1], [1, 2, 0],
          [2, 0, 1], [2, 1, 0],
          [3, 0, 0],
      ]))
  def test_generate_partitions(self, num_elements, num_parts, expected_result):
    actual = list(tf_coder_utils.generate_partitions(num_elements, num_parts))
    self.assertCountEqual(actual, expected_result)

  @parameterized.named_parameters(
      ('negative_elements', -1, 1),
      ('negative_parts', 1, -1),
      ('zero_parts', 1, 0))
  def test_generate_partitions_raises_on_invalid_input(self, num_elements,
                                                       num_parts):
    with self.assertRaises(ValueError):
      list(tf_coder_utils.generate_partitions(num_elements, num_parts))


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

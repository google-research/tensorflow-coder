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
"""Tests for collect_tensor_data."""

from absl.testing import absltest
from absl.testing import parameterized
import mock
import six
import tensorflow as tf
from tf_coder.datasets import collect_tensor_data
from tf_coder.value_search import all_operations
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search_settings as settings_module


def new_input(raw_value):
  return value_module.InputValue(raw_value, 'NEW_INPUT')


class CollectTensorDataTest(parameterized.TestCase):

  def setUp(self):
    super(CollectTensorDataTest, self).setUp()
    self.settings = settings_module.default_settings()

    operations = all_operations.get_operations()
    self.unique_with_counts_operation = all_operations.find_operation_with_name(
        'tf.unique_with_counts(x)', operation_list=operations)
    self.indexing_operation = all_operations.find_operation_with_name(
        'IndexingOperation', operation_list=operations)
    self.gather_operation = all_operations.find_operation_with_name(
        'tf.gather(params, indices)', operation_list=operations)
    self.add_operation = all_operations.find_operation_with_name(
        'tf.add(x, y)', operation_list=operations)

    # Example with many operations.
    in1 = value_module.InputValue([1, 1, 2, 5, 6, 5], 'in1')
    in2 = value_module.InputValue([0, 10, 20, 30, 40, 50, 60, 70], 'in2')
    constant_1 = value_module.ConstantValue(1)

    unique = self.unique_with_counts_operation.apply([in1], self.settings)
    indexed = self.indexing_operation.apply([unique, constant_1], self.settings)
    gathered = self.gather_operation.apply([in2, in1], self.settings)
    self.example_value_1 = self.add_operation.apply([indexed, gathered],
                                                    self.settings)

    self.assertEqual(
        self.example_value_1.reconstruct_expression(),
        'tf.add(tf.unique_with_counts(in1)[1], tf.gather(in2, in1))')
    self.assertEqual(self.example_value_1,
                     value_module.OutputValue([10, 10, 21, 52, 63, 52]))

    # Example with many variables and new inputs.
    in3 = value_module.InputValue([1], 'in3')
    in4 = value_module.InputValue([2], 'in4')

    a = self.add_operation.apply([in3, new_input([10])], self.settings)
    b = self.add_operation.apply([in4, in3], self.settings)
    c = self.add_operation.apply([new_input([20]), in3], self.settings)
    d = self.add_operation.apply([a, b], self.settings)
    self.example_value_2 = self.add_operation.apply([c, d], self.settings)

    self.assertEqual(
        self.example_value_2.reconstruct_expression(),
        'tf.add(tf.add(NEW_INPUT, in3), '
        'tf.add(tf.add(in3, NEW_INPUT), tf.add(in4, in3)))')
    self.assertEqual(self.example_value_2, value_module.OutputValue([35]))

  def test_extract_values_with_collapsed_subtrees(self):
    expected_expressions = [
        'tf.add(tf.unique_with_counts(in1)[1], tf.gather(in2, in1))',
        'tf.add(tf.unique_with_counts(in1)[1], NEW_INPUT)',
        'tf.add(NEW_INPUT[1], tf.gather(in2, in1))',
        'tf.add(NEW_INPUT[1], NEW_INPUT)',
        'tf.add(NEW_INPUT, tf.gather(in2, in1))',
        'tf.add(NEW_INPUT, NEW_INPUT)',
        'NEW_INPUT',
    ]
    results = collect_tensor_data.extract_values_with_collapsed_subtrees(
        self.example_value_1)
    # The order doesn't matter except that 'NEW_INPUT' is last.
    self.assertEqual([v.reconstruct_expression() for v in results],
                     expected_expressions)

  def test_count_num_inputs(self):
    self.assertEqual(collect_tensor_data.count_num_inputs(self.example_value_2),
                     (4, {'in3': 3, 'in4': 1, 'NEW_INPUT': 2}))

  def test_normalize_names_and_extract_values(self):
    name_counter = {'in3': 3, 'in4': 1, 'NEW_INPUT': 2}
    copied_value = self.example_value_2.copy()
    name_value_map = collect_tensor_data.normalize_names_and_extract_values(
        copied_value, name_counter)
    self.assertEqual(
        copied_value.reconstruct_expression(use_cache=False),
        # in3 -> in1, in4 -> in2, and NEW_INPUT -> in3 and in4
        'tf.add(tf.add(in3, in1), tf.add(tf.add(in1, in4), tf.add(in2, in1)))')

    self.assertEqual(name_value_map,
                     # Check the raw values.
                     {'in1': value_module.OutputValue([1]),
                      'in2': value_module.OutputValue([2]),
                      'in3': value_module.OutputValue([20]),
                      'in4': value_module.OutputValue([10])})
    for name in name_value_map:
      self.assertIsInstance(name_value_map[name], value_module.InputValue)
      self.assertEqual(name, name_value_map[name].name)

  def test_extract_operations(self):
    self.assertCountEqual(
        collect_tensor_data.extract_operations(self.example_value_1),
        [self.unique_with_counts_operation,
         self.indexing_operation,
         self.gather_operation,
         self.add_operation])
    self.assertCountEqual(
        collect_tensor_data.extract_operations(self.example_value_2),
        [self.add_operation] * 5)

  @parameterized.named_parameters(
      ('1', 1, []),
      ('2', 2, ['tf.add(tf.unique_with_counts(in1)[1], tf.gather(in2, in1))',
                'tf.add(tf.unique_with_counts(in1)[1], in2)',
                'tf.add(in1[1], in2)',
                'tf.add(in1, in2)']),
      ('3', 3, ['tf.add(tf.unique_with_counts(in1)[1], tf.gather(in2, in1))',
                'tf.add(tf.unique_with_counts(in1)[1], in2)',
                'tf.add(in3[1], tf.gather(in2, in1))',
                'tf.add(in1[1], in2)',
                'tf.add(in3, tf.gather(in2, in1))',
                'tf.add(in1, in2)']))
  def test_extract_examples_from_value(self, max_num_inputs,
                                       expected_expressions):
    actual = collect_tensor_data.extract_examples_from_value(
        self.example_value_1, max_num_inputs=max_num_inputs)

    # Check that all expressions are as expected.
    self.assertCountEqual([example.expression for example in actual],
                          expected_expressions)

    # Check all elements of one IOExample namedtuple. This example is at index 1
    # when max_num_inputs > 1.
    if max_num_inputs > 1:
      expected_index_1 = collect_tensor_data.IOExample(
          expression='tf.add(tf.unique_with_counts(in1)[1], in2)',
          input_values=[value_module.InputValue([1, 1, 2, 5, 6, 5], 'in1'),
                        value_module.InputValue([10, 10, 20, 50, 60, 50],
                                                'in2')],
          output_value=value_module.OutputValue([10, 10, 21, 52, 63, 52]),
          num_inputs=2,
          operations=[self.add_operation,
                      self.indexing_operation,
                      self.unique_with_counts_operation])
      self.assertEqual(actual[1], expected_index_1)
      # Equality of Value objects is done by comparing the wrapped values. Check
      # the names in input_values too.
      for actual_value, expected_value in zip(actual[1].input_values,
                                              expected_index_1.input_values):
        self.assertIsInstance(actual_value, value_module.InputValue)
        self.assertEqual(actual_value.name, expected_value.name)

    # Check that all extracted examples actually work by eval-ing them.
    for example in actual:
      namespace_dict = {'tf': tf}
      self.assertLen(example.input_values, example.num_inputs)
      for input_value in example.input_values:
        namespace_dict[input_value.name] = input_value.value
      eval_output = eval(example.expression, namespace_dict)  # pylint: disable=eval-used
      self.assertEqual(value_module.OutputValue(eval_output),
                       example.output_value)
      self.assertEqual(example.output_value, self.example_value_1)

  def test_extract_examples_from_value_without_inputs(self):
    constant_1 = value_module.ConstantValue(1)
    constant_2 = value_module.ConstantValue(2)
    constant_3 = value_module.ConstantValue(3)

    subtree = self.add_operation.apply([constant_1, constant_2], self.settings)
    without_inputs = self.add_operation.apply([subtree, constant_3],
                                              self.settings)
    actual = collect_tensor_data.extract_examples_from_value(without_inputs)

    self.assertCountEqual(
        [example.expression for example in actual],
        # `tf.add(tf.add(1, 2), 3)` has no inputs and is not included.
        ['tf.add(in1, 3)'])

  def test_create_tf_examples(self):
    sparse_tensor = tf.SparseTensor(
        values=[0, -15, 30],
        indices=[[12], [34], [56]],
        dense_shape=[100])
    # This example does not represent a realistic tensor transformation. It uses
    # variety in the input/output tensors to exercise the featurization.
    io_example = collect_tensor_data.IOExample(
        expression='tf.dummy_expression(in1, in2)',
        input_values=[
            value_module.InputValue([[[0.5, 2.5, 9.0],
                                      [-0.25, 0.0, 1.25]]], 'in1'),
            value_module.InputValue(sparse_tensor, 'in2'),
        ],
        output_value=value_module.OutputValue([[1.0], [0.0], [1.0], [0.0]]),
        num_inputs=2,
        operations=[self.add_operation, self.add_operation,
                    self.gather_operation])

    with mock.patch.object(collect_tensor_data, 'COUNT_BOUNDARIES',
                           new=[0, 1, 3, 50, float('inf')]):
      with mock.patch.object(
          collect_tensor_data, 'FLOAT_BOUNDARIES',
          new=[-float('inf'), -10, -1e-8, 1e-8, 10, float('inf')]):
        tf_examples = collect_tensor_data.create_tf_examples(io_example)

    operation_list = all_operations.get_operations(
        include_sparse_operations=True)
    expected_operations = [
        2 if op.name == 'tf.add(x, y)'
        else 1 if op.name == 'tf.gather(params, indices)' else 0
        for op in operation_list]

    expected_tf_example_1 = {
        # Features from featurize_value.
        'kind': [2, 2, 3, 0],
        'dtype': [8, 8, 0, 0],
        'rank': [2, 3, 1, 0],
        'shape': [4, 1, 0, 0,
                  1, 2, 3, 0,
                  100, 0, 0, 0,
                  0, 0, 0, 0],
        'shape_buckets': [2, 1, 0, 0,
                          1, 1, 2, 0,
                          3, 0, 0, 0,
                          0, 0, 0, 0],
        'floats': [1.0, 0.0, 0.5, 0.5,
                   9.0, -0.25, 13/6, 13.5/6,
                   30, -15, 5, 15,
                   0, 0, 0, 0],
        'float_buckets': [3, 2, 3, 3,
                          3, 1, 3, 3,
                          4, 0, 3, 4,
                          2, 2, 2, 2],
        'counts': [4, 4, 2, 2, 2, 0, 4, 2,
                   6, 6, 4, 1, 0, 1, 2, 6,
                   100, 3, 1, 1, 0, 1, 1, 3,
                   1, 1, 0, 1, 0, 0, 1, 1],
        'count_buckets': [2, 2, 1, 1, 1, 0, 2, 1,
                          2, 2, 2, 1, 0, 1, 1, 2,
                          3, 2, 1, 1, 0, 1, 1, 2,
                          1, 1, 0, 1, 0, 0, 1, 1],
        'fractions': [4/4, 2/4, 2/4, 2/4, 0/4, 4/4, 2/4,
                      6/6, 4/6, 1/6, 0/6, 1/6, 2/6, 6/6,
                      3/100, 1/3, 1/3, 0/3, 1/3, 1/3, 3/3,
                      1/1, 0/1, 1/1, 0/1, 0/1, 1/1, 1/1],
        'booleans': [1, 1, 0, 1, 0, 1, 1, 0,
                     1, 1, 0, 0, 0, 0, 0, 1,
                     0, 1, 0, 0, 0, 0, 0, 1,
                     1, 1, 0, 1, 0, 1, 1, 1],
        'value_string': [
            b'tf.float32:[[1.0], [0.0], [1.0], [0.0]]',
            b'tf.float32:[[[0.5, 2.5, 9.0], [-0.25, 0.0, 1.25]]]',
            (b'SparseTensor(indices=tf.Tensor(\n[[12]\n [34]\n [56]], '
             b'shape=(3, 1), dtype=int64), '
             b'values=tf.Tensor([  0 -15  30], shape=(3,), dtype=int32), '
             b'dense_shape=tf.Tensor([100], shape=(1,), dtype=int64))'),
            b'0'],

        # Features from featurize_input_and_output.
        'io_comparisons': [2, 2, 2, 0, 2, 2, 1,
                           2, 0, 0, 2, 0, 1, 1,
                           1, 1, 1, 1, 1, 1, 1],
        'io_booleans': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'io_counts': [1, 2, 1,
                      1, 2, 1,
                      1, 1, 1],
        'io_count_buckets': [1, 1, 1,
                             1, 1, 1,
                             1, 1, 1],
        'io_fractions': [1/6, 2/4, 1/6, 1/4,
                         1/3, 2/4, 1/3, 1/4,
                         1/1, 1/1, 1/1, 1/1],

        # Features added in create_examples.
        'num_inputs': [2],
        'operations': expected_operations,
        'expression': [b'tf.dummy_expression(in1, in2)'],
    }

    print(tf_examples)

    self.assertLen(tf_examples, 2)
    actual_tf_example_1, actual_tf_example_2 = tf_examples

    # Check the entire first example.
    for key, expected in six.iteritems(expected_tf_example_1):
      some_list = actual_tf_example_1.features.feature[key]
      if some_list.HasField('float_list'):
        actual = some_list.float_list.value
        actual = [round(f, 6) for f in actual]
        expected = [round(f, 6) for f in expected]
      elif some_list.HasField('int64_list'):
        actual = some_list.int64_list.value
      elif some_list.HasField('bytes_list'):
        actual = some_list.bytes_list.value
      else:
        self.fail('Failed to extract list from TF example.')

      # Printing side-by-side like this in the test log is more helpful than the
      # AssertionError message. Look at the Python3 log, which prints ints
      # without the L suffix.
      print('key: {}\n'
            '  expected: {}\n'
            '  got:      {}'.format(key, expected, actual))
      self.assertEqual(actual, expected)

    # Less rigorous checks for the second example, where the two inputs have
    # swapped.
    self.assertEqual(
        actual_tf_example_2.features.feature['rank'].int64_list.value,
        [2, 1, 3, 0])
    self.assertEqual(
        actual_tf_example_2.features.feature['shape'].int64_list.value,
        [4, 1, 0, 0,
         100, 0, 0, 0,
         1, 2, 3, 0,
         0, 0, 0, 0])


if __name__ == '__main__':
  absltest.main()

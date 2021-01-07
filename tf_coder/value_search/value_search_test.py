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
"""Tests for value_search.py."""

import collections

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import mock
import numpy as np
import six
import tensorflow as tf
from tf_coder import tf_functions
from tf_coder.benchmarks import all_benchmarks
from tf_coder.benchmarks import benchmark as benchmark_module
from tf_coder.benchmarks import simple_benchmarks
from tf_coder.models import tensor_features_model
from tf_coder.natural_language import bag_of_words_handlers
from tf_coder.natural_language import description_handler_factory
from tf_coder.value_search import all_operations
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search
from tf_coder.value_search import value_search_settings as settings_module


def _constant(constant):
  return value_module.ConstantValue(constant)


def _inverse_sigmoid(np_array):
  return np.log(np_array / (1 - np_array))


def _parameterize_simple_benchmarks():
  """Creates parameterized test cases for all simple benchmarks.

  This is useful so that tests can run on each benchmark individually, instead
  of having a single test loop over all benchmarks. In this way, issues with
  multiple benchmarks can be identified in one round of testing, and it is
  clearer which of the benchmarks need further attention.

  Returns:
    A list of tuples (test_case_name, benchmark) for all benchmarks in
    simple_benchmarks.py.
  """
  parameterized_tuples = []
  for index, simple_benchmark in enumerate(
      all_benchmarks.all_benchmarks(modules=[simple_benchmarks])):
    # The index ensures all test cases have distinct names, even if multiple
    # benchmarks have the same name.
    test_case_name = '{index}_{name}'.format(index=index,
                                             name=simple_benchmark.name)
    parameterized_tuples.append((test_case_name, simple_benchmark))
  return parameterized_tuples


class ValueSearchTest(parameterized.TestCase):

  def setUp(self):
    super(ValueSearchTest, self).setUp()
    operations = all_operations.get_operations()
    self._constant_operation = None
    for operation in operations:
      if operation.name == tf_functions.CONSTANT_OPERATION_NAME:
        self._constant_operation = operation
        break
    self.assertIsNotNone(self._constant_operation)
    self.settings = settings_module.from_dict({'timeout': 10})

  @parameterized.named_parameters(
      ('not_sparse_list',
       [[1, 2], [3, 4, 5]],
       [1, 2, 3, 4, 5],
       'a description that does not mention that specific word',
       False),
      ('not_sparse_dict',
       {'in1': [1, 2], 'in2': [3, 4, 5]},
       [1, 2, 3, 4, 5],
       'a description that does not mention that specific word',
       False),
      ('sparse_input_list',
       [[1, 2], tf.SparseTensor(values=[1], indices=[[3]], dense_shape=[5])],
       [1, 2, 3, 4, 5],
       'a description that does not mention that specific word',
       True),
      ('sparse_input_dict',
       {'in1': [1, 2], 'in2': tf.SparseTensor(values=[1], indices=[[3]],
                                              dense_shape=[5])},
       [1, 2, 3, 4, 5],
       'a description that does not mention that specific word',
       True),
      ('sparse_output',
       [[1, 2], [3, 4, 5]],
       tf.SparseTensor(values=[1], indices=[[3]], dense_shape=[5]),
       'a description that does not mention that specific word',
       True),
      ('sparse_description',
       {'in1': [1, 2], 'in2': [3, 4, 5]},
       [1, 2, 3, 4, 5],
       'a description that does mention the word sparse',
       True),
  )
  def test_contains_sparse(self, inputs, output, description, expected):
    benchmark = benchmark_module.Benchmark(
        examples=[benchmark_module.Example(inputs, output)],
        description=description)
    self.assertEqual(value_search._contains_sparse(benchmark), expected)

  @mock.patch('sys.stdout', new_callable=six.StringIO)
  def test_add_constants_and_inputs_and_print(self, mock_stdout):
    values_by_weight = [collections.OrderedDict() for _ in range(100)]
    examples = [
        benchmark_module.Example(
            inputs=[
                [[[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]],  # Shape (1, 1, 1, 9).
                [list(range(6)) for _ in range(15)],  # Shape (15, 6).
                42,
            ],
            output=tf.constant(list(range(50)), dtype=tf.int64),  # Shape (50).
        ),
    ]
    constants = [123, 45, 1, 9]
    benchmark = benchmark_module.Benchmark(examples, constants)
    output_value = value_module.OutputValue(benchmark.examples[0].output)

    value_search._add_constants_and_inputs_and_print(
        values_by_weight, benchmark, output_value, self._constant_operation,
        self.settings)

    expected_constants_ordered = [
        # Provided constants.
        _constant(123), _constant(45), _constant(1), _constant(9),
        # Input variables.
        value_module.InputValue([1], 'in1'),  # Reconstructs to 'in1'.
        value_module.InputValue([2], 'in2'),  # Reconstructs to 'in2'.
        value_module.InputValue([3], 'in3'),  # Reconstructs to 'in3'.
        # Common constants.
        _constant(0), _constant(-1), _constant(True), _constant(False),
        # Common DTypes.
        _constant(tf.int32), _constant(tf.float32), _constant(tf.bool),
        # Primitive inputs as scalar tensors. The actual Value is an
        # OperationValue, but this ExpressionValue has the same expression.
        value_module.ExpressionValue(None, 'tf.constant(in3)'),
        # Axis constants.
        _constant(2), _constant(3),
        # Uncomon DTypes.
        _constant(tf.int64),
        # Shape constants.
        _constant(6), _constant(15), _constant(50),
        # Output shape tuple.
        _constant((50,)),
    ]
    # values_by_weight is a list of OrderedDicts. Convert to list of lists.
    actual_constants_ordered = sum([list(ordered_dict)
                                    for ordered_dict in values_by_weight], [])
    self.assertEqual(
        [v.reconstruct_expression() for v in actual_constants_ordered],
        [v.reconstruct_expression() for v in expected_constants_ordered])

    # The output shape tuple (50,) is included as a constant, but not printed.
    self.assertIn(
        'Constants: [123, 45, 1, 9, 0, -1, True, False, 2, 3, 6, 15, 50]\n',
        mock_stdout.getvalue())

  @parameterized.named_parameters(
      ('keyword', 'def'),
      ('literal', 'True'),
      ('space', ' '),
      ('number', '0'),
      ('tf', 'tf'),
      ('np', 'np'),
      ('dot', 'x.y'),
      ('not_string', 1))
  def test_add_constants_and_inputs_and_print_checks_names(self, bad_name):
    values_by_weight = [collections.OrderedDict() for _ in range(100)]
    examples = [benchmark_module.Example(inputs={bad_name: [1, 2, 3]},
                                         output=[1, 2])]
    benchmark = benchmark_module.Benchmark(examples, constants=[])
    output_value = value_module.OutputValue(benchmark.examples[0].output)
    with self.assertRaises(ValueError):
      value_search._add_constants_and_inputs_and_print(
          values_by_weight, benchmark, output_value, self._constant_operation,
          self.settings)

  @parameterized.named_parameters(
      ('with_description', 'do foo to tensor', True),
      ('empty_description', '', False),
      ('none_description', None, False))
  @mock.patch('sys.stdout', new_callable=six.StringIO)
  def test_add_constants_and_inputs_and_print_handles_description(
      self, description, expect_print, mock_stdout):
    examples = [benchmark_module.Example(inputs=[[1, 4], [2, 7]],
                                         output=[3, 11])]
    benchmark = benchmark_module.Benchmark(examples, description=description)
    values_by_weight = [collections.OrderedDict() for _ in range(5)]
    output_value = value_module.OutputValue(benchmark.examples[0].output)
    value_search._add_constants_and_inputs_and_print(
        values_by_weight, benchmark, output_value, self._constant_operation,
        self.settings)
    if expect_print:
      self.assertIn('Description: {}\n'.format(description),
                    mock_stdout.getvalue())
    else:
      self.assertNotIn('Description:', mock_stdout.getvalue())

  @parameterized.named_parameters(_parameterize_simple_benchmarks())
  def test_run_value_search_works_for_simple_benchmarks(self, simple_benchmark):
    handler = description_handler_factory.create_handler('tfidf')
    results = value_search.run_value_search(
        benchmark=simple_benchmark,
        settings=settings_module.from_dict({
            'timeout': 20,
            'printing.statistics': True}),
        description_handler=handler)
    self.assertLen(results.solutions, 1)
    self.assertIsNotNone(results.solutions[0].expression)
    self.assertGreater(results.solutions[0].weight, 0)
    self.assertGreater(results.total_time, 0)
    self.assertNotEmpty(results.value_set)
    self.assertGreaterEqual(results.statistics.total_apply_count, 0)

  @parameterized.named_parameters(
      ('stackoverflow_06', 'stackoverflow_06', 10.0,
       'tf.cast(tf.equal(in1, tf.expand_dims(in1, 1)), tf.float32)'),
      ('stackoverflow_22', 'stackoverflow_22', 10.0,
       'tf.tensordot(tf.cast(in1, tf.float32), in2, 1)'),
      ('google_01', 'google_01', 90.0,
       'tf.cast(tf.where(tf.sequence_mask(tf.math.bincount(in1))), tf.int32)'))
  def test_run_value_search_works_for_important_benchmarks(
      self, benchmark_name, timeout, solution):
    # These benchmarks are frequently used in demos. We should ensure that they
    # are solved quickly and consistently. While the exact solution may change
    # as the algorithm improves, we should ensure that the solution is always
    # simple and understandable.
    handler = description_handler_factory.create_handler('tfidf')

    benchmark = all_benchmarks.find_benchmark_with_name(benchmark_name)
    results = value_search.run_value_search(
        benchmark=benchmark,
        description_handler=handler,
        settings=settings_module.from_dict({'timeout': timeout}))
    print('Time for benchmark {}: {:.2f} sec'.format(
        benchmark_name, results.total_time))
    self.assertLen(results.solutions, 1)
    self.assertEqual(results.solutions[0].expression, solution)

  @parameterized.named_parameters(
      ('true', True, ['tf.add(in1, in2)', 'tf.add(in2, in1)']),
      ('false', False, ['tf.add(in1, in2)',
                        'tf.add(in2, in1)',
                        'tf.add_n((in1, in2))',
                        'tf.add_n((in2, in1))']))
  def test_run_value_search_only_minimal_solutions(
      self, only_minimal_solutions, expected_solutions):
    benchmark = benchmark_module.Benchmark(
        examples=[benchmark_module.Example(inputs=[[1, 4], [2, 7]],
                                           output=[3, 11])])
    results = value_search.run_value_search(
        benchmark=benchmark,
        settings=settings_module.from_dict({
            'timeout': 20,
            'max_solutions': 4,
            'only_minimal_solutions': only_minimal_solutions,
            'max_extra_solutions_time': 20}))
    self.assertLen(results.solutions, len(expected_solutions))
    self.assertEqual([solution.expression for solution in results.solutions],
                     expected_solutions)

  @parameterized.named_parameters(
      ('stackoverflow_06', 'stackoverflow_06', 0),
      ('stackoverflow_22', 'stackoverflow_22', 1))
  def test_run_value_search_works_for_small_max_weight(
      self, benchmark_name, max_weight):
    benchmark = all_benchmarks.find_benchmark_with_name(benchmark_name)
    results = value_search.run_value_search(
        benchmark=benchmark,
        settings=settings_module.from_dict({'max_weight': max_weight}))
    print("Note: if you're inspecting the test log, the above failure is "
          "expected in test_run_value_search_works_for_small_max_weight.")
    self.assertEmpty(results.solutions)

  def test_run_value_handles_large_weight_constants(self):
    benchmark = benchmark_module.Benchmark(
        examples=[benchmark_module.Example(inputs=[[1], [2]], output=[[3]])])
    results = value_search.run_value_search(
        benchmark=benchmark, settings=self.settings)
    self.assertNotEmpty(results.solutions)
    self.assertEqual(results.solutions[0].expression,
                     'tf.add(in1, tf.expand_dims(in2, 0))')
    output_shape_constant = value_module.ConstantValue((1, 1))
    self.assertIn(output_shape_constant, results.value_set)
    # Find the element in value_set equal to output_shape_constant and assert
    # that it's actually a ConstantValue, as opposed to an OperationValue.
    for value in results.value_set:
      if value == output_shape_constant:
        self.assertIsInstance(value, value_module.ConstantValue)

  def test_run_value_search_from_example(self):
    results = value_search.run_value_search_from_example(
        inputs=[
            [1, 2, 3],
            [10, 20, 30]
        ],
        output=[11, 22, 33],
        settings=settings_module.from_dict({
            'timeout': 5, 'max_solutions': 2}))
    self.assertLen(results.solutions, 2)
    self.assertEqual(results.solutions[0].expression, 'tf.add(in1, in2)')
    self.assertEqual(results.solutions[1].expression, 'tf.add(in2, in1)')
    # Linter suggests self.assertLess() but it's wrong.
    self.assertTrue(0.0 < results.total_time < 5.0)  # pylint: disable=g-generic-assert

  def test_run_value_search_from_example_with_constants(self):
    results = value_search.run_value_search_from_example(
        inputs=[
            [10, 20, 30],
        ],
        output=[54, 64, 74],
        settings=self.settings,
        constants=[44])
    self.assertLen(results.solutions, 1)
    self.assertEqual(results.solutions[0].expression,
                     'tf.add(in1, tf.constant(44))')

  def test_value_search_checks_bad_solutions(self):
    inputs = {'tensor': [10, 20, 30],
              'index': -2}
    output = 20
    constants = [1, 20]

    results = value_search.run_value_search_from_example(
        inputs=inputs, output=output, constants=constants,
        settings=settings_module.from_dict({'require_all_inputs_used': True}))
    self.assertLen(results.solutions, 1)
    self.assertEqual(results.solutions[0].expression, 'tensor[index]')

    results = value_search.run_value_search_from_example(
        inputs=inputs, output=output, constants=constants,
        settings=settings_module.from_dict({'require_all_inputs_used': False,
                                            'require_one_input_used': True}))
    self.assertLen(results.solutions, 1)
    self.assertEqual(results.solutions[0].expression, 'tensor[1]')

    results = value_search.run_value_search_from_example(
        inputs=inputs, output=output, constants=constants,
        settings=settings_module.from_dict({'require_all_inputs_used': False,
                                            'require_one_input_used': False}))
    self.assertLen(results.solutions, 1)
    self.assertEqual(results.solutions[0].expression, 'tf.constant(20)')

  def test_value_search_prioritizes_operations(self):
    # Make sure prioritized ops get printed out.
    settings = settings_module.from_dict({
        'timeout': 0.5,  # We don't actually care about solving the problem.
        'printing.prioritized_operations': True,
        'printing.deprioritized_operations': True,
        'tensor_model.prioritize_threshold': 0.5,
        'tensor_model.deprioritize_threshold': 0.2,
    })

    # Create data for the mocks.
    all_ops = all_operations.get_operations(include_sparse_operations=True)
    op_names = [op.name for op in all_ops]
    abs_index = None  # Prioritized by tensor features model.
    argmax_index = None  # Deprioritized by tensor features model.
    for i, op in enumerate(all_ops):
      if op.name == 'tf.abs(x)':
        abs_index = i
      elif op.name == 'tf.argmax(input, axis)':
        argmax_index = i
    self.assertIsNotNone(abs_index)
    self.assertIsNotNone(argmax_index)

    operation_probs = np.repeat(0.4, len(all_ops))
    operation_probs[abs_index] = 0.6  # Above prioritize threshold.
    operation_probs[argmax_index] = 0.1  # Below deprioritize threshold.
    operation_logits = np.expand_dims(_inverse_sigmoid(operation_probs), axis=0)

    nl_data = [
        # Associate "apple" with tf.zeros.
        {'docstring': ['apple pie'], 'tf_functions': ['tf.zeros'],
         'comments': [], 'names': [], 'strings': []},
    ] * 100

    # Mock the tensor model's predictions, the NL model's data, and print().
    with mock.patch('tf_coder.models.tensor_features_model.eval_single_example',
                    return_value=tensor_features_model.Result(
                        operation_logits=operation_logits)), \
        mock.patch('tf_coder.datasets.github.data_loader.load_data',
                   return_value=nl_data), \
        mock.patch('sys.stdout', new_callable=six.StringIO) as mock_stdout:

      handler = bag_of_words_handlers.NaiveBayesDescriptionHandler(
          max_num_prioritized=1)
      benchmark = benchmark_module.Benchmark(
          # I/O example doesn't matter.
          examples=[benchmark_module.Example({'my_var': [1, 2]}, [2, 1])],
          # Description contains "apple"!
          description='honeycrisp apple')

      value_search.run_value_search(
          benchmark=benchmark,
          description_handler=handler,
          settings=settings,
          tensor_model=mock.Mock(),
          tensor_config={'max_num_inputs': 3,
                         'operation_names': op_names})

      self.assertIn('BOW handler prioritized tf.zeros',
                    mock_stdout.getvalue())
      self.assertIn('Tensor features model prioritized tf.abs(x), p=0.6',
                    mock_stdout.getvalue())
      self.assertIn(
          'Tensor features model deprioritized tf.argmax(input, axis), p=0.1',
          mock_stdout.getvalue())

  @mock.patch('sys.stdout', new_callable=six.StringIO)
  def test_value_search_can_load_data(self, mock_stdout):
    benchmark = all_benchmarks.find_benchmark_with_name('simple_cast')
    handler = description_handler_factory.create_handler('naive_bayes')

    settings = settings_module.from_dict({
        'timeout': 5,
        'printing.prioritized_operations': True,
        'printing.deprioritized_operations': True,
    })

    tensor_config = tensor_features_model.load_config(
        settings.tensor_model.config_path)
    tensor_model = tensor_features_model.get_model(tensor_config)
    tensor_checkpoint = tensor_features_model.create_checkpoint(tensor_model)
    tensor_checkpoint.restore(
        settings.tensor_model.checkpoint_path).expect_partial()

    results = value_search.run_value_search(
        benchmark=benchmark,
        description_handler=handler,
        settings=settings,
        tensor_model=tensor_model,
        tensor_config=tensor_config)

    self.assertLen(results.solutions, 1)
    self.assertIn('BOW handler prioritized tf.cast(x, dtype)',
                  mock_stdout.getvalue())
    self.assertIn('Tensor features model prioritized tf.cast(x, dtype)',
                  mock_stdout.getvalue())


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

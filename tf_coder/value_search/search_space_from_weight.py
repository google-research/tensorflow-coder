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
"""Computes the size of value search's search space."""

import collections
import functools
import operator
import os
import sys

from absl import app
from absl import flags
from tf_coder import tf_coder_utils
from tf_coder import tf_functions
from tf_coder.benchmarks import all_benchmarks
from tf_coder.natural_language import description_handler_factory
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search
from tf_coder.value_search import value_search_settings as settings_module


FLAGS = flags.FLAGS

flags.DEFINE_string('benchmark_name', 'google_02',
                    'The name of a benchmark to analyze.')
flags.DEFINE_multi_string('settings',
                          [],
                          'Settings to override the defaults.')


# Inspired by https://stackoverflow.com/a/45669280/9589593.
class SuppressPrint(object):
  """A context manager for suppressing print() calls temporarily."""

  def __enter__(self):
    self._old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

  def __exit__(self, exc_type, exc_val, exc_tb):
    del exc_type, exc_val, exc_tb
    sys.stdout.close()
    sys.stdout = self._old_stdout


def compute_search_space_size(benchmark, settings, description_handler):
  """Computes and prints the size of the search space.

  This counts the total number of expressions with weight at most max_weight.
  The weights come from the benchmark (for constants and inputs) and the
  description handler (for determining the op weights). Distinct expressions
  will be counted separately even if they evaluate to the same value, unlike in
  TF-Coder's value_search algorithm which does value-based pruning.

  Args:
    benchmark: The Benchmark object defining the problem to analyze.
    settings: A Settings object containing settings for value search.
    description_handler: The DescriptionHandler used, which can modify weights
      of operations.

  Returns:
    Nothing. All output is printed to stdout.
  """

  max_weight = settings.max_weight
  print('Computing search space.\n'
        'Benchmark name: {}\n'
        'Description handler: {}\n'
        'Max weight: {}'.format(
            benchmark.name, description_handler, max_weight))

  # TODO(kshi): Update to load the tensor features model/config.
  operations = value_search.get_reweighted_operations(benchmark,
                                                      settings,
                                                      description_handler,
                                                      tensor_model=None,
                                                      tensor_config=None)

  # These loops are not the most efficient, but it doesn't really matter.
  print('\nFound {} operations.'.format(len(operations)))
  print()
  for weight in range(1, max(op.weight for op in operations) + 1):
    print('# operations with weight {}: {}'.format(
        weight, sum(1 for op in operations if op.weight == weight)))
  print()
  for arity in range(1, max(op.num_args for op in operations) + 1):
    print('# operations with arity {}: {}'.format(
        arity, sum(1 for op in operations if op.num_args == arity)))

  output_value = value_module.OutputValue(benchmark.examples[0].output)
  values_by_weight = [collections.OrderedDict()
                      for _ in range(max_weight + 1)]

  constant_operation = None
  for operation in operations:
    if operation.name == tf_functions.CONSTANT_OPERATION_NAME:
      constant_operation = operation
      break
  with SuppressPrint():
    value_search._add_constants_and_inputs_and_print(  # pylint: disable=protected-access
        values_by_weight, benchmark, output_value, constant_operation, settings)

  num_expressions_with_weight = [len(values_with_weight)
                                 for values_with_weight in values_by_weight]
  print()
  max_weight_with_initial_value = max(w for w in range(max_weight + 1)
                                      if num_expressions_with_weight[w])
  for weight in range(1, max_weight_with_initial_value + 1):
    print('# initial values with weight {}: {}'.format(
        weight, num_expressions_with_weight[weight]))

  for total_weight in range(2, max_weight + 1):
    for operation in operations:
      # All operations should have strictly positive weight and num_args.
      op_weight = operation.weight
      op_arity = operation.num_args

      if total_weight - op_weight < op_arity:
        continue

      # Partition `total_weight - op_weight` into `op_arity` positive pieces.
      # Equivalently, partition `total_weight - op_weight - op_arity` into
      # `op_arity` nonnegative pieces.
      for partition in tf_coder_utils.generate_partitions(
          total_weight - op_weight - op_arity, op_arity):
        arg_weights = [part + 1 for part in partition]
        num_expressions_with_weight[total_weight] += functools.reduce(
            operator.mul,
            (num_expressions_with_weight[w] for w in arg_weights))

  print()
  for weight in range(1, max_weight + 1):
    print('# expressions with weight exactly {}: {}'.format(
        weight, num_expressions_with_weight[weight]))

  print()
  for weight in range(1, max_weight + 1):
    print('# expressions with weight up to {}: {}'.format(
        weight, sum(num_expressions_with_weight[:weight + 1])))


def main(unused_argv):
  settings = settings_module.from_list(FLAGS.settings)
  description_handler = description_handler_factory.create_handler(
      settings.description_handler_name)
  benchmark = all_benchmarks.find_benchmark_with_name(FLAGS.benchmark_name)
  if not benchmark:
    raise ValueError('Unknown benchmark: {}'.format(FLAGS.benchmark_name))

  compute_search_space_size(benchmark=benchmark,
                            settings=settings,
                            description_handler=description_handler)

if __name__ == '__main__':
  app.run(main)

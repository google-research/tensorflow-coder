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
"""Exhaustive value search (enumerating by weight of expression)."""

import collections
import keyword
import re
import sys
import timeit
import tokenize
from typing import Any, Dict, List, NamedTuple, Optional, Set, Text, Tuple, Union

from absl import logging
import numpy as np
import six
import tensorflow as tf
from tf_coder import tf_functions
from tf_coder.benchmarks import benchmark as benchmark_module
from tf_coder.datasets import collect_tensor_data
from tf_coder.models import tensor_features_model
from tf_coder.natural_language import description_handler as description_handler_module
from tf_coder.value_search import all_operations
from tf_coder.value_search import filtered_values_cache
from tf_coder.value_search import operation_base
from tf_coder.value_search import operation_filtering
from tf_coder.value_search import operation_statistics
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search_settings as settings_module


ValuesByWeight = operation_base.ValuesByWeightDict
DescriptionHandler = description_handler_module.DescriptionHandler

Solution = NamedTuple('Solution', [
    ('value', value_module.Value),
    ('expression', Text),
    ('weight', int),
    ('time', float),
])

ValueSearchResults = NamedTuple('ValueSearchResults', [
    ('solutions', List[Solution]),
    ('total_time', float),
    ('value_set', Set[value_module.Value]),
    ('values_by_weight', ValuesByWeight),
    ('benchmark', benchmark_module.Benchmark),
    ('settings', settings_module.Settings),
    ('statistics', Optional[operation_statistics.OperationStatistics]),
])


def _suppress_warnings() -> None:
  """Suppress TensorFlow and Numpy warnings."""
  # TensorFlow will produce tons of error logging because we often apply
  # TensorFlow operations with bad arguments. Suppressing logging noticeably
  # improves performance.
  logging.set_verbosity(logging.ERROR)

  # Numpy sometimes produces warnings for overflow, etc., which can be
  # distracting.
  np.seterr(all='ignore')


def _user_inputs(inputs: Union[Dict[Text, Any], List[Any]]) -> List[Any]:
  """Takes the inputs dict or list and extracts the input tensors."""
  if isinstance(inputs, list):
    return inputs
  elif isinstance(inputs, dict):
    return list(inputs.values())
  elif isinstance(inputs, tuple):
    return list(inputs)
  else:
    raise ValueError('inputs must be a list or dict, but is {}'.format(
        type(inputs)))


def _contains_sparse(benchmark: benchmark_module.Benchmark) -> bool:
  """Returns whether the benchmark involves SparseTensors."""
  # TODO(kshi): These heuristics are okay, but we should let the user choose if
  # they want to.
  for example in benchmark.examples:
    if (isinstance(example.output, tf.SparseTensor) or
        any(isinstance(input_object, tf.SparseTensor)
            for input_object in _user_inputs(example.inputs))):
      return True
  return 'sparse' in benchmark.description.lower()


def _add_value_by_weight(values_by_weight: ValuesByWeight,
                         value: value_module.Value,
                         weight: int) -> None:
  """Adds a value of a given weight to values_by_weight."""
  if weight < len(values_by_weight):
    values_by_weight[weight][value] = value


def _constant_exists(constant: Any, constants_so_far: Set[Any]) -> bool:
  """Checks whether a constant exists already."""
  # We can't use the `in` keyword because `True in [1, 2, 3]` evaluates to True!
  # (`True == 1` evaluates to True.)
  return any(constant == existing and type(constant) is type(existing)
             for existing in constants_so_far)


def _is_valid_name(name: Text) -> bool:
  """Returns whether name is an acceptable Python identifier."""
  # Behavior is slightly different between Python versions, e.g., `await` is a
  # keyword only in PY3, and `print` is keyword only in PY2.
  if name in ['tf', 'np'] or keyword.iskeyword(name):
    return False
  if six.PY3:
    return name.isidentifier()
  else:
    return (bool(re.match(tokenize.Name + '$', name)) and
            name not in ['True', 'False', 'None'])


def _input_names_to_objects(
    inputs_collection: Union[List[Any], Dict[Text, Any]]) -> Dict[Text, Any]:
  """Returns a mapping from input names to objects, also validating names."""
  if isinstance(inputs_collection, (list, tuple)):
    input_names_to_objects = collections.OrderedDict(
        ('in' + str(i + 1), input_object)
        for i, input_object in enumerate(inputs_collection))
  elif isinstance(inputs_collection, dict):
    for name in inputs_collection:
      if not isinstance(name, six.string_types):
        raise ValueError('The input name {!r} must be a string.'.format(name))
      if not _is_valid_name(name):
        raise ValueError('The input name {!r} is not a valid Python identifier.'
                         .format(name))
    input_names_to_objects = inputs_collection
  else:
    raise ValueError('The collection of inputs has the wrong format. It can be '
                     'a list of input objects, or a dict mapping string names '
                     'to input objects.')
  return input_names_to_objects


def _add_constants_and_inputs_and_print(
    values_by_weight: ValuesByWeight,
    benchmark: benchmark_module.Benchmark,
    output_value: value_module.OutputValue,
    constant_operation: operation_base.Operation,
    settings: settings_module.Settings) -> None:
  """Adds constant/input Values to values_by_weight, and prints to stdout."""
  # Conceptually this is a set, but it's actually a list so that constants can
  # be printed in the same order they are chosen by the heuristics. The reduced
  # efficiency of membership-checking is not a big deal because we have few
  # constants.
  constants_so_far = set()
  constants_to_print = []

  # User-provided constants.
  for c in benchmark.constants:
    if not _constant_exists(c, constants_so_far):
      constant_value = value_module.ConstantValue(c)
      weight = tf_functions.PROVIDED_CONSTANT_WEIGHT
      _add_value_by_weight(values_by_weight, constant_value, weight)
      constants_so_far.add(c)
      constants_to_print.append(c)

  # Add inputs, while computing some info for extra constants later.
  max_input_tensor_rank = 0
  dimension_lengths = set()
  input_names_to_objects = _input_names_to_objects(benchmark.examples[0].inputs)
  for name, input_object in input_names_to_objects.items():
    input_value = value_module.InputValue(input_object, name)
    if input_value.is_tensor:
      max_input_tensor_rank = max(max_input_tensor_rank, len(input_value.shape))
      dimension_lengths.update(input_value.shape)
    if input_value.is_primitive and constant_operation is not None:
      scalar_tensor_value = constant_operation.apply([input_value], settings)
      _add_value_by_weight(values_by_weight, scalar_tensor_value,
                           tf_functions.PRIMITIVE_INPUT_AS_TENSOR_WEIGHT)

    _add_value_by_weight(values_by_weight, input_value,
                         tf_functions.INPUT_VARIABLE_WEIGHT)
    if input_value.is_primitive:
      constants_so_far.add(input_value.value)

    print("Input '{}':\n{!s}\n".format(name, input_value.value))

  if output_value.shape is not None:
    dimension_lengths.update(output_value.shape)

  # Always include these as constants.
  common_constants = [0, 1, -1, True, False]
  # Also include 2, 3, ..., max_example_input_tensor_rank - 1 when applicable.
  axis_constants = list(range(2, max_input_tensor_rank))
  # Also include dimension lengths of input and output tensors.
  shape_constants = sorted(dimension_lengths)

  constant_weight_pairs = (
      [(c, tf_functions.COMMON_CONSTANT_WEIGHT) for c in common_constants] +
      [(c, tf_functions.AXIS_CONSTANT_WEIGHT) for c in axis_constants] +
      [(c, tf_functions.SHAPE_CONSTANT_WEIGHT) for c in shape_constants])

  for constant, weight in constant_weight_pairs:
    if not _constant_exists(constant, constants_so_far):
      constant_value = value_module.ConstantValue(constant)
      _add_value_by_weight(values_by_weight, constant_value, weight)
      constants_so_far.add(constant)
      constants_to_print.append(constant)

  # DTypes for casting.
  for dtype, weight in tf_functions.CONSTANT_DTYPES_AND_WEIGHTS.items():
    dtype_value = value_module.ConstantValue(dtype)
    _add_value_by_weight(values_by_weight, dtype_value, weight)

  if output_value.shape:
    # Add the output shape as a constant.
    shape_tuple = tuple(output_value.shape)
    shape_tuple_value = value_module.ConstantValue(shape_tuple)
    weight = tf_functions.OUTPUT_SHAPE_TUPLE_WEIGHT
    _add_value_by_weight(values_by_weight, shape_tuple_value, weight)
    # Don't add shape_tuple to constants_to_print, because printing it out could
    # be confusing to users.

  # Only for experiments in the PLDI paper.
  if settings.paper_experiments.uniform_weights:
    # Count the number of values.
    num_values = sum(len(values_with_weight)
                     for values_with_weight in values_by_weight)
    # Take all values and put them in the collection for weight 1.
    for weight in range(2, len(values_by_weight)):
      for heavy_value in values_by_weight[weight]:
        values_by_weight[1][heavy_value] = heavy_value
      values_by_weight[weight].clear()
    # Make sure we did it right.
    for weight, values_with_weight in enumerate(values_by_weight):
      assert len(values_with_weight) == (num_values if weight == 1 else 0)

  print('Output:\n{!s}\n'.format(output_value.value))
  print('Constants: {!r}\n'.format(constants_to_print))
  if benchmark.description:
    print('Description: {}\n'.format(benchmark.description))
  print('Searching...\n')
  sys.stdout.flush()  # Flush so the inputs/output appear in Colab immediately.


def _check_solution(expression: Text,
                    used_input_names: Set[Text],
                    benchmark: benchmark_module.Benchmark,
                    settings: settings_module.Settings) -> bool:
  """Checks that the solution is good."""
  del expression  # Unused for now.

  if settings.require_all_inputs_used:
    if len(used_input_names) < len(benchmark.examples[0].inputs):
      return False
  elif settings.require_one_input_used:
    if not used_input_names:
      return False

  # TODO(kshi): Check that the solution works (floating-point errors may
  # accumulate beyond an acceptable threshold).
  return True


def _record_solutions(value: value_module.Value,
                      weight: int,
                      start_time: float,
                      solutions: List[Solution],
                      solution_expression_set: Set[Text],
                      benchmark: benchmark_module.Benchmark,
                      settings: settings_module.Settings) -> None:
  """Records new solutions in the `solutions` list."""
  reconstructions = value.reconstruct_all_expressions_with_input_names()
  this_solution_time = timeit.default_timer() - start_time
  for expression, used_input_names in reconstructions:
    if expression in solution_expression_set:
      continue
    if not _check_solution(expression, used_input_names, benchmark, settings):
      if settings.printing.bad_solutions:
        print('Bad solution: {}'.format(expression))
      continue
    solution_expression_set.add(expression)
    solutions.append(Solution(value=value, expression=expression,
                              weight=weight, time=this_solution_time))
    print('Found solution: {}'.format(expression))
    # Flush so the solutions appear in Colab immediately.
    sys.stdout.flush()
    if len(solutions) >= settings.max_solutions:
      break


def _find_solutions(
    benchmark: benchmark_module.Benchmark,
    operations: List[operation_base.Operation],
    start_time: float,
    settings: settings_module.Settings
) -> Tuple[List[Solution], Set[value_module.Value], ValuesByWeight,
           Optional[operation_statistics.OperationStatistics]]:
  """Helper, returning (solutions, value_set, values_by_weight, statistics)."""
  timeout_reached = False
  end_time = start_time + settings.timeout

  only_minimal_solutions = settings.only_minimal_solutions
  if settings.max_solutions == 1:
    # If we only want one solution, it will be minimal.
    only_minimal_solutions = True

  # An object to track statistics, if requested.
  statistics = (operation_statistics.OperationStatistics()
                if settings.printing.statistics
                else None)

  # A list of Solution namedtuples.
  solutions = []

  # A set of string solution expressions (don't return duplicate solutions).
  solution_expression_set = set()

  # The output value to search for.
  output_value = value_module.OutputValue(benchmark.examples[0].output)

  # A list of OrderedDicts mapping Value objects to themselves. The i-th
  # OrderedDict contains all Value objects of weight i.
  values_by_weight = [collections.OrderedDict()
                      for _ in range(settings.max_weight + 1)]

  # Find and cache the cast and constant operations for use later.
  cast_operation = None
  constant_operation = None
  for operation in operations:
    if operation.name == tf_functions.CAST_OPERATION_NAME:
      cast_operation = operation
    elif operation.name == tf_functions.CONSTANT_OPERATION_NAME:
      constant_operation = operation
  # Create the output dtype value for use later.
  dtype_value = value_module.ConstantValue(output_value.dtype)

  # Populate values_by_weight with inputs and constants. This also prints
  # inputs/output/constants to stdout.
  _add_constants_and_inputs_and_print(
      values_by_weight, benchmark, output_value, constant_operation, settings)

  # A set storing all values found so far.
  value_set = set().union(*values_by_weight)

  filter_cache = filtered_values_cache.FilteredValuesCache()

  # Value search by weight.
  for weight in range(1, settings.max_weight + 1):
    if settings.printing.progress:
      print('Searching weight {}...'.format(weight))

    # Values with the current weight. This might already include leaf values.
    new_values = values_by_weight[weight]

    for operation in operations:
      for value in operation.enumerate_values_with_weight(
          target_weight=weight,
          values_by_weight=values_by_weight,
          filter_cache=filter_cache,
          end_time=end_time,
          settings=settings,
          statistics=statistics):

        if value not in value_set:
          # This value has never been seen before, or it's the desired output.
          if settings.printing.verbose:
            expression = value.reconstruct_expression()
            print('{} produces:\n{}'.format(expression, value))

          if value == output_value:
            possible_first_solution = not solutions
            # Found solution(s), but some may be bad.
            _record_solutions(value, weight, start_time, solutions,
                              solution_expression_set, benchmark, settings)
            if possible_first_solution and solutions:
              end_time = min(
                  end_time,
                  timeit.default_timer() + settings.max_extra_solutions_time)
            if len(solutions) >= settings.max_solutions:
              return solutions, value_set, values_by_weight, statistics
          else:
            # Only store the value if it isn't a solution. Otherwise, we'll get
            # lots of "almost duplicate" solutions, e.g., by adding 0.
            new_values[value] = value
            # We should never add output_value (or anything equal) to value_set
            # so that we can continue finding other solutions.
            value_set.add(value)
        else:  # This value has been seen before.
          if value in new_values:
            # The value was already computed differently with this weight.
            original_value = new_values[value]
            if isinstance(original_value, value_module.OperationValue):
              # Only merge reconstructions if this was originally an
              # OperationValue. (It could be a ConstantValue instead.)
              operation_value = original_value   # type: value_module.OperationValue
              operation_value.merge_reconstructions(value)
          elif not only_minimal_solutions:
            # If we want non-minimal solutions, we need to store the value even
            # if we have already seen that value with a smaller weight.
            new_values[value] = value

      if timeit.default_timer() > end_time:
        timeout_reached = True
        # Don't return immediately; still try to cast new values because this is
        # relatively quick.
        break

    # Try casting new values to the output dtype if this has a chance of being
    # a correct solution.
    for new_value in new_values:
      if (cast_operation is not None and
          new_value.shape == output_value.shape and
          new_value.dtype != output_value.dtype and
          operation_filtering.is_castable(new_value, dtype_value)):
        casted_value = cast_operation.apply([new_value, dtype_value], settings)
        if casted_value == output_value:
          possible_first_solution = not solutions
          # Found solution(s), but some may be bad.
          _record_solutions(casted_value, weight, start_time, solutions,
                            solution_expression_set, benchmark, settings)
          if possible_first_solution and solutions:
            end_time = min(
                end_time,
                timeit.default_timer() + settings.max_extra_solutions_time)
          if len(solutions) >= settings.max_solutions:
            return solutions, value_set, values_by_weight, statistics

    if settings.printing.progress:
      print('Found {} distinct values of weight {}, or {} total.'.format(
          len(new_values), weight, len(value_set)))
    if only_minimal_solutions and solutions:
      return solutions, value_set, values_by_weight, statistics
    if timeout_reached:
      break

  return solutions, value_set, values_by_weight, statistics


def operation_multipliers_from_tensor_model(
    benchmark: benchmark_module.Benchmark,
    tensor_model: tensor_features_model.Model,
    tensor_config: Dict[Text, Any],
    settings: settings_module.Settings) -> Dict[Text, float]:
  """Runs the tensor features model to get operation weight multipliers."""
  # TODO(kshi): Duplicate creation of InputValue and OutputValue objects.
  inputs = [value_module.InputValue(user_input, '')
            for user_input in _user_inputs(benchmark.examples[0].inputs)]
  output = value_module.OutputValue(benchmark.examples[0].output)
  example_protos = collect_tensor_data.create_tf_examples(
      io_example=collect_tensor_data.IOExample(
          expression='',
          input_values=inputs,
          output_value=output,
          num_inputs=benchmark.num_inputs,
          operations=[]),
      max_num_inputs=tensor_config['max_num_inputs'],
      permute_inputs=False)
  if example_protos:
    example_proto = example_protos[0]
    result = tensor_features_model.eval_single_example(
        tensor_model, example_proto.SerializeToString())
    operation_ranks = tf.argsort(tf.argsort(tf.squeeze(result.operation_logits),
                                            stable=True))
    operation_probabilities = tf.squeeze(tf.sigmoid(result.operation_logits))
    operation_names = tensor_config['operation_names']
    assert len(operation_names) == len(operation_probabilities)

    multipliers = {}
    for index in tf.where(operation_probabilities
                          >= settings.tensor_model.prioritize_threshold):
      chosen_op_name = operation_names[int(index)]
      if settings.printing.prioritized_operations:
        print('Tensor features model prioritized {}, p={}'.format(
            chosen_op_name, operation_probabilities[int(index)]))
      multipliers[chosen_op_name] = settings.tensor_model.prioritize_multiplier
    for index in tf.where(operation_probabilities
                          <= settings.tensor_model.deprioritize_threshold):
      int_index = int(index)
      chosen_op_name = operation_names[int_index]
      if (int(operation_ranks[int_index])
          >= settings.tensor_model.max_deprioritized):
        continue
      if settings.printing.deprioritized_operations:
        print('Tensor features model deprioritized {}, p={}, logit={}'.format(
            chosen_op_name, operation_probabilities[int_index],
            result.operation_logits[0][int_index]))
      multipliers[chosen_op_name] = (
          settings.tensor_model.deprioritize_multiplier)
    return multipliers
  else:
    logging.error('Failed to create example for inputs %s and output %s',
                  inputs, output)
    return {}


def _combine_multipliers(first: Dict[Text, float],
                         second: Dict[Text, float]) -> Dict[Text, float]:
  """Combines operation weight multiplier dicts. Modifies the first dict."""
  for name in second:
    first[name] = first.get(name, 1.0) * second[name]
  return first


def get_reweighted_operations(
    benchmark: benchmark_module.Benchmark,
    settings: settings_module.Settings,
    description_handler: Optional[DescriptionHandler] = None,
    tensor_model: Optional[tensor_features_model.Model] = None,
    tensor_config: Optional[Dict[Text, Any]] = None,
) -> List[operation_base.Operation]:
  """Returns a list of operations with correct weights for the problem."""
  include_sparse_operations = (
      not settings.operations.limit_sparse_operations or
      _contains_sparse(benchmark))
  operations = all_operations.get_operations(
      include_sparse_operations=include_sparse_operations)

  operation_names = [op.name for op in operations]
  if len(operation_names) != len(set(operation_names)):
    raise ValueError('Operation names were not unique.')

  if settings.paper_experiments.uniform_weights:
    # Only for experiments in the PLDI paper.
    for operation in operations:
      operation.weight = 1
    return operations

  multipliers = {}
  if description_handler and benchmark.description:
    multipliers = _combine_multipliers(
        multipliers,
        description_handler.get_operation_multipliers(
            benchmark.description, settings))
  if tensor_model is not None and tensor_config is not None:
    multipliers = _combine_multipliers(
        multipliers,
        operation_multipliers_from_tensor_model(benchmark, tensor_model,
                                                tensor_config, settings))
  for operation in operations:
    operation.weight = max(
        1, int(round(operation.weight * multipliers.get(operation.name, 1))))

  return operations


def run_value_search(
    benchmark: benchmark_module.Benchmark,
    settings: settings_module.Settings,
    description_handler: Optional[DescriptionHandler] = None,
    tensor_model: Optional[tensor_features_model.Model] = None,
    tensor_config: Optional[Dict[Text, Any]] = None) -> ValueSearchResults:
  """Performs value search, iterating by the expression weight.

  Starts with the constants and user-provided inputs, and applies the given
  operations, for a given number of iterations. An expression's "weight" is the
  number of nodes in the expression tree.

  Args:
    benchmark: The Benchmark containing input-output examples and constants.
    settings: A Settings object containing settings for this search.
    description_handler: A DescriptionHandler that scores operations based on
      the benchmark's description.
    tensor_model: The tensor features model to use, already restored from a
      checkpoint. If None, do not run the model.
    tensor_config: The config to use with the tensor features model.

  Returns:
    A ValueSearchResults namedtuple.

  Raises:
    ValueError: If max_weight is too large to be reasonable.
  """
  _suppress_warnings()
  if len(benchmark.examples) > 1:
    print('Warning: for now, value search only uses a single example.')

  start_time = timeit.default_timer()

  operations = get_reweighted_operations(
      benchmark,
      settings,
      description_handler=description_handler,
      tensor_model=tensor_model,
      tensor_config=tensor_config)

  solutions, value_set, values_by_weight, statistics = _find_solutions(
      benchmark=benchmark,
      operations=operations,
      start_time=start_time,
      settings=settings)

  total_time = timeit.default_timer() - start_time

  if solutions:
    print()
    print('Solution was found in {:.1f} seconds:\n{}'.format(
        solutions[0].time, solutions[0].expression))
    if settings.max_solutions != 1:
      print('Found {} solution(s) in {:.1f} seconds total.'.format(
          len(solutions), total_time))
  else:
    print('Could not find solution within {} seconds.'.format(
        min(settings.timeout, total_time)))
  sys.stdout.flush()

  return ValueSearchResults(
      solutions=solutions,
      total_time=total_time,
      value_set=value_set,
      values_by_weight=values_by_weight,
      benchmark=benchmark,
      settings=settings,
      statistics=statistics)


def run_value_search_from_example(
    inputs: Union[List[Any], Dict[Text, Any]],
    output: Any,
    settings: Optional[settings_module.Settings] = None,
    **kwargs) -> ValueSearchResults:
  """Performs value search for a single user-provided input-output example.

  Args:
    inputs: A list of inputs, or a dict mapping input names to inputs.
    output: The corresponding desired output.
    settings: An optional Settings object to use, or None to use defaults.
    **kwargs: The kwarg 'constants' can be used to specify a list of constants,
      and 'description' can be used to provide a natural language description of
      the task. Other arguments are passed directly to run_value_search().

  Returns:
    A ValueSearchResults namedtuple.
  """
  if settings is None:
    settings = settings_module.default_settings()
  constants = kwargs.pop('constants', None)
  description = kwargs.pop('description', None)
  source = kwargs.pop('source', 'From user-provided example.')
  benchmark = benchmark_module.Benchmark(
      examples=[benchmark_module.Example(inputs, output)],
      constants=constants,  # Will turn into empty list if constants=None.
      description=description,  # Will turn into '' if description=None.
      source=source)

  return run_value_search(benchmark, settings, **kwargs)

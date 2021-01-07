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
"""Runs value search to get a dataset of I/O pairs with expressions."""

import collections
import copy
import enum
import itertools
import random
import typing
from typing import Any, Dict, List, NamedTuple, Optional, Text, Tuple, Union

from absl import logging
import six
import tensorflow as tf
from tf_coder import tf_coder_utils
from tf_coder.value_search import all_operations
from tf_coder.value_search import operation_base
from tf_coder.value_search import value as value_module


IOExample = NamedTuple('IOExample',
                       [('expression', Text),
                        ('input_values', List[value_module.Value]),
                        ('output_value', value_module.Value),
                        ('num_inputs', int),
                        ('operations', List[operation_base.Operation])])

NEW_INPUT_NAME = 'NEW_INPUT'

EPSILON = 1e-8

# Bucket boundaries for nonnegative integers. Anything greater than or equal to
# 1000 will be placed in the last bucket, so the number of buckets equals the
# length of the list.
COUNT_BOUNDARIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    12, 15, 20, 25, 50, 75, 100, 200, 500, 1000]

# Bucket boundaries for arbitrary floats.
_POSITIVE_BOUNDARIES = [
    EPSILON, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
FLOAT_BOUNDARIES = ([-float('inf')] + [-x for x in _POSITIVE_BOUNDARIES[::-1]] +
                    _POSITIVE_BOUNDARIES)

# The max rank for tensor featurization.
MAX_RANK = 4


def extract_values_with_collapsed_subtrees(
    value: value_module.Value) -> List[value_module.Value]:
  """Collapses subtrees, see docstring of extract_examples_from_value."""
  if not isinstance(value, value_module.OperationValue):
    # Base case: the value is a leaf.
    return [value]

  value = typing.cast(value_module.OperationValue, value)

  results = []  # type: List[value_module.Value]

  # Each OperationApplication namedtuple represents one way of reaching this
  # Value by applying an operation to some arguments. Choose one at random.
  # If we recursed through all possible choices, we would heavily bias the
  # dataset toward Values with many possible expressions, e.g., those using many
  # commutative ops.
  operation_application = random.choice(value.operation_applications)

  children_possibilities = [extract_values_with_collapsed_subtrees(child)
                            for child in operation_application.arg_values]
  for children_choices in itertools.product(*children_possibilities):
    results.append(value_module.OperationValue(
        value=value.value,
        operation=operation_application.operation,
        arg_values=children_choices))

  # Include the case where the entire AST is replaced with a new input. Skip
  # tensor conversion, meaning don't convert a tuple of tensors into a single
  # tensor with one greater rank; keep the object as-is.
  results.append(value_module.InputValue(value=value.value,
                                         name=NEW_INPUT_NAME,
                                         skip_tensor_conversion=True))
  return results


def count_num_inputs(
    value: value_module.Value,
    name_counter: Optional[collections.Counter] = None
) -> Tuple[int, collections.Counter]:
  """Counts the number of distinct inputs, where all new inputs are distinct.

  The new inputs are created by extract_values_with_collapsed_subtrees, where
  each collapsed subtree becomes a new input.

  Args:
    value: The Value object to process.
    name_counter: Used for recursion by this function. Clients should ignore
      this argument.

  Returns:
    The number of inputs, and a Counter over the input names.
  """
  if name_counter is None:
    name_counter = collections.Counter()

  if isinstance(value, value_module.InputValue):
    name_counter[value.name] += 1
  elif isinstance(value, value_module.OperationValue):
    if len(value.operation_applications) != 1:
      raise ValueError('Expected exactly one way of constructing this Value.')
    for child in value.operation_applications[0].arg_values:
      count_num_inputs(child, name_counter)

  num_existing_names = len(name_counter)
  if NEW_INPUT_NAME in name_counter:
    num_existing_names -= 1
  num_inputs = num_existing_names + name_counter[NEW_INPUT_NAME]
  return num_inputs, name_counter


def normalize_names_and_extract_values(
    value: value_module.Value,
    name_counter: collections.Counter) -> Dict[Text, value_module.InputValue]:
  """Normalizes names of inputs to be in1, in2, etc., in place.

  Important: This implementation assumes that all InputValue objects in the tree
  have distinct identities, so changing the name of one InputValue will not
  affect other InputValues elsewhere in the tree. This requirement can be met by
  calling `value.copy()` before passing `value` to this function.

  Args:
    value: The Value to process.
    name_counter: A dict containing all existing names as keys, e.g., as
      returned by count_num_inputs.

  Returns:
    A dict mapping input names to InputValue objects.
  """
  renaming_map = {}
  for old_name in sorted(name_counter):
    if old_name == NEW_INPUT_NAME:
      continue  # Handle these later, not via the renaming_map.
    renaming_map[old_name] = 'in{}'.format(len(renaming_map) + 1)

  name_to_value_map = {}

  def recurse(value: value_module.Value, new_name_index: int) -> int:
    """Recursively normalizes names using renaming_map and new_name_index.

    Args:
      value: The Value to recurse on.
      new_name_index: An index for naming the next "new input" encountered,
        which was created via the subtree-collapsing process.

    Returns:
      The updated new_name_index to use in future recursion.
    """
    if isinstance(value, value_module.InputValue):

      if value.name in renaming_map:
        value.name = renaming_map[value.name]
      else:
        assert value.name == NEW_INPUT_NAME
        value.name = 'in{}'.format(new_name_index)
        new_name_index += 1

      if value.name in name_to_value_map:
        assert name_to_value_map[value.name] == value
      else:
        name_to_value_map[value.name] = value

    elif isinstance(value, value_module.OperationValue):
      if len(value.operation_applications) != 1:
        raise ValueError('Expected exactly one way of constructing this Value.')
      for child in value.operation_applications[0].arg_values:
        new_name_index = recurse(child, new_name_index)

    return new_name_index

  recurse(value, len(renaming_map) + 1)

  return name_to_value_map


def extract_operations(
    value: value_module.Value,
    operation_list: Optional[List[operation_base.Operation]] = None
) -> List[operation_base.Operation]:
  """Returns a list of Operations used in the Value."""
  if operation_list is None:
    operation_list = []

  if isinstance(value, value_module.OperationValue):
    if len(value.operation_applications) != 1:
      raise ValueError('Expected exactly one way of constructing this Value.')
    operation_list.append(value.operation_applications[0].operation)
    for child in value.operation_applications[0].arg_values:
      extract_operations(child, operation_list)

  return operation_list


def extract_examples_from_value(value: value_module.Value,
                                max_num_inputs: int = 3) -> List[IOExample]:
  """Extracts IOExample namedtuples from a Value.

  A Value represents an expression as well as its value when evaluated. Multiple
  I/O examples can be extracted from a single Value, where different
  subexpressions are considered inputs.

  For an example, we will denote tensors and other "complex" objects with
  floats, and "simple" constants with ints. Suppose that we run value search
  starting with inputs `in1 = 4.0` and `in2 = 3.0`, using a variety of int
  constants including `1` and `2`. During value search we may find a Value
  representing the expression `((1 + in1) * (in2 - 1)) + 2` that evaluates to
  12.0. From this Value, we may extract many examples including:

    * The original expression:
      ((1 + in1) * (in2 - 1)) + 2 -> 12.0, where in1 = 4.0, in2 = 3.0

    * Collapse different subtrees and treat them as new inputs:
      (in1 * (in2 - 1)) + 2 -> 12.0, where in1 = 5.0, in2 = 3.0
      ((1 + in1) * in2) + 2 -> 12.0, where in1 = 4.0, in2 = 2.0
      (in1 * in2) + 2 -> 12.0, where in1 = 5.0, in2 = 2.0
      in1 + 2 -> 12.0, where in1 = 10.0

  We do not turn constants into extra inputs because inputs are most commonly
  Tensors. We do not permute inputs (e.g., turning `tf.add(in1, in2)` into
  `tf.add(in2, in1)`) here because that can easily be done later. This also
  means we do not have to permute inputs within the expression string, which is
  tricky and unnecessary for synthetic data generation.

  Note that all expressions evaluate to the same result as the original Value
  does. This is sufficient because we can extract examples from all Values found
  during value search, which includes all subtrees of the current Value.

  Args:
    value: The starting Value to extract examples from.
    max_num_inputs: The maximum number of inputs for an extracted example. All
      extracted examples will have at least one input.

  Returns:
    A list of IOExample namedtuples.
  """
  values = extract_values_with_collapsed_subtrees(value)
  if values[-1].reconstruct_expression(use_cache=False) == NEW_INPUT_NAME:
    # The last element is the case where the entire expression is turned into a
    # single new input, which is not useful.
    del values[-1]

  # Create IOExamples by filtering out Values with too many inputs and
  # normalizing names.
  examples = []
  for v in values:
    # Copy (recursively) to completely isolate different expressions and
    # different subtrees within one expression.
    v = v.copy()

    num_inputs, name_counter = count_num_inputs(v)
    if not 0 < num_inputs <= max_num_inputs:
      # If `value` contains only constants and no inputs, then the unchanged
      # expression will have 0 inputs, but inputs will be introduced by
      # collapsing subtrees.
      continue
    name_to_value_map = normalize_names_and_extract_values(v, name_counter)
    assert len(name_to_value_map) == num_inputs

    examples.append(IOExample(
        expression=v.reconstruct_expression(use_cache=False),
        input_values=[name_to_value_map[name]
                      for name in sorted(name_to_value_map)],
        output_value=v,
        num_inputs=num_inputs,
        operations=extract_operations(v)))

  return examples


def _bucket(number: float, bucket_boundaries: List[float]) -> int:
  """Returns an int representing a bucket for the number."""
  for index, (left, right) in enumerate(zip(bucket_boundaries,
                                            bucket_boundaries[1:])):
    if left <= number < right:
      return index
  return len(bucket_boundaries) - 1


def _as_tensor_to_featurize(
    value: value_module.Value) -> Union[tf.Tensor, tf.SparseTensor]:
  """Extracts a Tensor or SparseTensor to featurize."""
  to_featurize = None
  if value.is_tensor or value.is_sparse_tensor:
    to_featurize = value.value
  elif value.is_primitive and isinstance(value.value, (int, float, bool)):
    to_featurize = tf.constant(value.value)
  elif value.is_sequence:
    if value.elem_type_is_tensor or value.elem_type_is_sparse_tensor:
      to_featurize = value.value[0]
    else:
      # In order to be a valid Value, a possibly-multidimensional sequence of
      # primitives must be convertible to a tensor.
      to_featurize = tf.constant(value.value)

  if to_featurize is None:
    raise ValueError('Could not featurize value {}'.format(value))

  assert isinstance(to_featurize, (tf.Tensor, tf.SparseTensor))
  return to_featurize


def _elements(tensor: Union[tf.Tensor, tf.SparseTensor]) -> tf.Tensor:
  """Returns a Tensor of non-default elements in a Tensor or SparseTensor."""
  return tensor.values if isinstance(tensor, tf.SparseTensor) else tensor


def _elements_list(tensor: Union[tf.Tensor, tf.SparseTensor]) -> List[float]:
  """Returns a list of non-default elements in a Tensor or SparseTensor."""
  return list(map(float, tf.reshape(_elements(tensor), [-1]).numpy()))


def _safe_divide(numerator: int, denominator: int) -> float:
  """Normal division except anything/0 becomes 0.0."""
  return numerator / denominator if denominator else 0.0


def _fixed_length_shape(tensor) -> List[int]:
  """Returns the tensor shape, truncated or padded with 0 to length MAX_RANK."""
  fixed_length_shape = list(tensor.shape[:MAX_RANK])
  fixed_length_shape += [0] * (MAX_RANK - len(fixed_length_shape))
  return fixed_length_shape


def featurize_value(
    value: value_module.Value) -> Dict[Text, Union[List[float], List[Text]]]:
  """Returns a dict that featurizes a Value.

  The feature dict will contain the following:

   * 'kind': One int denoting whether the value is a primitive, sequence,
       Tensor, SparseTensor, or other.
   * 'dtype': One int denoting the DType of the value.
   * 'rank': One int in the range [0, MAX_RANK], where MAX_RANK denotes any rank
       greater than or equal to MAX_RANK.
   * 'shape': A list of MAX_RANK ints denoting the shape, 0-padded.
   * 'shape_buckets': Like 'shape' but bucketed.
   * 'floats': Stats about the elements as raw floats.
   * 'float_buckets': Like 'floats' but bucketed.
   * 'counts': Numbers of elements with various properties.
   * 'count_buckets': Like 'counts' but bucketed.
   * 'fractions': Fractions of elements with various properties.
   * 'booleans': Boolean properties of the value, as 0-1 ints.
   * 'value_string': The input as a string.

  Args:
    value: The Value to featurize.
  """
  to_featurize = _as_tensor_to_featurize(value)
  features = {}  # type: Dict[Text, Union[List[float], List[Text]]]

  features['kind'] = [(
      0 if value.is_primitive else  # pylint: disable=g-long-ternary
      1 if value.is_sequence else  # pylint: disable=g-long-ternary
      2 if value.is_tensor else
      3 if value.is_sparse_tensor else
      4)]

  supported_dtypes = (
      tf_coder_utils.INT_DTYPES + tf_coder_utils.FLOAT_DTYPES + (tf.bool,))
  if to_featurize.dtype not in supported_dtypes:
    raise ValueError('Cannot featurize unsupported dtype: {}'.format(
        to_featurize.dtype))
  features['dtype'] = [supported_dtypes.index(to_featurize.dtype)]

  features['rank'] = [min(len(to_featurize.shape), MAX_RANK)]

  features['shape'] = _fixed_length_shape(to_featurize)
  features['shape_buckets'] = [
      _bucket(dimension_length, COUNT_BOUNDARIES)
      for dimension_length in features['shape']]

  # "Elements" are the provided values (without default elements in
  # SparseTensors). Cast elements to float32 because, e.g., we can't compute the
  # mean of a bool tensor.
  elements = tf.cast(_elements(to_featurize), tf.float32)
  elements_list = _elements_list(to_featurize)

  max_value = tf_coder_utils.max_tensor_value(elements)
  min_value = tf_coder_utils.min_tensor_value(elements)
  mean_value = float(tf.reduce_mean(elements))
  mean_magnitude = float(tf.reduce_mean(tf.abs(elements)))
  features['floats'] = [max_value, min_value, mean_value, mean_magnitude]
  features['float_buckets'] = [_bucket(f, FLOAT_BOUNDARIES)
                               for f in features['floats']]

  # The total size of the tensor, meaning the product of dimension lengths. This
  # may be huge for SparseTensors.
  size = tf_coder_utils.num_tensor_elements(to_featurize)
  # For SparseTensors, num_elements != size.
  num_elements = len(elements_list)

  positive = int(tf.reduce_sum(tf.cast(elements > 0, tf.int32)))
  zeros = int(tf.reduce_sum(tf.cast(
      tf.abs(tf.cast(elements, tf.float32)) < EPSILON, tf.int32)))
  ones = int(tf.reduce_sum(tf.cast(
      tf.abs(tf.cast(elements, tf.float32) - 1.0) < EPSILON, tf.int32)))
  negative = int(tf.reduce_sum(tf.cast(elements < 0, tf.int32)))
  probabilities = int(tf.reduce_sum(tf.cast(
      tf.logical_and(0 <= elements, elements <= 1), tf.int32)))
  unique = len(set(elements_list))

  counts = [size, num_elements, positive, zeros, ones, negative, probabilities,
            unique]

  features['counts'] = counts
  features['count_buckets'] = [_bucket(c, COUNT_BOUNDARIES)
                               for c in features['counts']]

  assert counts[0] == size and counts[1] == num_elements
  features['fractions'] = [_safe_divide(num_elements, size)]
  features['fractions'].extend([_safe_divide(c, num_elements)
                                for c in counts[2:]])

  # TODO(kshi): Consider computing the number of times the mode appears.

  # TODO(kshi): Consider computing, for each axis, whether that axis represents
  # a probability distribution, with all elements in [0, 1] summing to 1.

  is_sorted = (int(tf.reduce_all(tf.equal(elements, tf.sort(elements))))
               if len(elements.shape) else 1)
  is_finite = int(tf.reduce_all(tf.math.is_finite(elements)))
  all_positive = int(positive == num_elements)
  all_nonnegative = int(positive + zeros == num_elements)
  all_negative = int(negative == num_elements)
  all_zero_one = int(zeros + ones == num_elements)
  all_probabilities = int(probabilities == num_elements)
  all_unique = int(unique == num_elements)

  features['booleans'] = [
      is_sorted, is_finite, all_positive, all_nonnegative, all_negative,
      all_zero_one, all_probabilities, all_unique]

  features['value_string'] = [repr(value)]

  return features


def _comparison(a, b) -> int:
  """Returns an int in [0, 2] representing the comparison result."""
  return 0 if a < b else 1 if a == b else 2


def featurize_input_and_output(
    input_value: value_module.Value,
    output_value: value_module.Value) -> Dict[Text, List[float]]:
  """Returns a dict that featurizes relationships between an input and output.

  The feature dict will have the following keys (all start with 'io_'):

   * 'io_comparisons', list[int]: Ints in the range [0, 2] denoting which is
       larger according to various metrics.
   * 'io_counts', list[int]: Numbers of elements with various properties.
   * 'io_count_buckets', list[int]: Like 'io_counts' but bucketed.
   * 'io_fractions', list[float]: Fractions of elements with various properties.
   * 'io_booleans', list[int]: Ints in the range [0, 1] denoting various boolean
       properties.

  Args:
    input_value: The input Value.
    output_value: The output Value.
  """
  input_tensor = _as_tensor_to_featurize(input_value)
  output_tensor = _as_tensor_to_featurize(output_value)

  features = {}  # type: Dict[Text, List[float]]

  input_size = tf_coder_utils.num_tensor_elements(input_tensor)
  output_size = tf_coder_utils.num_tensor_elements(output_tensor)

  input_elements = _elements_list(input_tensor)
  output_elements = _elements_list(output_tensor)
  input_num_elements = len(input_elements)
  output_num_elements = len(output_elements)

  input_rank = len(input_tensor.shape)
  output_rank = len(output_tensor.shape)

  input_shape = _fixed_length_shape(input_tensor)
  output_shape = _fixed_length_shape(output_tensor)

  features['io_comparisons'] = [
      _comparison(input_size, output_size),
      _comparison(input_num_elements, output_num_elements),
      _comparison(input_rank, output_rank),
  ]
  features['io_comparisons'].extend(
      [_comparison(input_length, output_length)
       for input_length, output_length in zip(input_shape, output_shape)])

  # Count elements appearing in both the input and output.
  input_elements_set = set(input_elements)
  output_elements_set = set(output_elements)

  inputs_in_output = sum(e in output_elements_set for e in input_elements)
  outputs_in_input = sum(e in input_elements_set for e in output_elements)
  num_unique_overlaps = len(input_elements_set & output_elements_set)

  features['io_counts'] = [inputs_in_output, outputs_in_input,
                           num_unique_overlaps]
  features['io_count_buckets'] = [_bucket(c, COUNT_BOUNDARIES)
                                  for c in features['io_counts']]
  features['io_fractions'] = [
      _safe_divide(inputs_in_output, input_num_elements),
      _safe_divide(outputs_in_input, output_num_elements),
      _safe_divide(num_unique_overlaps, input_num_elements),
      _safe_divide(num_unique_overlaps, output_num_elements),
  ]

  all_inputs_in_output = int(inputs_in_output == len(input_elements))
  all_outputs_in_input = int(outputs_in_input == len(output_elements))
  features['io_booleans'] = [
      int(input_tensor.shape.as_list() == output_tensor.shape.as_list()),
      int(input_tensor.dtype == output_tensor.dtype),
      all_inputs_in_output,
      all_outputs_in_input,
  ]
  features['io_booleans'].extend([
      int(input_length in output_shape) if input_length else 0
      for input_length in input_shape])
  features['io_booleans'].extend([
      int(output_length in input_shape) if output_length else 0
      for output_length in output_shape])

  return features


def create_examples(io_example: IOExample,
                    max_num_inputs: int = 3,
                    permute_inputs: bool = True) -> List[Dict[Text, Any]]:
  """Creates example dicts for the I/O example."""
  examples = []
  operation_list = all_operations.get_operations(include_sparse_operations=True)
  operation_counter = collections.Counter(
      [op.name for op in io_example.operations])
  operation_counts = [operation_counter[op.name] for op in operation_list]

  num_inputs = len(io_example.input_values)

  try:
    num_inputs_feature = min(num_inputs, max_num_inputs)
    output_features = featurize_value(io_example.output_value)
    input_features = []
    for input_value in io_example.input_values[:max_num_inputs]:
      combined_features = featurize_value(input_value)
      combined_features.update(
          featurize_input_and_output(input_value, io_example.output_value))
      input_features.append(combined_features)

    dummy_value = value_module.ConstantValue(0)
    dummy_input_features = featurize_value(dummy_value)
    dummy_input_features.update(
        featurize_input_and_output(dummy_value, dummy_value))

  except ValueError as e:
    logging.warning('%s: could not featurize IOExample %s', e, io_example)
    return []

  permutations = (itertools.permutations(range(num_inputs)) if permute_inputs
                  else [list(range(num_inputs))])
  for permutation in permutations:

    feature_dict = collections.defaultdict(list)
    feature_dict['num_inputs'] = [num_inputs_feature]
    feature_dict.update(copy.deepcopy(output_features))

    padded_input_features = [input_features[index] for index in permutation]
    for _ in range(max_num_inputs - num_inputs):
      padded_input_features.append(dummy_input_features)

    for input_features_dict in padded_input_features:
      for key, value in six.iteritems(input_features_dict):
        feature_dict[key].extend(value)

    feature_dict['operations'] = operation_counts
    feature_dict['expression'] = [io_example.expression]

    examples.append(feature_dict)

  return examples


def _bytes_feature(strings):
  """Returns a bytes_list from a list of strings."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[s.encode('utf-8') for s in strings]))


def _int64_feature(ints):
  """Returns an int64_list from a list of ints."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=ints))


def _float_feature(floats):
  """Returns a float_list from a list of numbers."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=floats))


@enum.unique
class FeatureType(enum.Enum):
  """Types of features in the tf.train.Example proto."""
  # TODO(kshi): Can the values be _int64_feature etc.? Pytype doesn't seem to
  # understand that correctly -- this may be a pytype bug.
  INT = 1
  FLOAT = 2
  STRING = 3


FEATURE_NAME_TO_TYPE = {
    # Features from featurize_value.
    'kind': FeatureType.INT,
    'dtype': FeatureType.INT,
    'rank': FeatureType.INT,
    'shape': FeatureType.INT,
    'shape_buckets': FeatureType.INT,
    'floats': FeatureType.FLOAT,
    'float_buckets': FeatureType.INT,
    'counts': FeatureType.INT,
    'count_buckets': FeatureType.INT,
    'fractions': FeatureType.FLOAT,
    'booleans': FeatureType.INT,
    'value_string': FeatureType.STRING,

    # Features from featurize_input_and_output.
    'io_comparisons': FeatureType.INT,
    'io_counts': FeatureType.INT,
    'io_count_buckets': FeatureType.INT,
    'io_fractions': FeatureType.FLOAT,
    'io_booleans': FeatureType.INT,

    # Features added in create_examples.
    'num_inputs': FeatureType.INT,
    'operations': FeatureType.INT,
    'expression': FeatureType.STRING,
}


def create_tf_examples(io_example: IOExample,
                       max_num_inputs: int = 3,
                       permute_inputs: bool = True) -> List[tf.train.Example]:
  """Creates tf.train.Example protos for the I/O example."""
  tf_examples = []
  for example in create_examples(io_example,
                                 max_num_inputs=max_num_inputs,
                                 permute_inputs=permute_inputs):
    feature_dict = {}
    assert set(FEATURE_NAME_TO_TYPE.keys()) == set(example.keys())

    for feature_name, feature_value in six.iteritems(example):
      feature_type = FEATURE_NAME_TO_TYPE[feature_name]
      conversion_function = (
          _int64_feature if feature_type == FeatureType.INT else  # pylint: disable=g-long-ternary
          _float_feature if feature_type == FeatureType.FLOAT else
          _bytes_feature if feature_type == FeatureType.STRING else
          None)
      assert conversion_function is not None
      feature_dict[feature_name] = conversion_function(feature_value)

    tf_examples.append(tf.train.Example(features=tf.train.Features(
        feature=feature_dict)))

  return tf_examples

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
"""Functions and arguments used in the TF-Coder project."""

import ast
import collections

import tensorflow as tf
from tf_coder import filter_group


FilterGroup = filter_group.FilterGroup


FunctionInfo = collections.namedtuple(
    'FunctionInfo',
    ['name', 'filter_group', 'weight'])


# Weights for leaf nodes in the AST.

# Constants given by the user.
PROVIDED_CONSTANT_WEIGHT = 7

# Ubiquitous constants: 0, 1, -1.
COMMON_CONSTANT_WEIGHT = 8

# A tf.constant() wrapper around an input primitive.
PRIMITIVE_INPUT_AS_TENSOR_WEIGHT = 9

# Int constants meant to be axis values, chosen based on input tensor ranks.
AXIS_CONSTANT_WEIGHT = 14

# Int constants obtained from input/output tensor shapes.
SHAPE_CONSTANT_WEIGHT = 24

# Weight of constructing a tuple with the output shape.
OUTPUT_SHAPE_TUPLE_WEIGHT = 32

# Input variable nodes (in1, in2, etc.).
INPUT_VARIABLE_WEIGHT = 8

# DTypes with weights to add to the pool of constants.
CONSTANT_DTYPES_AND_WEIGHTS = collections.OrderedDict([
    (tf.int32, 8),
    (tf.float32, 8),
    (tf.bool, 8),
    (tf.int64, 16),
])

# Used in value search for custom cast logic.
CAST_OPERATION_NAME = 'tf.cast(x, dtype)'

# Used in value search to convert primitive inputs (e.g., 3) into scalar tensors
# (e.g., tf.constant(3)).
CONSTANT_OPERATION_NAME = 'tf.constant(value)'


# A list of FunctionInfo namedtuples, each describing one function usable by a
# program synthesizer. Each FunctionInfo's name contains the function name along
# with the names of the arguments for that function, in the order given in the
# function's signature. A function may appear multiple times with different
# lists of usable arguments. This list is ordered, so value search will try
# earlier functions before later ones.

# FunctionInfo name format: "tf.module.function(arg_1, arg_2, arg_3='value')"
# means call the function `tf.module.function` with varying inputs `arg_1` and
# `arg_2`, where `arg_3` is fixed and set to the literal constant `'value'`.
TF_FUNCTIONS = [
    FunctionInfo(name='tf.abs(x)',
                 filter_group=FilterGroup.PRIMITIVE_OR_TENSOR_1,
                 weight=40),
    FunctionInfo(name='tf.add(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=28),
    FunctionInfo(name='tf.add_n(inputs)',
                 filter_group=FilterGroup.SEQUENCE_1,
                 weight=44),
    FunctionInfo(name='tf.argmax(input, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=32),
    FunctionInfo(name='tf.argmin(input, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=48),
    FunctionInfo(name='tf.argsort(values, axis, stable=True)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name=("tf.argsort(values, axis, direction='DESCENDING', "
                       "stable=True)"),
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=48),
    FunctionInfo(name='tf.boolean_mask(tensor, mask)',
                 filter_group=FilterGroup.TENSOR_BOOLTENSOR_2,
                 weight=28),
    FunctionInfo(name='tf.broadcast_to(input, shape)',
                 # TODO(kshi): BROADCAST_TO_2 is a workaround for b/124261199.
                 # Once that is fixed, the TENSOR_SHAPE filter_group would be
                 # simpler but might be slower (need to measure).
                 filter_group=FilterGroup.BROADCAST_TO_2,
                 weight=44),
    FunctionInfo(name=CAST_OPERATION_NAME,  # 'tf.cast(x, dtype)'.
                 filter_group=FilterGroup.CASTABLE_DTYPE_2,
                 weight=16),
    FunctionInfo(name='tf.clip_by_value(t, clip_value_min, clip_value_max)',
                 filter_group=FilterGroup.CLIP_BY_VALUE_3,
                 weight=44),
    FunctionInfo(name='tf.concat(values, axis)',
                 filter_group=FilterGroup.TENSORSEQUENCE_AXIS_2,
                 weight=36),
    FunctionInfo(name=CONSTANT_OPERATION_NAME,  # 'tf.constant(value)'.
                 filter_group=FilterGroup.NOT_TENSOR_1,
                 weight=23),  # Less weight than cast, accounting for the dtype.
    FunctionInfo(name='tf.constant(value, dtype)',
                 filter_group=FilterGroup.CASTABLE_DTYPE_2,
                 weight=24),
    FunctionInfo(name='tf.divide(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=28),
    FunctionInfo(name='tf.equal(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=24),
    FunctionInfo(name='tf.exp(x)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=52),
    FunctionInfo(name='tf.expand_dims(input, axis)',
                 filter_group=FilterGroup.EXPAND_DIMS_2,
                 weight=18),
    FunctionInfo(name='tf.eye(num_rows)',
                 filter_group=FilterGroup.EYE_1,
                 weight=40),
    FunctionInfo(name='tf.eye(num_rows, num_columns)',
                 filter_group=FilterGroup.EYE_ROWS_COLS_2,
                 weight=60),
    FunctionInfo(name='tf.eye(num_rows, dtype)',
                 filter_group=FilterGroup.EYE_ROWS_DTYPE_2,
                 weight=48),
    FunctionInfo(name='tf.fill(dims, value)',
                 filter_group=FilterGroup.SHAPE_PRIMITIVE_2,
                 weight=40),
    FunctionInfo(name='tf.gather(params, indices)',
                 filter_group=FilterGroup.GATHER_2,
                 weight=24),
    FunctionInfo(name='tf.gather(params, indices, axis, batch_dims)',
                 filter_group=FilterGroup.GATHER_4,
                 weight=48),
    FunctionInfo(name='tf.gather_nd(params, indices)',
                 filter_group=FilterGroup.GATHER_ND_2,
                 weight=28),
    FunctionInfo(name='tf.gather_nd(params, indices, batch_dims)',
                 filter_group=FilterGroup.GATHER_ND_3,
                 weight=48),
    FunctionInfo(name='tf.greater(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=24),
    FunctionInfo(name='tf.greater_equal(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=32),
    FunctionInfo(name='tf.math.bincount(arr)',
                 filter_group=FilterGroup.BINCOUNT_1,
                 weight=40),
    FunctionInfo(name='tf.math.ceil(x)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=44),
    FunctionInfo(name='tf.math.count_nonzero(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=56),
    FunctionInfo(name='tf.math.count_nonzero(input, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=56),
    FunctionInfo(name='tf.math.cumsum(x, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=44),
    FunctionInfo(name='tf.math.cumsum(x, axis, exclusive=True)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=48),
    FunctionInfo(name='tf.math.divide_no_nan(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_FLOAT_BROADCASTABLE_2,
                 weight=52),
    FunctionInfo(name='tf.math.floor(x)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=44),
    FunctionInfo(name='tf.math.log(x)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=52),
    FunctionInfo(name='tf.math.negative(x)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=48),
    FunctionInfo(name='tf.math.reciprocal(x)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=52),
    FunctionInfo(name='tf.math.reciprocal_no_nan(x)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=60),
    FunctionInfo(name='tf.math.segment_max(data, segment_ids)',
                 filter_group=FilterGroup.SEGMENT_OPERATION_2,
                 weight=40),
    FunctionInfo(name='tf.math.segment_mean(data, segment_ids)',
                 filter_group=FilterGroup.SEGMENT_OPERATION_2,
                 weight=56),
    FunctionInfo(name='tf.math.segment_min(data, segment_ids)',
                 filter_group=FilterGroup.SEGMENT_OPERATION_2,
                 weight=48),
    FunctionInfo(name='tf.math.segment_prod(data, segment_ids)',
                 filter_group=FilterGroup.SEGMENT_OPERATION_2,
                 weight=60),
    FunctionInfo(name='tf.math.segment_sum(data, segment_ids)',
                 filter_group=FilterGroup.SEGMENT_OPERATION_2,
                 weight=40),
    FunctionInfo(name='tf.math.squared_difference(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=52),
    FunctionInfo(name='tf.math.top_k(input, k)',
                 filter_group=FilterGroup.TOP_K_2,
                 weight=48),
    FunctionInfo(name=('tf.math.unsorted_segment_max(data, segment_ids, '
                       'num_segments)'),
                 filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
                 weight=40),
    FunctionInfo(name=('tf.math.unsorted_segment_mean(data, segment_ids, '
                       'num_segments)'),
                 filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
                 weight=56),
    FunctionInfo(name=('tf.math.unsorted_segment_min(data, segment_ids, '
                       'num_segments)'),
                 filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
                 weight=48),
    FunctionInfo(name=('tf.math.unsorted_segment_prod(data, segment_ids, '
                       'num_segments)'),
                 filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
                 weight=60),
    FunctionInfo(name=('tf.math.unsorted_segment_sum(data, segment_ids, '
                       'num_segments)'),
                 filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
                 weight=40),
    FunctionInfo(name='tf.matmul(a, b)',
                 filter_group=FilterGroup.MATMUL_2,
                 weight=24),
    FunctionInfo(name='tf.maximum(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=24),
    FunctionInfo(name='tf.minimum(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=32),
    FunctionInfo(name='tf.multiply(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=24),
    FunctionInfo(name='tf.not_equal(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=44),
    FunctionInfo(name='tf.one_hot(indices, depth)',
                 filter_group=FilterGroup.ONE_HOT_2,
                 weight=28),
    FunctionInfo(name='tf.ones(shape)',
                 filter_group=FilterGroup.SHAPE_1,
                 weight=44),
    FunctionInfo(name='tf.ones_like(input)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=36),
    FunctionInfo(name="tf.pad(tensor, paddings, mode='CONSTANT')",
                 filter_group=FilterGroup.PAD_2,
                 weight=40),
    FunctionInfo(name=("tf.pad(tensor, paddings, mode='CONSTANT', "
                       "constant_values)"),
                 filter_group=FilterGroup.PAD_3,
                 weight=52),
    FunctionInfo(name="tf.pad(tensor, paddings, mode='REFLECT')",
                 filter_group=FilterGroup.PAD_2,
                 weight=60),
    FunctionInfo(name="tf.pad(tensor, paddings, mode='SYMMETRIC')",
                 filter_group=FilterGroup.PAD_2,
                 weight=60),
    # This behaves like tf.range(limit), but the function signature has `start`
    # as the first argument, and it's not a keyword argument. Fortunately, when
    # we print expressions using this function, it'll be like 'tf.range(1)', and
    # not 'tf.range(start=1)' which would be technically correct but confusing.
    FunctionInfo(name='tf.range(start)',
                 filter_group=FilterGroup.RANGE_1,
                 weight=28),
    FunctionInfo(name='tf.range(start, limit, delta)',
                 filter_group=FilterGroup.RANGE_3,
                 weight=56),
    FunctionInfo(name='tf.reduce_any(input_tensor, axis)',
                 filter_group=FilterGroup.BOOLTENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name='tf.reduce_max(input_tensor)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=24),
    FunctionInfo(name='tf.reduce_max(input_tensor, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=24),
    FunctionInfo(name='tf.reduce_mean(input_tensor)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=40),
    FunctionInfo(name='tf.reduce_mean(input_tensor, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name='tf.reduce_min(input_tensor)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=40),
    FunctionInfo(name='tf.reduce_min(input_tensor, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name='tf.reduce_prod(input_tensor, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=52),
    FunctionInfo(name='tf.reduce_sum(input_tensor)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=24),
    FunctionInfo(name='tf.reduce_sum(input_tensor, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=24),
    FunctionInfo(name='tf.reshape(tensor, shape)',
                 filter_group=FilterGroup.TENSOR_SHAPE_2,
                 weight=28),
    FunctionInfo(name='tf.reverse(tensor, axis)',
                 filter_group=FilterGroup.TENSOR_AXISSEQUENCE_2,
                 weight=48),
    FunctionInfo(name='tf.roll(input, shift, axis)',
                 filter_group=FilterGroup.ROLL_3,
                 weight=48),
    FunctionInfo(name='tf.round(x)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=52),
    FunctionInfo(name='tf.scatter_nd(indices, updates, shape)',
                 filter_group=FilterGroup.SCATTER_ND_3,
                 weight=52),
    FunctionInfo(name="tf.searchsorted(sorted_sequence, values, side='left')",
                 filter_group=FilterGroup.SEARCHSORTED_2,
                 weight=56),
    FunctionInfo(name="tf.searchsorted(sorted_sequence, values, side='right')",
                 filter_group=FilterGroup.SEARCHSORTED_2,
                 weight=56),
    FunctionInfo(name='tf.sequence_mask(lengths)',
                 filter_group=FilterGroup.SEQUENCE_MASK_1,
                 weight=32),
    FunctionInfo(name='tf.sequence_mask(lengths, maxlen)',
                 filter_group=FilterGroup.SEQUENCE_MASK_2,
                 weight=44),
    FunctionInfo(name='tf.shape(input)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=36),
    FunctionInfo(name='tf.sign(x)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=48),
    FunctionInfo(name='tf.sort(values, axis)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=52),
    FunctionInfo(name="tf.sort(values, axis, direction='DESCENDING')",
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=60),
    FunctionInfo(name='tf.sqrt(x)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=56),
    FunctionInfo(name='tf.square(x)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=28),
    FunctionInfo(name='tf.squeeze(input)',
                 filter_group=FilterGroup.SQUEEZE_1,
                 weight=24),
    FunctionInfo(name='tf.squeeze(input, axis)',
                 filter_group=FilterGroup.SQUEEZE_2,
                 weight=23),  # Less weight than tf.reduce_max(input, axis).
    FunctionInfo(name='tf.stack(values, axis)',
                 filter_group=FilterGroup.TENSORSEQUENCE_AXIS_2,
                 weight=36),
    FunctionInfo(name='tf.subtract(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=28),
    FunctionInfo(name='tf.tensor_scatter_nd_update(tensor, indices, updates)',
                 filter_group=FilterGroup.TENSOR_SCATTER_ND_UPDATE_3,
                 weight=44),
    FunctionInfo(name='tf.tensordot(a, b, axes)',
                 filter_group=FilterGroup.TENSORDOT_3,
                 weight=24),
    FunctionInfo(name='tf.tile(input, multiples)',
                 filter_group=FilterGroup.TILE_2,
                 weight=28),
    FunctionInfo(name='tf.transpose(a)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=24),
    FunctionInfo(name='tf.transpose(a, perm)',
                 filter_group=FilterGroup.TRANSPOSE_2,
                 weight=44),
    FunctionInfo(name='tf.unique_with_counts(x)',
                 filter_group=FilterGroup.TENSOR_1D_1,
                 weight=48),
    FunctionInfo(name='tf.unstack(value, axis)',
                 filter_group=FilterGroup.TENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name='tf.where(condition)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=24),
    FunctionInfo(name='tf.where(condition, x, y)',
                 filter_group=FilterGroup.WHERE_3,
                 weight=24),
    FunctionInfo(name='tf.zeros(shape)',
                 filter_group=FilterGroup.SHAPE_1,
                 weight=40),
    FunctionInfo(name='tf.zeros_like(input)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=32),
]

# Operations for manipulating SparseTensors.
SPARSE_FUNCTIONS = [
    FunctionInfo(name='tf.SparseTensor(indices, values, dense_shape)',
                 filter_group=FilterGroup.SPARSETENSOR_3,
                 weight=20),
    FunctionInfo(name='tf.sparse.add(a, b)',
                 filter_group=FilterGroup.SAME_SHAPE_ONE_SPARSE_2,
                 weight=32),
    FunctionInfo(name='tf.sparse.concat(axis, sp_inputs)',
                 filter_group=FilterGroup.AXIS_SPARSESEQUENCE_2,
                 weight=40),
    FunctionInfo(name='tf.sparse.expand_dims(sp_input, axis)',
                 filter_group=FilterGroup.SPARSE_AXIS_2,
                 weight=24),
    FunctionInfo(name='tf.sparse.from_dense(tensor)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=20),
    FunctionInfo(name='tf.sparse.maximum(sp_a, sp_b)',
                 filter_group=FilterGroup.SAME_SHAPE_BOTH_SPARSE_2,
                 weight=32),
    FunctionInfo(name='tf.sparse.minimum(sp_a, sp_b)',
                 filter_group=FilterGroup.SAME_SHAPE_BOTH_SPARSE_2,
                 weight=40),
    FunctionInfo(name='tf.sparse.reduce_max(sp_input, axis, output_is_sparse)',
                 filter_group=FilterGroup.SPARSE_AXIS_BOOL_3,
                 weight=28),
    FunctionInfo(name='tf.sparse.reduce_sum(sp_input, axis, output_is_sparse)',
                 filter_group=FilterGroup.SPARSE_AXIS_BOOL_3,
                 weight=28),
    FunctionInfo(name='tf.sparse.reset_shape(sp_input)',
                 filter_group=FilterGroup.SPARSE_1,
                 weight=40),
    FunctionInfo(name='tf.sparse.reshape(sp_input, shape)',
                 filter_group=FilterGroup.SPARSE_SHAPE_2,
                 weight=40),
    FunctionInfo(name='tf.sparse.retain(sp_input, to_retain)',
                 filter_group=FilterGroup.SPARSE_RETAIN_2,
                 weight=36),
    FunctionInfo(name='tf.sparse.slice(sp_input, start, size)',
                 filter_group=FilterGroup.SPARSE_SLICE_3,
                 weight=32),
    FunctionInfo(name='tf.sparse.split(sp_input, num_split, axis)',
                 filter_group=FilterGroup.SPARSE_INT_AXIS_3,
                 weight=32),
    FunctionInfo(name='tf.sparse.to_dense(sp_input)',
                 filter_group=FilterGroup.SPARSE_1,
                 weight=20),
    FunctionInfo(name='tf.sparse.to_dense(sp_input, default_value)',
                 filter_group=FilterGroup.SPARSE_PRIMITIVE_2,
                 weight=36),
    FunctionInfo(name='tf.sparse.to_indicator(sp_input, vocab_size)',
                 filter_group=FilterGroup.SPARSE_TO_INDICATOR_2,
                 weight=44),
    FunctionInfo(name='tf.sparse.transpose(sp_input)',
                 filter_group=FilterGroup.SPARSE_1,
                 weight=36),
    FunctionInfo(name='tf.sparse.transpose(sp_input, perm)',
                 filter_group=FilterGroup.SPARSE_TRANSPOSE_2,
                 weight=56),
]

# A list of operation names that require filtering for value search to work at
# all, i.e., avoid segfaults and huge memory usage. Only relevant for PLDI paper
# experiments that turn off filtering for operations not listed here.
REQUIRES_FILTERING = [
    # Potentially huge memory usage.
    'tf.broadcast_to(input, shape)',
    'tf.eye(num_rows)',
    'tf.eye(num_rows, num_columns)',
    'tf.eye(num_rows, dtype)',
    'tf.fill(dims, value)',
    'tf.math.bincount(arr)',
    'tf.math.unsorted_segment_max(data, segment_ids, num_segments)',
    'tf.math.unsorted_segment_mean(data, segment_ids, num_segments)',
    'tf.math.unsorted_segment_min(data, segment_ids, num_segments)',
    'tf.math.unsorted_segment_prod(data, segment_ids, num_segments)',
    'tf.math.unsorted_segment_sum(data, segment_ids, num_segments)',
    'tf.one_hot(indices, depth)',
    'tf.ones(shape)',
    "tf.pad(tensor, paddings, mode='CONSTANT')",
    "tf.pad(tensor, paddings, mode='CONSTANT', constant_values)",
    "tf.pad(tensor, paddings, mode='REFLECT')",
    "tf.pad(tensor, paddings, mode='SYMMETRIC')",
    'tf.range(start)',
    'tf.range(start, limit, delta)',
    'tf.sequence_mask(lengths)',
    'tf.sequence_mask(lengths, maxlen)',
    'tf.tile(input, multiples)',
    'tf.zeros(shape)',

    # Avoid segfaults.
    'tf.cast(x, dtype)',
    'tf.constant(value, dtype)',
    'tf.gather(params, indices, axis, batch_dims)',
    'tf.math.segment_max(data, segment_ids)',
    'tf.math.segment_mean(data, segment_ids)',
    'tf.math.segment_min(data, segment_ids)',
    'tf.math.segment_prod(data, segment_ids)',
    'tf.math.segment_sum(data, segment_ids)',
    'tf.reshape(tensor, shape)',
    'tf.squeeze(input, axis)',
    'tf.sparse.to_indicator(sp_input, vocab_size)',
    'tf.sparse.reshape(sp_input, shape)',
    'tf.sparse.slice(sp_input, start, size)',
    'tf.sparse.split(sp_input, num_split, axis)',
]


def parse_function_info_name(function_info):
  """Takes a FunctionInfo and returns (function_name, list_of_args).

  Args:
    function_info: A FunctionInfo namedtuple.

  Returns:
    A tuple (function_name, list_of_args, constant_kwargs), where function_name
    is a string, list_of_args is a list of strings, and constant_kwargs is a
    dict mapping argument names to their constant literal values. For example,
    if the FunctionInfo's name is 'tf.foo.bar(x, axis, baz=True)', then
    this function would return ('tf.foo.bar', ['x', 'axis'], {'baz': True}).

  Raises:
    ValueError: If the FunctionInfo's name is not properly formatted.
  """
  name = function_info.name

  if name.count('(') != 1:
    raise ValueError("The FunctionInfo's name must have exactly one open "
                     "parenthesis.")
  if name.count(')') != 1 or name[-1] != ')':
    raise ValueError("The FunctionInfo's name must have exactly one close "
                     "parenthesis, at the end of the name.")

  open_paren = name.index('(')
  close_paren = name.index(')')
  function_name = name[ : open_paren]
  arg_list = name[open_paren + 1 : close_paren]
  split_by_comma = [arg.strip() for arg in arg_list.split(',')]
  list_of_args = []
  constant_kwargs = collections.OrderedDict()
  for part in split_by_comma:
    if '=' in part:
      kwarg_name, literal_as_string = [x.strip() for x in part.split('=')]
      constant_kwargs[kwarg_name] = ast.literal_eval(literal_as_string)
    else:
      list_of_args.append(part)
  return function_name, list_of_args, constant_kwargs

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
"""Defines the FilterGroup enum.

Every function used in TF-Coder is associated with one such FilterGroup.
"""

import enum


# LINT.IfChange(FilterGroup)
@enum.unique
class FilterGroup(enum.Enum):
  """A group of similar operations that should have the same filters.

  The number of arguments is crucially important when adding filters, so by
  convention the enum names have the number of arguments at the end.
  """
  # No filters. Even if some filtering might be reasonable, it could be faster
  # to just try all values to avoid the filtering overhead.
  NONE = 'NONE'

  #############################
  # Operations with 1 argument.

  # The argument is a shape.
  SHAPE_1 = 'SHAPE_1'
  # The argument is a sequence.
  SEQUENCE_1 = 'SEQUENCE_1'
  # The argument is a tensor.
  TENSOR_1 = 'TENSOR_1'
  # The argument is a float tensor.
  FLOATTENSOR_1 = 'FLOATTENSOR_1'
  # The argument is an int or float tensor.
  NUMERICTENSOR_1 = 'NUMERICTENSOR_1'
  # The argument is a SparseTensor.
  SPARSE_1 = 'SPARSE_1'
  # The argument is not a tensor.
  NOT_TENSOR_1 = 'NOT_TENSOR_1'
  # The argument is a 1-D tensor.
  TENSOR_1D_1 = 'TENSOR_1D_1'
  # The argument is a primitive or tensor.
  PRIMITIVE_OR_TENSOR_1 = 'PRIMITIVE_OR_TENSOR_1'

  ################################
  # Operations with 2 arguments.

  # The first argument is a tensor, and the second argument is an int
  # representing an axis, i.e., an int in the range [1, rank_of_tensor).
  TENSOR_AXIS_2 = 'TENSOR_AXIS_2'
  # The first argument is a tensor, and the second argument is an axis sequence.
  TENSOR_AXISSEQUENCE_2 = 'TENSOR_AXISSEQUENCE_2'
  # The first argument is a boolean tensor, the second is an axis.
  BOOLTENSOR_AXIS_2 = 'BOOLTENSOR_AXIS_2'
  # The first argument is an int or float tensor, the second is an axis.
  NUMERICTENSOR_AXIS_2 = 'NUMERICTENSOR_AXIS_2'
  # The first argument is a sequence of tensors, the second is an axis.
  TENSORSEQUENCE_AXIS_2 = 'TENSORSEQUENCE_AXIS_2'
  # The first argument is a tensor, the second is a shape.
  TENSOR_SHAPE_2 = 'TENSOR_SHAPE_2'
  # The first argument is a shape, the second is a primitive.
  SHAPE_PRIMITIVE_2 = 'SHAPE_PRIMITIVE_2'
  # The first argument is a tensor, the second is a boolean tensor.
  TENSOR_BOOLTENSOR_2 = 'TENSOR_BOOLTENSOR_2'
  # The second argument is a dtype. If the dtype is integral, the first argument
  # must be safely castable to that dtype.
  CASTABLE_DTYPE_2 = 'CASTABLE_DTYPE_2'
  # The two arguments are numeric (int or float) tensors with the same dtype,
  # and the two tensors are broadcastable, or one of the arguments is an int or
  # float primitive.
  SAME_DTYPE_NUMERIC_BROADCASTABLE_2 = 'SAME_DTYPE_NUMERIC_BROADCASTABLE_2'
  # The two arguments are float tensors with the same dtype and the two tensors
  # are broadcastable, or one of the arguments is a float primitive.
  SAME_DTYPE_FLOAT_BROADCASTABLE_2 = 'SAME_DTYPE_FLOAT_BROADCASTABLE_2'
  # Of the two arguments, one must be a SparseTensor, and the other must be a
  # Tensor or SparseTensor. The two arguments must have the same shape.
  SAME_SHAPE_ONE_SPARSE_2 = 'SAME_SHAPE_ONE_SPARSE_2'
  # The two arguments arguments must be SparseTensors with the same shape.
  SAME_SHAPE_BOTH_SPARSE_2 = 'SAME_SHAPE_BOTH_SPARSE_2'
  # The first argument is an int representing an axis, the second is a list of
  # SparseTensor.
  AXIS_SPARSESEQUENCE_2 = 'AXIS_SPARSESEQUENCE_2'
  # The first argument is a SparseTensor, the second is an axis.
  SPARSE_AXIS_2 = 'SPARSE_AXIS_2'
  # The first argument is a SparseTensor, the second is a shape.
  SPARSE_SHAPE_2 = 'SPARSE_SHAPE_2'
  # The first argument is a SparseTensor, the second is a primitive.
  SPARSE_PRIMITIVE_2 = 'SPARSE_PRIMITIVE_2'
  # The first argument is an int or float tensor, and the second argument is a
  # 1D int tensor with nonnegative nondecreasing entries, and the two arguments
  # have equal lengths in the first dimension.
  SEGMENT_OPERATION_2 = 'SEGMENT_OPERATION_2'

  ##############################
  # Operations with 3 arguments.

  # The first argument is a SparseTensor, the second is an int, and the third is
  # an axis.
  SPARSE_INT_AXIS_3 = 'SPARSE_INT_AXIS_3'
  # The first argument is a SparseTensor, the second is an axis, the third is a
  # boolean.
  SPARSE_AXIS_BOOL_3 = 'SPARSE_AXIS_BOOL_3'
  # The first argument is an int or float tensor, the second argument is an int
  # tensor with nonnegative entries and with a shape that is a prefix of the
  # first argument's shape, and the third argument is an integer greater than
  # the maximum element of the second argument.
  UNSORTED_SEGMENT_OPERATION_3 = 'UNSORTED_SEGMENT_OPERATION_3'

  #########################################
  # Operations with other special handling.

  # The argument contains nonnegative ints with a small maximum.
  BINCOUNT_1 = 'BINCOUNT_1'
  # The argument results in a small tensor.
  EYE_1 = 'EYE_1'
  # The argument results in a small tensor.
  RANGE_1 = 'RANGE_1'
  # The argument contains a few ints with a small maximum.
  SEQUENCE_MASK_1 = 'SEQUENCE_MASK_1'
  # The argument is a tensor with at least one squeezable dimension.
  SQUEEZE_1 = 'SQUEEZE_1'

  # The first argument is a tensor, the second is a compatible shape.
  BROADCAST_TO_2 = 'BROADCAST_TO_2'
  # The first argument is a tensor, the second is an axis in the range
  # [-1, rank_of_tensor]. Note that this range is slightly different from the
  # TENSOR_AXIS_2 filter.
  EXPAND_DIMS_2 = 'EXPAND_DIMS_2'
  # The arguments result in a small tensor.
  EYE_ROWS_COLS_2 = 'EYE_ROWS_COLS_2'
  # The arguments result in a small tensor.
  EYE_ROWS_DTYPE_2 = 'EYE_ROWS_DTYPE_2'
  # The first argument is a tensor, the second is a tensor containing ints
  # suitable for indexing into the first tensor on axis 0.
  GATHER_2 = 'GATHER_2'
  # The first argument is a tensor, the second is a tensor containing ints
  # suitable for indexing into the first tensor on multiple dimensions.
  GATHER_ND_2 = 'GATHER_ND_2'
  # Ensures the tensors are both numeric and have the same dtype and rank.
  MATMUL_2 = 'MATMUL_2'
  # Ensures that tf.one_hot(indices, depth) produces a small result.
  ONE_HOT_2 = 'ONE_HOT_2'
  # The first argument must be a tensor, and the second must be a nested int
  # list or int32 tensor of shape [rank_of_arg_1, 2].
  PAD_2 = 'PAD_2'
  # Ensures that tf.tile(input, multiples) produces a small result.
  TILE_2 = 'TILE_2'
  # The first argument is a numeric tensor, the second is a compatible int
  # primitive or tensor.
  TOP_K_2 = 'TOP_K_2'
  # The first argument is a tensor, the second is 1D (ints) with correct length.
  TRANSPOSE_2 = 'TRANSPOSE_2'
  # The first argument is sorted in the last dimension, the second argument is
  # the same dtype and rank, and all dimension lengths match except the last.
  SEARCHSORTED_2 = 'SEARCHSORTED_2'
  # The first argument contains a few ints with a small maximum, and the second
  # is an int primitive or tensor with a value in range.
  SEQUENCE_MASK_2 = 'SEQUENCE_MASK_2'
  # Ensures that tf.sparse.retain(sp_input, to_retain) has arguments that match,
  # i.e., to_retain is 1D, contains booleans, and has the right length.
  SPARSE_RETAIN_2 = 'SPARSE_RETAIN_2'
  # Ensures that tf.sparse.to_indicator(sp_input, vocab_size) produces a small
  # result.
  SPARSE_TO_INDICATOR_2 = 'SPARSE_TO_INDICATOR_2'
  # The first argument is a SparseTensor, the second is 1D (ints) with correct
  # length.
  SPARSE_TRANSPOSE_2 = 'SPARSE_TRANSPOSE_2'
  # The first argument is a tensor with more than 1 squeezable dimension, and
  # the second argument is an int specifying a squeezable dimension.
  SQUEEZE_2 = 'SQUEEZE_2'

  # The first argument must be an int or float tensor, and the second and third
  # arguments must both be nondecreasing primitives.
  CLIP_BY_VALUE_3 = 'CLIP_BY_VALUE_3'
  # The first argument is a tensor, the second is a tensor containing ints
  # suitable for indexing into the first tensor on multiple dimensions, and the
  # third is a number of batch dimensions.
  GATHER_ND_3 = 'GATHER_ND_3'
  # The first argument must be a tensor, the second must be a nested int list or
  # int32 tensor of shape [rank_of_arg_1, 2], and the third must be a scalar of
  # the same type as the first argument.
  PAD_3 = 'PAD_3'
  # The arguments result in a small tensor.
  RANGE_3 = 'RANGE_3'
  # The second and third arguments must be int primitives, lists of ints, or 1D
  # int tensors, and they must have the same shape.
  ROLL_3 = 'ROLL_3'
  # In tf.scatter_nd(indices, updates, shape), `indices` must be an int tensor
  # representing valid indices, `shape` must be a shape tuple, and `updates`
  # must have a compatible shape with `indices` and `shape`.
  SCATTER_ND_3 = 'SCATTER_ND_3'
  # In tf.sparse.slice(sp_input, start, size), the arguments start and size must
  # be 1D (ints) with the same length as sp_input.dense_shape.
  SPARSE_SLICE_3 = 'SPARSE_SLICE_3'
  # The arguments must have the right shape to make a SparseTensor.
  SPARSETENSOR_3 = 'SPARSETENSOR_3'
  # In tf.tensor_scatter_nd_update(tensor, indices, updates), `indices` must be
  # an int tensor representing valid indices, and `updates` must have a
  # compatible shape and be the same dtype as `tensor`.
  TENSOR_SCATTER_ND_UPDATE_3 = 'TENSOR_SCATTER_ND_UPDATE_3'
  # The first two arguments are tensors with the same dtype, and the third
  # contains ints of the appropriate shape.
  TENSORDOT_3 = 'TENSORDOT_3'
  # Ensures that the shapes and dtypes for tf.where(condition, x, y) match.
  WHERE_3 = 'WHERE_3'

  # Ensures that params, indices, axis, and batch_dims are all compatible
  # according to the tf.gather documentation.
  GATHER_4 = 'GATHER_4'

# LINT.ThenChange(value_search/operation_filtering.py:add_filters_to_function_operation)

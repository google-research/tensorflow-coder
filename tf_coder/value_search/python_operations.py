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
"""Defines Operation objects for Python operators."""

import abc

import six
from tf_coder import tensor_limits as limits
from tf_coder import tf_coder_utils
from tf_coder.value_search import operation_base
from tf_coder.value_search import operation_filtering as filtering
from tf_coder.value_search import value

# Weights for Python operations.
INDEXING_WEIGHT = 32
SLICING_WEIGHT = 36
SINGLETON_TUPLE_CREATION_WEIGHT = 20
PAIR_CREATION_WEIGHT = 20
TRIPLE_CREATION_WEIGHT = 36

# "Docstrings" for Python operations, so they can used for ranking in the same
# way as for TensorFlow operations.

INDEXING_DOCSTRING = """
Indexes axis 0 of a tensor or list using standard Python bracket notation.

Selects an element from the tensor or list according to the index. The index
applies to the first dimension (axis 0). Negative indices are supported,
counting backwards from the end.
"""

INDEXING_AXIS_1_DOCSTRING = """
Indexes axis 1 of a tensor using standard Python bracket notation.

Selects a slice of the tensor according to the index. The index applies to the
second dimension (axis 1). Negative indices are supported, counting backwards
from the end.
"""

SINGLETON_TUPLE_CREATION_DOCSTRING = """
Creates a single-element tuple.

This may be used to create a shape for tensors.
"""

PAIR_CREATION_DOCSTRING = """
Creates a pair (2-tuple).

This may be used to pack multiple tensors into a sequence, or create a shape for
tensors.
"""

TRIPLE_CREATION_DOCSTRING = """
Creates a triple (3-tuple).

This may be used to pack multiple tensors into a sequence, or create a shape for
tensors.
"""

# This is used as a format string for all of the slicing variations.
# `slice_kind` should be 'prefix', 'suffix', or 'range'.
# `axis` should be an integer axis number.
SLICING_DOCSTRING_FORMAT = """
Slices a tensor along one dimension using standard Python notation.

Selects a {slice_kind} of indices along axis {axis} of the tensor.
"""


class IndexingOperation(operation_base.Operation):
  """An indexing operation of the form "arg0[arg1]"."""

  def __init__(self):
    metadata = operation_base.OperationMetadata(docstring=INDEXING_DOCSTRING)
    super(IndexingOperation, self).__init__(
        num_args=2, weight=INDEXING_WEIGHT, metadata=metadata)

    def _sequence_or_tensor(arg_value):
      return (arg_value.is_sequence or
              (arg_value.is_tensor and len(arg_value.shape)))
    self.add_value_filters([_sequence_or_tensor,
                            filtering.INT_OR_INT_TENSOR_FILTER])
    def _check_index(arg_values):
      """Checks that the index is in range for the sequence or tensor."""
      indexable, index = arg_values
      if indexable.is_sequence:
        length = len(indexable.value)
      else:  # indexable is a tensor with at least 1 dimension.
        length = indexable.shape[0]
      return -length <= int(index.value) < length
    self.set_apply_filter(_check_index)

  def apply(self, arg_values, settings):
    """See base class."""
    try:
      return value.OperationValue(arg_values[0].value[int(arg_values[1].value)],
                                  self, arg_values)
    except Exception:  # pylint: disable=broad-except
      return None

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return arg_strings[0] + '[' + arg_strings[1] + ']'


class IndexingAxis1Operation(operation_base.Operation):
  """An indexing operation of the form "arg0[:, arg1]"."""

  def __init__(self):
    metadata = operation_base.OperationMetadata(
        docstring=INDEXING_AXIS_1_DOCSTRING)
    super(IndexingAxis1Operation, self).__init__(
        num_args=2, weight=INDEXING_WEIGHT, metadata=metadata)

    self.add_value_filters([filtering.get_tensor_min_rank_filter(2),
                            filtering.INT_OR_INT_TENSOR_FILTER])
    def _check_index(arg_values):
      """Checks that the index is in range for the tensor."""
      tensor, index = arg_values
      length = tensor.shape[1]
      return -length <= int(index.value) < length
    self.set_apply_filter(_check_index)

  def apply(self, arg_values, settings):
    """See base class."""
    try:
      return value.OperationValue(
          arg_values[0].value[:, int(arg_values[1].value)], self, arg_values)
    except Exception:  # pylint: disable=broad-except
      return None

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return arg_strings[0] + '[:, ' + arg_strings[1] + ']'


class SingletonTupleCreationOperation(operation_base.Operation):
  """An operation of the form "(arg0,)".

  The argument must be a primitive or sequence. This should not be used to
  create a single-element sequence of tensors, since ops requiring a list of
  tensors are only meaningful if there are multiple tensors in the list.
  """

  def __init__(self):
    metadata = operation_base.OperationMetadata(
        docstring=SINGLETON_TUPLE_CREATION_DOCSTRING)
    super(SingletonTupleCreationOperation, self).__init__(
        num_args=1, weight=SINGLETON_TUPLE_CREATION_WEIGHT, metadata=metadata)

    def _primitives_or_sequences_filter(arg_value):
      """The result must be a possibly-nested list of primitives."""
      return (arg_value.is_primitive or
              filtering.TENSOR_LIKE_SEQUENCE_FILTER(arg_value))
    self.add_value_filters([_primitives_or_sequences_filter])

  def apply(self, arg_values, settings):
    """See base class."""
    try:
      return value.OperationValue((arg_values[0].value,), self, arg_values)
    except ValueError:
      return None

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return '({},)'.format(arg_strings[0])


class PairCreationOperation(operation_base.Operation):
  """An operation of the form "(arg0, arg1)".

  The arguments must be primitives of the same type, or tensors of the same
  dtype and similar shape, or sequences with the same sequence_dtype and
  sequence_shape.
  """

  def __init__(self):
    metadata = operation_base.OperationMetadata(
        docstring=PAIR_CREATION_DOCSTRING)
    super(PairCreationOperation, self).__init__(
        num_args=2, weight=PAIR_CREATION_WEIGHT, metadata=metadata)

    for primitive_type in tf_coder_utils.PRIMITIVE_TYPES:
      self.add_value_filters(
          [filtering.get_type_filter(primitive_type)] * 2)

    def _tensor_value_filter(arg_value):
      """Only keeps values that are "small" tensors."""
      return (arg_value.is_tensor and
              arg_value.num_elements() * 2 <= limits.MAX_TENSOR_ELEMENTS)
    self.add_value_filters([_tensor_value_filter] * 2)

    self.add_value_filters([filtering.TENSOR_LIKE_SEQUENCE_FILTER] * 2)

    def _apply_filter(arg_values):
      """Ensures dtype and shape compatibility."""
      first, second = arg_values
      if first.is_tensor:  # Implies second is also a tensor.
        if first.dtype != second.dtype:
          return False
        shape_1 = first.shape
        shape_2 = second.shape
        if shape_1 == shape_2:
          return True
        if len(shape_1) != len(shape_2):
          return False
        num_different = sum(len_1 != len_2
                            for len_1, len_2 in zip(shape_1, shape_2))
        return num_different <= 1
      elif first.is_sequence:  # Implies second is also a sequence.
        return (first.sequence_dtype == second.sequence_dtype and
                first.sequence_shape == second.sequence_shape)
      else:  # Implies first and second are same-type primitives.
        return True
    self.set_apply_filter(_apply_filter)

  def apply(self, arg_values, settings):
    """See base class."""
    try:
      return value.OperationValue((arg_values[0].value, arg_values[1].value),
                                  self, arg_values)
    except ValueError:
      return None

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return '(' + ', '.join(arg_strings) + ')'


class TripleCreationOperation(operation_base.Operation):
  """An operation of the form "(arg0, arg1, arg2)".

  The arguments must be primitives of the same type, or tensors of the same
  dtype and similar shape, or sequences with the same sequence_dtype and
  sequence_shape.
  """

  def __init__(self):
    metadata = operation_base.OperationMetadata(
        docstring=TRIPLE_CREATION_DOCSTRING)
    super(TripleCreationOperation, self).__init__(
        num_args=3, weight=TRIPLE_CREATION_WEIGHT, metadata=metadata)

    for primitive_type in tf_coder_utils.PRIMITIVE_TYPES:
      self.add_value_filters(
          [filtering.get_type_filter(primitive_type)] * 3)

    def _tensor_value_filter(arg_value):
      """Only keeps values that are "small" tensors."""
      return (arg_value.is_tensor and
              arg_value.num_elements() * 3 <= limits.MAX_TENSOR_ELEMENTS)
    self.add_value_filters([_tensor_value_filter] * 3)

    self.add_value_filters([filtering.TENSOR_LIKE_SEQUENCE_FILTER] * 3)

    def _apply_filter(arg_values):
      """Ensures dtype and shape compatibility."""
      first, second, third = arg_values
      if first.is_tensor:  # Implies second and third are also tensors.
        if not first.dtype == second.dtype == third.dtype:
          return False
        shape_1 = first.shape
        shape_2 = second.shape
        shape_3 = third.shape
        if not len(shape_1) == len(shape_2) == len(shape_3):
          return False
        num_different = sum(
            not len_1 == len_2 == len_3
            for len_1, len_2, len_3 in zip(shape_1, shape_2, shape_3))
        return num_different <= 1
      elif first.is_sequence:  # Implies second and third is also sequences.
        return (first.sequence_dtype == second.sequence_dtype
                == third.sequence_dtype and
                first.sequence_shape == second.sequence_shape
                == third.sequence_shape)
      else:  # Implies that all args are same-type primitives.
        return True

    self.set_apply_filter(_apply_filter)

  def apply(self, arg_values, settings):
    """See base class."""
    try:
      return value.OperationValue(
          (arg_values[0].value, arg_values[1].value, arg_values[2].value),
          self, arg_values)
    except ValueError:
      return None

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return '(' + ', '.join(arg_strings) + ')'


@six.add_metaclass(abc.ABCMeta)
class SlicingBaseOperation(operation_base.Operation):
  """Common functionality for slicing operations."""

  def __init__(self):
    num_args = self._num_args()
    min_rank = self._min_rank()
    docstring = self._get_docstring()
    metadata = operation_base.OperationMetadata(docstring=docstring)
    super(SlicingBaseOperation, self).__init__(
        num_args=num_args, weight=SLICING_WEIGHT, metadata=metadata)
    self.add_value_filters(
        [filtering.get_tensor_min_rank_filter(min_rank)] +
        [filtering.INT_OR_INT_TENSOR_FILTER] * (num_args - 1))
    self.set_apply_filter(self._get_apply_filter())

  @abc.abstractmethod
  def _num_args(self):
    """Returns the number of arguments used by this operation."""

  @abc.abstractmethod
  def _min_rank(self):
    """Returns the minimum allowable tensor rank for the first argument."""

  @abc.abstractmethod
  def _get_apply_filter(self):
    """Returns an apply filter."""

  @abc.abstractmethod
  def _get_docstring(self):
    """Returns a docstring for this operation."""

  @abc.abstractmethod
  def _evaluate_slice(self, raw_values):
    """Evaluates the slice, e.g., `raw_values[0][raw_values[1] : ]`."""

  def apply(self, arg_values, settings):
    """See base class."""
    try:
      result = self._evaluate_slice(
          [arg_values[0].value] +
          [int(arg_value.value) for arg_value in arg_values[1:]])
      return value.OperationValue(result, self, arg_values)
    except Exception:  # pylint: disable=broad-except
      return None


class SlicingAxis0LeftOperation(SlicingBaseOperation):
  """An operation of the form "arg0[arg1:]"."""

  def _num_args(self):
    return 2

  def _min_rank(self):
    return 1

  def _get_apply_filter(self):
    def _check_index(arg_values):
      """Checks that the index is in range for the tensor."""
      tensor, index = arg_values
      length = tensor.shape[0]
      return -length <= int(index.value) < length
    return _check_index

  def _get_docstring(self):
    return SLICING_DOCSTRING_FORMAT.format(slice_kind='suffix', axis=0)

  def _evaluate_slice(self, raw_values):
    return raw_values[0][raw_values[1] : ]

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return arg_strings[0] + '[' + arg_strings[1] + ':]'


class SlicingAxis0RightOperation(SlicingBaseOperation):
  """An operation of the form "arg0[:arg1]"."""

  def _num_args(self):
    return 2

  def _min_rank(self):
    return 1

  def _get_apply_filter(self):
    def _check_index(arg_values):
      """Checks that the index is in range for the tensor."""
      tensor, index = arg_values
      length = tensor.shape[0]
      return -length <= int(index.value) < length
    return _check_index

  def _get_docstring(self):
    return SLICING_DOCSTRING_FORMAT.format(slice_kind='prefix', axis=0)

  def _evaluate_slice(self, raw_values):
    return raw_values[0][ : raw_values[1]]

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return arg_strings[0] + '[:' + arg_strings[1] + ']'


class SlicingAxis0BothOperation(SlicingBaseOperation):
  """An operation of the form "arg0[arg1:arg2]"."""

  def _num_args(self):
    return 3

  def _min_rank(self):
    return 1

  def _get_apply_filter(self):
    def _check_indices(arg_values):
      """Checks that the indices are in range for the tensor."""
      tensor, left_value, right_value = arg_values
      length = tensor.shape[0]
      left = int(left_value.value)
      right = int(right_value.value)
      if left < 0:
        left += length
      if right < 0:
        right += length
      return 0 <= left < right < length
    return _check_indices

  def _get_docstring(self):
    return SLICING_DOCSTRING_FORMAT.format(slice_kind='range', axis=0)

  def _evaluate_slice(self, raw_values):
    return raw_values[0][raw_values[1] : raw_values[2]]

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return arg_strings[0] + '[' + arg_strings[1] + ':' + arg_strings[2] + ']'


class SlicingAxis1LeftOperation(SlicingBaseOperation):
  """An operation of the form "arg0[:, arg1:]"."""

  def _num_args(self):
    return 2

  def _min_rank(self):
    return 2

  def _get_apply_filter(self):
    def _check_index(arg_values):
      """Checks that the index is in range for the tensor."""
      tensor, index = arg_values
      length = tensor.shape[1]
      return -length <= int(index.value) < length
    return _check_index

  def _get_docstring(self):
    return SLICING_DOCSTRING_FORMAT.format(slice_kind='suffix', axis=1)

  def _evaluate_slice(self, raw_values):
    return raw_values[0][:, raw_values[1] : ]

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return arg_strings[0] + '[:, ' + arg_strings[1] + ':]'


class SlicingAxis1RightOperation(SlicingBaseOperation):
  """An operation of the form "arg0[:, :arg1]"."""

  def _num_args(self):
    return 2

  def _min_rank(self):
    return 2

  def _get_apply_filter(self):
    def _check_index(arg_values):
      """Checks that the index is in range for the tensor."""
      tensor, index = arg_values
      length = tensor.shape[1]
      return -length <= int(index.value) < length
    return _check_index

  def _get_docstring(self):
    return SLICING_DOCSTRING_FORMAT.format(slice_kind='prefix', axis=1)

  def _evaluate_slice(self, raw_values):
    return raw_values[0][:, : raw_values[1]]

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return arg_strings[0] + '[:, :' + arg_strings[1] + ']'


class SlicingAxis1BothOperation(SlicingBaseOperation):
  """An operation of the form "arg0[:, arg1:arg2]"."""

  def _num_args(self):
    return 3

  def _min_rank(self):
    return 2

  def _get_apply_filter(self):
    def _check_indices(arg_values):
      """Checks that the indices are in range for the tensor."""
      tensor, left_value, right_value = arg_values
      length = tensor.shape[1]
      left = int(left_value.value)
      right = int(right_value.value)
      if left < 0:
        left += length
      if right < 0:
        right += length
      return 0 <= left < right < length
    return _check_indices

  def _get_docstring(self):
    return SLICING_DOCSTRING_FORMAT.format(slice_kind='range', axis=1)

  def _evaluate_slice(self, raw_values):
    return raw_values[0][:, raw_values[1] : raw_values[2]]

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return arg_strings[0] + '[:, ' + arg_strings[1] + ':' + arg_strings[2] + ']'

# TODO(kshi): Implement .shape attribute access. However, we already have a
# tf.shape() operation, so perhaps .shape access is not necessary.

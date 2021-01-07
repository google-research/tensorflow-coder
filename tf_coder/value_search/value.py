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
"""Defines the Value objects used in value search."""

import abc
import collections
import functools
import itertools
import operator
import six

import tensorflow as tf
from tf_coder import tf_coder_utils
from tf_coder.value_search import value_search_utils


def cache(fn):
  def new_fn(self):
    key = fn.__name__
    if key in self.cached_info:
      return self.cached_info[key]
    result = fn(self)
    self.cached_info[key] = result
    return result
  return new_fn


@six.add_metaclass(abc.ABCMeta)
class Value(object):
  """The Value class wraps a value (a Python object) used in value search.

  This class serves the following purposes:
    * During initialization, it computes and caches information about the
      wrapped value, making it easy to filter values based on their properties.
    * It stores information about how the wrapped value was created, so that all
      Value objects can compute their corresponding Python expressions.
    * It provides a consistent way of comparing wrapped values for equality.

  Once created, a Value is treated as immutable. This means that the Value
  should not be passed to operations that modify their arguments.

  If the wrapped value is a sequence, it must not be empty, its elements must
  all have the same type, and the elements must be primitives, tf.Tensors, or
  sequences. In the nested sequence case, we also require that it is tensor-like
  (i.e., calling tf.constant() succeeds without error).

  Attributes:
    value: The value (any eligible Python object) wrapped inside this Value.
    type: Exactly `type(value)`, as an attribute for convenience.

    is_primitive: True iff `type` is one of int, list, bool, or string.
    is_dtype: True iff `value` is an instance of tf.DType.
    is_sequence: True iff `value` is an instance of list or tuple.
    is_tensor: True iff `value` an instance of tf.Tensor.
    is_sparse_tensor: True iff `value` an instance of tf.SparseTensor.

    elem_type: If `value` is a sequence, this is the type of the sequence
      elements. Otherwise, this is None.
    elem_type_is_tensor: If `value` is a sequence, this is True iff the sequence
      elements are instances of tf.Tensor. Otherwise, this is False.
    elem_type_is_sparse_tensor: If `value` is a sequence, this is True iff the
      sequence elements are instances of tf.SparseTensor. Otherwise, this is
      False.
    sequence_dtype: If `value` is a sequence of primitives or a nested sequence,
      this is the dtype of the result of calling tf.constant(). Otherwise, this
      is None.
    sequence_shape: If `value` is a sequence of primitives or a nested sequence,
      this is the shape of the result of calling tf.constant(), as a list of
      ints. Otherwise, this is None.

    dtype: If `value` is a Tensor or SparseTensor, this is its dtype. Otherwise,
      this is None.
    shape: If `value` is a Tensor or SparseTensor, this is its shape (as a list
      of ints). Othwerwise, this is None.

    cached_info: A dict containing cached computations, such as the maximum
      element in a tensor.
  """

  def __init__(self, value):
    """Initializes a Value object to contain `value`.

    Args:
      value: The actual Python object that this Value represents.

    Raises:
      ValueError: If `value` is an empty sequence, or a sequence of different
        types of elements, or anything that is too large.
    """
    self.value = value
    self.type = type(value)

    self.is_primitive = self.type in tf_coder_utils.PRIMITIVE_TYPES
    self.is_dtype = isinstance(self.value, tf.DType)
    self.is_sequence = isinstance(self.value, (list, tuple))
    self.is_tensor = isinstance(value, tf.Tensor)
    self.is_sparse_tensor = isinstance(value, tf.SparseTensor)

    self.cached_info = {}

    # Sequence-related attributes and checks.
    # Sequences include lists, tuples, and namedtuples.
    self.elem_type = None
    self.elem_type_is_tensor = False
    self.elem_type_is_sparse_tensor = False
    self.sequence_dtype = None
    self.sequence_shape = None
    if self.is_sequence:
      # Sequences must be nonempty.
      # Do not assume empty sequences will implicitly convert to False in a
      # boolean context (e.g., TensorShape does not).
      if not len(value):  # pylint: disable=g-explicit-length-test
        raise ValueError('Sequences must be nonempty.')
      self.elem_type = type(value[0])

      # Sequences must contain elements of the same type.
      # Using isinstance() here is less correct because it is not symmetric,
      # i.e., `isinstance(value[i], type(value[0]))` can be different after
      # switching `i` and `0`. Instead, we force all elements to be exactly
      # the same type.
      if not all(type(elem) is self.elem_type for elem in value):  # pylint: disable=unidiomatic-typecheck
        raise ValueError('Sequences must contain elements of the same type.')

      self.elem_type_is_tensor = isinstance(value[0], tf.Tensor)
      self.elem_type_is_sparse_tensor = isinstance(value[0], tf.SparseTensor)

      # Sequences must contain tensors, SparseTensors, primitives, or other
      # sequences.
      if (not self.elem_type_is_tensor and
          not self.elem_type_is_sparse_tensor and
          self.elem_type not in tf_coder_utils.PRIMITIVE_TYPES and
          self.elem_type not in (list, tuple)):
        raise ValueError('Sequence must contain Tensors, SparseTensors, '
                         'primitives, or other sequences.')

      # The sequence must be tensor-like (i.e., tf.constant() succeeds), unless
      # the sequence contains Tensors or SparseTensors (which may have different
      # shapes or dtypes).
      if not self.elem_type_is_tensor and not self.elem_type_is_sparse_tensor:
        try:
          as_tensor = tf.constant(self.value)
          self.sequence_shape = as_tensor.shape.as_list()
          self.sequence_dtype = as_tensor.dtype
        except (ValueError, TypeError):
          raise ValueError('Sequence is not tensor-like.')
        if self.num_elements() == 0:
          raise ValueError('Sequence represents an empty tensor.')

    # Tensor-related attributes and checks.
    self.dtype = None
    self.shape = None
    if self.is_tensor:
      self.dtype = value.dtype
      self.shape = value.shape.as_list()
      if self.num_elements() == 0:
        raise ValueError('Tensor is empty.')
      if not value_search_utils.check_tensor_size(self):
        raise ValueError('Tensor value is too large.')

    # Sparse tensor checks.
    if self.is_sparse_tensor:
      self.dtype = value.dtype
      self.shape = value.shape.as_list()

      # Check that the indices are in bounds.
      # TODO(kshi): This test is expensive (~7% slowdown). Consider only doing
      # this test when necessary, e.g., when the user provides a SparseTensor,
      # or on the result of a tf.SparseTensor(...) operation.
      bounding_box = tf.sparse.reset_shape(value)
      try:
        tf.sparse.reset_shape(bounding_box, new_shape=value.shape)
      except ValueError:
        raise ValueError('SparseTensor has out-of-bounds index.')

      if not value_search_utils.check_sparse_tensor_size(self):
        raise ValueError('SparseTensor value is too large.')
      if self.num_elements() == 0:
        raise ValueError('SparseTensor is empty.')

    # At most one of is_primitive, is_dtype, is_sequence, is_tensor, and
    # is_sparse_tensor can be True, or there is a major bug.
    assert sum((self.is_primitive, self.is_dtype, self.is_sequence,
                self.is_tensor, self.is_sparse_tensor)) <= 1

    self._repr_cache = None

  def __repr__(self):
    """Returns a string representation of the value.

    Values are considered equal if and only if their string representations (as
    computed by this function) are equal.
    """
    if self._repr_cache is None:
      self._repr_cache = tf_coder_utils.object_to_string(self.value)
    return self._repr_cache

  def __hash__(self):
    """Implements hash so that Value objects can be used as dict keys."""
    return hash(repr(self))

  def __eq__(self, other):
    """Returns whether this Value object is equal to `other`.

    Args:
      other: The other object to compare to.

    Values are considered equal if and only if their string representations (as
    computed by __repr__) are equal.
    """
    if not isinstance(other, Value):
      return NotImplemented
    return repr(self) == repr(other)

  def __ne__(self, other):
    """Returns whether this Value object is not equal to `other`."""
    return not self == other

  def copy(self):
    """Returns a copy of this Value, recursing through OperationValues."""
    raise NotImplementedError('Calling copy() on unhandled Value subclass {}'
                              .format(type(self)))

  @abc.abstractmethod
  def reconstruct_expression(self, use_cache=True):
    """Returns a code expression (as a string) that creates this value.

    This can be slow and should not be called in a tight loop.

    Args:
      use_cache: If True, the reconstruction may be looked up from a cache. If
        False, the reconstruction will be recomputed on each call.
    """

  def reconstruct_expression_with_input_names(self):
    """Returns the code expression and a set of input names that were used.

    The input names can be used to determine if a solution uses any or all of
    the inputs.

    This implementation assumes no inputs are used. Subclasses should override
    this method if they can use inputs.
    """
    return self.reconstruct_expression(), set()

  def reconstruct_all_expressions_with_input_names(self, seen_values=None):
    """Returns a list of code expressions and input names that were used.

    By default this simply calls reconstruct_expression_with_input_names().
    Subclasses should override this method if there are multiple possible
    reconstructions of the value.

    Args:
      seen_values: A set of Value objects that should not be used as
        descendants because they already appeared as an ancestor.
    """
    del seen_values  # Unused here, but used by OperationValue's implementation.
    return [self.reconstruct_expression_with_input_names()]

  @cache
  def num_elements(self):
    """Returns the number of elements in the wrapped value."""
    if self.is_sparse_tensor:
      return tf_coder_utils.num_tensor_elements(self.value.values)
    else:
      return tf_coder_utils.num_tensor_elements(self.value)

  @cache
  def max(self):
    """Returns the maximum of the wrapped value."""
    if self.elem_type in (int, float):
      return max(self.value)
    elif self.is_sparse_tensor:
      return tf_coder_utils.max_tensor_value(self.value.values)
    else:
      return tf_coder_utils.max_tensor_value(self.value)

  @cache
  def min(self):
    """Returns the minimum of the wrapped value."""
    if self.elem_type in (int, float):
      return min(self.value)
    elif self.is_sparse_tensor:
      return tf_coder_utils.min_tensor_value(self.value.values)
    else:
      return tf_coder_utils.min_tensor_value(self.value)

  @cache
  def reduce_prod(self):
    """Returns the product of the wrapped value's elements."""
    if self.is_sequence:
      return functools.reduce(operator.mul, self.value, 1)
    else:
      return float(tf.reduce_prod(tf.cast(self.value, tf.float32)))

  @cache
  def numpy_tolist(self):
    """Returns the wrapped value's elements as a list."""
    return self.value.numpy().tolist()

  @cache
  def has_int_dtype(self):
    """Returns whether the wrapped value has an int dtype."""
    return self.dtype in tf_coder_utils.INT_DTYPES

  @cache
  def has_float_dtype(self):
    """Returns whether the wrapped value has a float dtype."""
    return self.dtype in tf_coder_utils.FLOAT_DTYPES

  @cache
  def is_int_dtype(self):
    """Returns whether the wrapped value is itself an int dtype."""
    return self.value in tf_coder_utils.INT_DTYPES


OperationApplication = collections.namedtuple(
    'OperationApplication', ['operation', 'arg_values'])


class OperationValue(Value):
  """A Value resulting from the application of an Operation.

  Attributes:
    operation_applications: A list of OperationApplication namedtuples that
      describe different ways of reaching the value wrapped by this Value.
  """

  def __init__(self, value, operation, arg_values):
    super(OperationValue, self).__init__(value)
    self.operation_applications = [OperationApplication(operation=operation,
                                                        arg_values=arg_values)]
    self._expression_cache = None

  def copy(self):
    """See base class."""
    copied_value = OperationValue(self.value, operation=None, arg_values=None)
    copied_value.operation_applications = []
    for operation_application in self.operation_applications:
      copied_arg_values = [arg_value.copy()
                           for arg_value in operation_application.arg_values]
      copied_value.operation_applications.append(
          OperationApplication(operation_application.operation,
                               copied_arg_values))
    return copied_value

  def reconstruct_expression(self, use_cache=True):
    """See base class."""
    if use_cache and self._expression_cache is not None:
      return self._expression_cache
    first_way = self.operation_applications[0]
    self._expression_cache = first_way.operation.reconstruct_expression(
        first_way.arg_values, use_cache=use_cache)
    return self._expression_cache

  def reconstruct_expression_with_input_names(self):
    """See base class."""
    first_way = self.operation_applications[0]
    return first_way.operation.reconstruct_expression_with_input_names(
        first_way.arg_values)

  def merge_reconstructions(self, other_operation_value):
    """Adds another way of constructing this same value.

    Args:
      other_operation_value: An OperationValue that is equal to this object, but
        with a different reconstruction.
    """
    self.operation_applications.extend(
        other_operation_value.operation_applications)

  def reconstruct_all_expressions_with_input_names(self, seen_values=None):
    """See base class."""
    if seen_values is None:
      seen_values = set()

    seen_values.add(self)

    results = []
    for operation_application in self.operation_applications:
      operation = operation_application.operation
      arg_values = operation_application.arg_values
      if any(arg_value in seen_values for arg_value in arg_values):
        continue

      # A list containing a List[Tuple[reconstruction, used input names]] for
      # each argument.
      all_arg_reconstructions = [
          arg_value.reconstruct_all_expressions_with_input_names(
              seen_values=seen_values.copy())
          for arg_value in arg_values
      ]

      for product in itertools.product(*all_arg_reconstructions):
        expressions_list, input_names_list = zip(*product)
        results.append(
            (operation.reconstruct_expression_from_strings(expressions_list),
             set.union(*input_names_list)))

    return results


class ConstantValue(Value):
  """A constant value that is not created by any operation."""

  def copy(self):
    """See base class."""
    return ConstantValue(self.value)

  def reconstruct_expression(self, use_cache=True):
    """See base class."""
    return repr(self.value)


class InputValue(Value):
  """A value provided by the user as an input.

  Attributes:
    name: The name of the "Python variable" represented by this InputValue.
  """

  def __init__(self, value, name, skip_tensor_conversion=False):
    """Initializes an InputValue to contain `value` with name `name`."""
    if (not skip_tensor_conversion and
        not isinstance(value, tf_coder_utils.PRIMITIVE_TYPES)):
      try:
        value = tf_coder_utils.convert_to_tensor(value)
      except (TypeError, ValueError):
        pass
    super(InputValue, self).__init__(value)
    self.name = name

  def copy(self):
    """See base class."""
    return InputValue(self.value, self.name, skip_tensor_conversion=True)

  def reconstruct_expression(self, use_cache=True):
    """See base class."""
    return self.name

  def reconstruct_expression_with_input_names(self):
    """See base class."""
    return self.name, {self.name}


class OutputValue(Value):
  """A Value representing the user's desired output.

  This class is simply a wrapper aound the output value so that it can be
  compared to other Value objects.
  """

  def __init__(self, value):
    """Initializes an OutputValue to contain `value`."""
    value = tf_coder_utils.convert_to_tensor(value)
    super(OutputValue, self).__init__(value)

  def reconstruct_expression(self, use_cache=True):
    """An OutputValue is not created from any expression."""
    raise NotImplementedError()


class ExpressionValue(Value):
  """A Value arising from a known simple expression."""
  # TODO(kshi): This class is currently unused (outside of being convenient for
  # tests), but it may be useful to implement expressions like `in1.shape[1]`.

  def __init__(self, value, expression, used_input_names=None):
    """Initializes the ExpressionValue."""
    super(ExpressionValue, self).__init__(value)
    self.expression = expression
    self.used_input_names = set(used_input_names) if used_input_names else set()

  def copy(self):
    """See base class."""
    return ExpressionValue(self.value, self.expression)

  def reconstruct_expression(self, use_cache=True):
    """See base class."""
    return self.expression

  def reconstruct_expression_with_input_names(self):
    """See base class."""
    return self.expression, self.used_input_names

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
"""Defines the base Operation class for value search."""

import abc
import itertools
import sys
import timeit
import typing
from typing import Callable, Dict, List, Optional, Sequence, Set, Text, Tuple, Union

import six
from tf_coder import tf_coder_utils
from tf_coder import tf_functions
from tf_coder.value_search import filtered_values_cache
from tf_coder.value_search import operation_statistics
from tf_coder.value_search import value
from tf_coder.value_search import value_search_settings as settings_module

################################################################################
# Type aliases.

# The i-th element contains all Value objects of weight i, mapped to themselves.
ValuesByWeightDict = List[Dict[value.Value, value.Value]]

# The i-th element is an iterable of all Value objects of weight i.
ValuesByWeightIterable = List[Union[List[value.Value],
                                    Dict[value.Value, value.Value]]]

# The i-th inner list contains Value objects that are candidates for argument i.
ArgOptionsType = List[List[value.Value]]

# The i-th Value is used as the i-th argument for an Operation application.
ArgValuesType = Sequence[value.Value]

# An optional filter function that is applied to single Value objects. If None,
# it is treated as a function that always returns True (all Value objects are
# allowed).
ValueFilterType = Optional[Callable[[value.Value], bool]]

# An optional filter function that is applied to a list of argument values. If
# None, it is treated as a function that always returns True (all argument lists
# are allowed).
ApplyFilterType = Optional[Callable[[ArgValuesType], bool]]

################################################################################


OperationMetadata = typing.NamedTuple(
    'OperationMetadata', [('docstring', Text)])


@six.add_metaclass(abc.ABCMeta)
class Operation(object):
  """An operation that can be applied to a constant number of arguments.

  Arguments are always ordered, and subclasses can choose their own conventions
  for this ordering. The operation must be deterministic, must not have
  side-effects, and must not modify its arguments.

  Attributes:
    name: A unique name for this operation.
    num_args: The number of arguments required by this Operation.
    weight: The weight of this node in the AST.
    metadata: Metadata for this Operation.

    _value_filters_list: A list of lists of filter functions.

      Each inner list has length num_args and contains a filter function for
      each argument, where the i-th filter function takes a Value and returns
      whether that Value should be an option for the i-th argument. Any filter
      function can be None, which means all values should be options for that
      argument.

      The outer list can have multiple lists of filter functions, where each
      inner list describes one class of valid argument values.

      The value_filters_list attribute can also be None, in which case all
      values should be options for all arguments.

    _apply_filter: A filter function that takes a list of Value objects of
      length num_args (the arguments to a potential application of this
      Operation), and returns whether those Value objects are compatible (i.e.,
      whether the operation should be applied). If None, the operation is always
      applied.

    _name_cache: A cached copy of this Operation's name.
  """

  def __init__(self, num_args: int, weight: int,
               metadata: OperationMetadata) -> None:
    """Initializes an Operation."""
    self.num_args = num_args
    self.weight = weight
    self.metadata = metadata
    self._value_filters_list = None  # type: Optional[List[List[ValueFilterType]]]
    self._apply_filter = None  # type: ApplyFilterType
    self._name_cache = None

  @property
  def name(self) -> Text:
    """The (cached) name of the operation."""
    if self._name_cache is not None:
      return self._name_cache
    self._name_cache = self._compute_name()
    return self._name_cache

  def _compute_name(self) -> Text:
    """Computes a name for this operation."""
    return self.__class__.__name__

  def add_value_filters(self, value_filters: List[ValueFilterType]) -> None:
    """Adds the given value filters to the value_filters_list attribute.

    Args:
      value_filters: A list of filter functions, one per argument, where the
        i-th filter function takes a Value and returns whether it should be
        an option for argument i.

    Raises:
      ValueError: If the list of filter functions has the wrong length.
    """
    if len(value_filters) != self.num_args:
      raise ValueError('value_filters must contain one filter per argument.')
    if self._value_filters_list is None:
      self._value_filters_list = []
    self._value_filters_list.append(value_filters)

  def set_apply_filter(self, apply_filter: ApplyFilterType) -> None:
    """Sets the given apply_filter."""
    self._apply_filter = apply_filter

  @abc.abstractmethod
  def apply(self,
            arg_values: ArgValuesType,
            settings: settings_module.Settings) -> Optional[value.Value]:
    """Applies this Operation to a list of arguments (Value objects).

    Args:
      arg_values: A list of Value objects representing the arguments.
      settings: A Settings object storing settings for this search.

    Returns:
      A Value object representing the result if successful, or None if the
      operation raises an exception.
    """

  def _enumerate_values(
      self,
      arg_options: ArgOptionsType,
      end_time: float,
      settings: settings_module.Settings,
      statistics: Optional[operation_statistics.OperationStatistics] = None,
  ) -> List[value.Value]:
    """Enumerates values that are created from multiple choices of arguments.

    Args:
      arg_options: A list of lists of Value objects, where the i-th list
        contains the possible Value objects for the i-th argument.
      end_time: A timeit.default_timer() cutoff where this should timeout.
      settings: A Settings object storing settings for this search.
      statistics: An optional OperationStatistics object to track statistics
        during this function's execution.

    Returns:
      A list of Value objects, one for every successful application of the
      operation.
    """
    results = []  # type: List[value.Value]
    apply_count = 0
    apply_successes = 0
    start_time = timeit.default_timer()

    for i, arg_values in enumerate(itertools.product(*arg_options)):
      # Check for timeout periodically.
      if i % 1000 == 0 and timeit.default_timer() > end_time:
        break

      # Skipping filtering is only used for experiments in the PLDI paper.
      if not (settings.paper_experiments.skip_filtering and
              self.name not in tf_functions.REQUIRES_FILTERING):
        # _apply_filter is either None or callable.
        if (self._apply_filter is not None and
            not self._apply_filter(arg_values)):  # pylint: disable=not-callable
          continue

      if settings.printing.all_apply:
        print('Applying {} on arguments: {}'.format(
            self.name,
            [arg_value.reconstruct_expression() for arg_value in arg_values]))
        # Print the output immediately so it isn't swallowed by a stacktrace.
        sys.stdout.flush()
      maybe_value = self.apply(arg_values, settings)
      apply_count += 1
      if maybe_value is not None:
        yes_value = maybe_value  # type: value.Value
        apply_successes += 1
        results.append(yes_value)

    elapsed_time = timeit.default_timer() - start_time
    if statistics:
      statistics.update(operation_name=self.name,
                        count=apply_count,
                        successes=apply_successes,
                        time=elapsed_time)
    return results

  def enumerate_values_with_weight(
      self,
      target_weight: int,
      values_by_weight: ValuesByWeightDict,
      filter_cache: filtered_values_cache.FilteredValuesCache,
      end_time: float,
      settings: settings_module.Settings,
      statistics: Optional[operation_statistics.OperationStatistics] = None
  ) -> List[value.Value]:
    """Enumerates values with a given target weight.

    Args:
      target_weight: The desired weight of resulting values.
      values_by_weight: A collection of Values organized by their weight.
      filter_cache: The FilteredValuesCache object used during this search.
      end_time: A timeit.default_timer() cutoff where this should timeout.
      settings: A Settings object storing settings for this search.
      statistics: An optional OperationStatistics object to track statistics
        during this function's execution.

    Returns:
      A list of Value objects of the specified weight.
    """
    num_args = self.num_args
    if num_args == 0:
      return []  # An operation with no arguments can't have variable weight.
    if target_weight - self.weight - num_args < 0:
      return []  # Too many arguments for this weight.

    results = []  # type: List[value.Value]

    for value_filters in self._value_filters_list:
      assert len(value_filters) == num_args

      # Enumerate ways of partitioning (target_weight - self.weight) into
      # (num_args) positive pieces.
      # Equivalently, partition (target_weight - self.weight - num_args) into
      # (num_args) nonnegative pieces.
      arg_options_list = []  # type: List[ArgOptionsType]
      for partition in tf_coder_utils.generate_partitions(
          target_weight - self.weight - num_args,
          num_args):  # type: Tuple[int, ...]
        if (settings.paper_experiments.skip_filtering and
            self.name not in tf_functions.REQUIRES_FILTERING):
          # Only for experiments in the PLDI paper.
          arg_options = [
              values_by_weight[weight_minus_1 + 1]
              for arg, weight_minus_1 in enumerate(partition)
          ]  # type: ArgOptionsType
        else:
          arg_options = [
              filter_cache.filter_values(value_filters[arg], weight_minus_1 + 1,
                                         values_by_weight[weight_minus_1 + 1])
              for arg, weight_minus_1 in enumerate(partition)
          ]  # type: ArgOptionsType
        arg_options_list.append(arg_options)

      for arg_options in arg_options_list:
        results.extend(self._enumerate_values(arg_options, end_time, settings,
                                              statistics))
    return results

  def reconstruct_expression(self, arg_values: ArgValuesType,
                             use_cache=True) -> Text:
    """Returns an expression for this operation applied to the given arguments.

    This can be slow and should not be called in a tight loop.

    Args:
      arg_values: A list of Value objects representing the arguments' values.
      use_cache: If True, the reconstruction may be looked up from a cache. If
        False, the reconstruction will be recomputed on each call.

    Returns:
      A string representation of the code expression.
    """
    arg_strings = [arg_value.reconstruct_expression(use_cache=use_cache)
                   for arg_value in arg_values]
    return self.reconstruct_expression_from_strings(arg_strings)

  def reconstruct_expression_with_input_names(
      self, arg_values: ArgValuesType) -> Tuple[Text, Set[Text]]:
    """Returns an expression for this operation and the used input names."""
    arg_strings_list, input_names_list = zip(
        *[arg_value.reconstruct_expression_with_input_names()
          for arg_value in arg_values])
    return (self.reconstruct_expression_from_strings(arg_strings_list),
            set.union(*input_names_list))

  @abc.abstractmethod
  def reconstruct_expression_from_strings(self,
                                          arg_strings: List[Text]) -> Text:
    """Returns an expression for this operation applied to the given arguments.

    This can be slow and should not be called in a tight loop.

    Args:
      arg_strings: A list of strings representing the arguments'
        reconstructions.

    Returns:
      A string representation of the code expression.
    """

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
"""Defines the Operation objects for functions."""

import re

import funcsigs
import tensorflow as tf
from tf_coder import tensor_limits as limits
from tf_coder import tf_coder_utils
from tf_coder import tf_functions
from tf_coder.value_search import operation_base
from tf_coder.value_search import operation_filtering
from tf_coder.value_search import value


class FunctionOperation(operation_base.Operation):
  """An operation that applies a function to some arguments.

  The arguments must be given in the same order as they appear in the function's
  signature.

  Arguments with default values in the function signature are optional at the
  time of FunctionOperation creation. However, once created, a FunctionOperation
  must be used with all of the arguments provided to its constructor.
  """

  def __init__(self, function_info):
    """Creates a FunctionOperation.

    Args:
      function_info: A tf_functions.FunctionInfo.
    """
    function_name, arg_names, constant_kwargs = (
        tf_functions.parse_function_info_name(function_info))
    self._function_obj = tf_coder_utils.get_tf_function(function_name)
    docstring = self._function_obj.__doc__
    if not docstring:
      print('Warning: could not get docstring for function {}'.format(
          function_name))
      docstring = ''

    # Make sure the function and argument names appear in the docstring. (Args
    # should already appear in the docstring "Args" section though.)
    docstring += '\n' + function_info.name
    # If 'reduce_max' is the function name, make sure 'reduce' and 'max' also
    # appear as separate words. Ditto for argument names as well.
    docstring += '\n' + function_info.name.replace('_', ' ')
    # Upweight the function name (moreso than the argument names).
    function_name_without_tf = re.sub(r'^tf\.', '', function_name)
    docstring += ('\n' + function_name_without_tf) * 4
    if '_' in function_name_without_tf:
      docstring += ('\n' + function_name_without_tf.replace('_', ' ')) * 2

    metadata = operation_base.OperationMetadata(docstring=docstring)
    super(FunctionOperation, self).__init__(
        num_args=len(arg_names), weight=function_info.weight, metadata=metadata)

    self.function_info = function_info
    self.function_name = function_name
    self.arg_names = arg_names
    self.constant_kwargs = constant_kwargs

    self._has_default = {}

    parameters = funcsigs.signature(self._function_obj).parameters
    for arg_name in arg_names:
      param = parameters[arg_name]
      has_default = param.default is not param.empty
      self._has_default[arg_name] = has_default

    operation_filtering.add_filters_to_function_operation(self)

  def _compute_name(self):
    return self.function_info.name

  def _print_warnings(self, arg_values, result_value):
    if isinstance(result_value, tf.Tensor):
      num_elements = tf_coder_utils.num_tensor_elements(result_value)
    elif isinstance(result_value, tf.SparseTensor):
      num_elements = tf_coder_utils.num_tensor_elements(result_value.values)
    else:
      return
    if num_elements > 10 * limits.MAX_TENSOR_ELEMENTS:
      print('Warning: {} produced much-too-large tensor of shape {} and {} '
            'elements.'.format(
                self.name,
                result_value.shape.as_list(),
                num_elements))
      for i, arg_value in enumerate(arg_values):
        if isinstance(arg_value.value, tf.Tensor):
          print('  argument {} has shape {} and {} elements'.format(
              i, arg_value.shape, arg_value.num_elements()))
          if arg_value.num_elements() <= 20:
            print('  argument {} is: {}'.format(i, arg_value.value))
        elif arg_value.is_primitive:
          print('  argument {} is: {}'.format(i, arg_value.value))
        else:
          print('  argument {} has type {}'.format(
              i, type(arg_value.value)))
        print('  argument {} has reconstruction: {}'.format(
            i, arg_value.reconstruct_expression()))

  def apply(self, arg_values, settings):
    """See base class."""
    value_objects = [arg_value.value for arg_value in arg_values]
    arg_dict = dict(zip(self.arg_names, value_objects))
    arg_dict.update(self.constant_kwargs)
    try:
      result_value = self._function_obj(**arg_dict)
    except Exception:  # pylint: disable=broad-except
      return None
    try:
      return value.OperationValue(result_value, self, arg_values)
    except ValueError:
      if settings.printing.tensor_size_warnings:
        self._print_warnings(arg_values, result_value)
      return None

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    arg_strings = list(arg_strings)
    for i, arg_name in enumerate(self.arg_names):
      if self._has_default[arg_name]:
        arg_strings[i] = arg_name + '=' + arg_strings[i]
    for kwarg_name, kwarg_value in self.constant_kwargs.items():
      arg_strings.append('{}={!r}'.format(kwarg_name, kwarg_value))
    return self.function_name + '(' + ', '.join(arg_strings) + ')'

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
"""An interface for handling natural language descriptions for TF-Coder."""

import abc
import re
from typing import Dict, List, Optional, Text

import six
from tf_coder.value_search import all_operations
from tf_coder.value_search import operation_base
from tf_coder.value_search import value_search_settings as settings_module


@six.add_metaclass(abc.ABCMeta)
class DescriptionHandler(object):
  """Handles a natural language description of a task.

  Attributes:
    operations: A list of operations that the handler knows about.
    all_names: A list of operation names, in the same order as the `operations`
      list.
  """

  def __init__(self,
               operations: Optional[List[operation_base.Operation]] = None):
    """Initializes the handler.

    Args:
      operations: A list of operations that the scorer should handle. Exposed
        for testing.
    Raises:
      ValueError: If there are duplicate operation names.
    """
    self.operations = (
        operations if operations
        else all_operations.get_operations(include_sparse_operations=True))
    self.all_names = [operation.name for operation in self.operations]
    if len(set(self.all_names)) != len(self.operations):
      raise ValueError('Duplicate operation name.')

  @abc.abstractmethod
  def get_operation_multipliers(
      self,
      description: Text,
      settings: settings_module.Settings) -> Dict[Text, float]:
    """Returns a map from operation names to their weight multiplier.

    The weight multiplier should be between 0 and 1 if the operation should be
    prioritized, or greater than 1 if it should be deprioritized.

    Args:
      description: The natural language description of a TF-Coder task, provided
        by the user.
      settings: A Settings object storing settings for this search.

    Returns:
      A map from operation name to weight multiplier, such that the operation
      with that name should have its weight modified by that multiplier. If the
      dict does not contain a key, it means the weight should not be modified
      (equivalent to a multiplier of 1).
    """

  def __repr__(self) -> Text:
    """Returns a string containing details about this handler and parameters."""
    return self.__class__.__name__


class NoChangeDescriptionHandler(DescriptionHandler):
  """A description handler that does not change any operation weights."""

  def get_operation_multipliers(
      self,
      description: Text,
      settings: settings_module.Settings) -> Dict[Text, float]:
    """See base class."""
    return {}


class FunctionNameDescriptionHandler(DescriptionHandler):
  """Prioritizes functions with names that appear in the docstring."""

  def __init__(self,
               operations: Optional[List[operation_base.Operation]] = None,
               multiplier: float = 0.75):
    """Creates a FunctionNameDescriptionHandler.

    Args:
      operations: A list of operations that the scorer should handle. Exposed
        for testing.
      multiplier: The multiplier applied to an operation's weight if it is
        prioritized.
    """
    super(FunctionNameDescriptionHandler, self).__init__(operations)
    self.multiplier = multiplier

  def get_operation_multipliers(
      self,
      description: Text,
      settings: settings_module.Settings) -> Dict[Text, float]:
    """See base class."""
    description = description.lower()
    multipliers = {}
    for name in self.all_names:
      if name.startswith('tf.') and '(' in name:
        function_name = name[len('tf.') : name.index('(')].lower()
        function_name_parts = re.split(r'[._]', function_name)
        if all(part in description for part in function_name_parts):
          if settings.printing.prioritized_operations:
            print('FunctionNameDescriptionHandler prioritized {}'.format(name))
          multipliers[name] = self.multiplier
    return multipliers

  def __repr__(self) -> Text:
    """See base class."""
    return '{}(multiplier={})'.format(self.__class__.__name__, self.multiplier)

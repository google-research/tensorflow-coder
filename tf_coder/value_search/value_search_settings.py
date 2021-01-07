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
"""Settings specific to the value search approach to the TF-Coder problem."""

import ast
import os
from typing import Any, Dict, List, Text


class Settings(object):
  """Stores settings for TF-Coder's value search algorithm."""

  def __init__(self):
    # A string describing the current version of the search algorithm.
    self.algorithm_version = (
        'Value search, '
        'TF-IDF (k=5, min_score=0.15), '
        'tensor features model with F_1 loss and max weighting, '
        '2020/08/26')

    # Time limit in seconds.
    self.timeout = 60

    # Maximum number of solutions to search for.
    self.max_solutions = 1

    # Whether to only search for solutions with minimal weight.
    self.only_minimal_solutions = True

    # Maximum number of seconds to spend searching for solutions after the
    # first.
    self.max_extra_solutions_time = 10

    # Maximum weight of an expression to search for.
    self.max_weight = 300

    # Whether to require solutions to use all inputs, at least one input, or no
    # restriction.
    self.require_all_inputs_used = True
    self.require_one_input_used = True

    # The description handler to use.
    self.description_handler_name = 'tfidf_5_0.15'

    # Other settings organized into separate objects.
    self.operations = OperationSettings()
    self.tensor_model = TensorModelSettings()
    self.printing = PrintSettings()
    self.paper_experiments = PaperExperimentSettings()

  # Used to parse setting names.
  _GROUP_NAMES = ['operations', 'tensor_model', 'printing', 'paper_experiments']

  def set(self, name: Text, value: Any) -> None:
    """Sets the setting with the given name to the given value.

    Args:
      name: The name of the setting to set. For example, 'timeout' is used to
        set `self.timeout`, and either 'printing.statistics' or
        'printing_statistics' can be used to set `self.printing.statistics`.
      value: The value to set the setting to.
    """
    if hasattr(self, name):
      setattr(self, name, value)
    else:
      for group_name in Settings._GROUP_NAMES:
        if (name.startswith(group_name) and
            name[len(group_name)] in {'.', '_'}):
          reduced_name = name[len(group_name) + 1 : ]
          group = getattr(self, group_name)
          if hasattr(group, reduced_name):
            setattr(group, reduced_name, value)
            break
      else:
        raise ValueError(
            'The name `{}` does not match any setting.'.format(name))

  def as_dict(self) -> Dict[Text, Any]:
    """Returns all settings as a dict."""
    result = {}
    for name, value in self.__dict__.items():
      if name in Settings._GROUP_NAMES:
        for inner_name, inner_value in value.__dict__.items():
          full_name = name + '.' + inner_name
          result[full_name] = inner_value
      else:
        result[name] = value
    return result


class OperationSettings(object):
  """Settings about operations to use during search."""

  def __init__(self):
    # Whether to limit sparse operations to benchmarks that contain
    # SparseTensors in their examples.
    self.limit_sparse_operations = True

    # TODO(kshi): Add options to exclude specific operations, or prioritize
    # user-chosen operations.


class TensorModelSettings(object):
  """Settings for the tensor features model."""

  def __init__(self):
    # This model is trained with max-weight F_1 loss, learning rate 1e-4, and 1
    # hidden layer of size 1024.
    model_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'models', 'trained_model'))
    self.checkpoint_path = os.path.join(model_dir, 'ckpt-1172')
    self.config_path = os.path.join(model_dir, 'config.json')

    # How the model should prioritize or deprioritize operations.
    self.prioritize_threshold = 0.5
    self.prioritize_multiplier = 0.75
    self.deprioritize_threshold = -1  # No deprioritization.
    self.deprioritize_multiplier = 1.25
    self.max_deprioritized = 100


class PrintSettings(object):
  """Settings that affect printing to stdout."""

  def __init__(self):
    # Whether to print intermediate results and progress. Setting this to True
    # will cause significant slowdown from computing and printing many
    # expressions.
    self.verbose = False

    # Whether to print every FunctionOperation application before it occurs.
    # Setting this to True will cause a huge amount of output and significant
    # slowdown.
    self.all_apply = False

    # Whether to print warnings about too-large tensors.
    self.tensor_size_warnings = False

    # Whether to print progress at each iteration of target expression weight.
    self.progress = False

    # Whether to print bad solutions.
    self.bad_solutions = False

    # Whether to print statistics about operations and executions.
    self.statistics = False

    # Whether to print statistics sorted by time (versus by name). Ignored if
    # `statistics` is False.
    self.statistics_sort_by_time = True

    # Whether to print the operations that are prioritized or deprioritized.
    self.prioritized_operations = False
    self.deprioritized_operations = False


class PaperExperimentSettings(object):
  """Settings for experiments in the PLDI 2020 paper."""

  def __init__(self):
    self.skip_filtering = False
    self.uniform_weights = False


def default_settings() -> Settings:
  """Returns a Settings object with default settings."""
  return Settings()


def from_dict(overrides: Dict[Text, Any]) -> Settings:
  """Sets settings using a dict to override defaults."""
  settings = default_settings()
  for name, value in overrides.items():
    settings.set(name, value)
  return settings


def from_list(overrides: List[Text]) -> Settings:
  """Sets settings using a list to override defaults.

  Args:
    overrides: A list of strings like 'timeout=120' or
      'printing.statistics=True'. Each string should contain exactly one '='
      character. The portion before the '=' character names a setting to
      override. The portion after the '=' character describes the value of the
      setting, in a form parseable by ast.literal_eval().

  Raises:
    ValueError: If any element of `overrides` cannot be processed
      successfully.

  Returns:
    A Settings object.
  """
  settings = default_settings()
  for override_string in overrides:
    if override_string.count('=') != 1:
      raise ValueError("The override string {!r} does not contain exactly "
                       "one '=' character.".format(override_string))
    equals_index = override_string.index('=')
    name = override_string[:equals_index]
    value_string = override_string[equals_index + 1:]
    try:
      value = ast.literal_eval(value_string)
      settings.set(name, value)
    except Exception as e:
      raise ValueError(
          'Exception raised in ast.literal_eval on {!r}: {}'.format(
              value_string, e))
  return settings

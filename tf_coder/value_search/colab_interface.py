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
"""Value search endpoint for the Colab interface."""

from typing import Any, Dict, List, Optional, Text

from tf_coder.benchmarks import all_benchmarks
from tf_coder.models import tensor_features_model
from tf_coder.natural_language import description_handler_factory
from tf_coder.value_search import all_operations
from tf_coder.value_search import value_search
from tf_coder.value_search import value_search_settings as settings_module

# A message to print when things go wrong.
CONTACT_MESSAGE = (
    'If you see this message, please raise a GitHub issue '
    '(https://github.com/google-research/tensorflow-coder/issues) describing '
    'what happened.')
# TODO(kshi): Add a link to GitHub.

# The global description handler for this session. This will be loaded during
# warm_up().
DESCRIPTION_HANDLER = None

# The tensor features model and config. These will be loaded during warm_up().
TENSOR_MODEL = None
TENSOR_CONFIG = None

# Whether warm_up() was called before.
WARMED_UP = False

# Default settings, if the user doesn't provide their own settings.
DEFAULT_SETTINGS = settings_module.default_settings()


def warm_up():
  """Loads and warms up the tensor features model."""
  global DESCRIPTION_HANDLER, TENSOR_MODEL, TENSOR_CONFIG, WARMED_UP
  if WARMED_UP:
    return
  WARMED_UP = True

  # Use the default choices for the description handler and tensor model. The
  # Colab interface will not allow users to change these.

  DESCRIPTION_HANDLER = description_handler_factory.create_handler(
      DEFAULT_SETTINGS.description_handler_name)
  if (not DEFAULT_SETTINGS.tensor_model.config_path or
      not DEFAULT_SETTINGS.tensor_model.checkpoint_path):
    return
  try:
    TENSOR_CONFIG = tensor_features_model.load_config(
        DEFAULT_SETTINGS.tensor_model.config_path)
    TENSOR_MODEL = tensor_features_model.get_model(TENSOR_CONFIG)
    tensor_checkpoint = tensor_features_model.create_checkpoint(TENSOR_MODEL)
    tensor_checkpoint.restore(
        DEFAULT_SETTINGS.tensor_model.checkpoint_path).expect_partial()

    # Warm up. Running the model for the first time takes an extra ~10 seconds.
    value_search.operation_multipliers_from_tensor_model(
        all_benchmarks.find_benchmark_with_name('simple_cast'),
        TENSOR_MODEL, TENSOR_CONFIG, DEFAULT_SETTINGS)
  except Exception:  # pylint: disable=broad-except
    # No matter what goes wrong with loading the tensor features model, we
    # should fall back to value search without the model.
    print('Could not load the tensor features model. ' + CONTACT_MESSAGE)
    TENSOR_MODEL = None
    TENSOR_CONFIG = None


def run_value_search_from_colab(
    inputs: Dict[Text, Any],
    output: Any,
    constants: Optional[List[Any]] = None,
    description: Optional[Text] = None,
    settings: Optional[settings_module.Settings] = None
) -> value_search.ValueSearchResults:
  """Value search endpoint for the Colab interface.

  Args:
    inputs: A dict mapping input variable names to input tensors.
    output: The corresponding desired output.
    constants: An optional list of scalar constants.
    description: An optional natural language description of the task.
    settings: A Settings object containing settings for the search.

  Returns:
    A ValueSearchResults namedtuple.
  """
  if not WARMED_UP:
    warm_up()

  return value_search.run_value_search_from_example(
      inputs=inputs,
      output=output,
      settings=settings if settings is not None else DEFAULT_SETTINGS,
      constants=constants,
      description=description,
      source='From TF-Coder Colab',
      description_handler=DESCRIPTION_HANDLER,
      tensor_model=TENSOR_MODEL,
      tensor_config=TENSOR_CONFIG)


def print_supported_operations():
  """Prints all of the supported operations."""
  print('TensorFlow functions:\n'
        '---------------------')
  for operation in all_operations.get_tf_operations():
    print(operation.name)
  print()
  print('SparseTensor functions:\n'
        '-----------------------')
  for operation in all_operations.get_sparse_operations():
    print(operation.name)
  print()
  print('Python-syntax operations:\n'
        '-------------------------')
  for operation in all_operations.get_python_operations():
    syntax_form = operation.reconstruct_expression_from_strings(
        ['arg{}'.format(i+1) for i in range(operation.num_args)])
    print('{:35} {}'.format(operation.name + ':', syntax_form))

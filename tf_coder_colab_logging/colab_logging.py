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
"""Sends logging information from the Colab interface to Google Analytics."""

import uuid

from tf_coder import version as tf_coder_version
from tf_coder_colab_logging import serialization
from tf_coder_colab_logging import version as logging_version

try:
  # pytype: disable=import-error
  import IPython  # pylint: disable=g-import-not-at-top
  # pytype: enable=import-error
except ImportError as e:
  print('Warning: colab_logging is not being loaded in an IPython environment. '
        'No logging will happen.')
  # Use a mock for testing purposes.
  import mock  # pylint: disable=g-import-not-at-top
  IPython = mock.MagicMock()  # pylint: disable=invalid-name
  _IPYTHON_MOCK = IPython


def get_uuid():
  """Returns a 64-bit random number."""
  return uuid.uuid4().int & ((1 << 64) - 1)

# A random ID for this session.
SESSION_ID = get_uuid()

# Character limit for each custom dimension sent to Google Analytics.
CHARS_PER_DIMENSION = 150
# The number of available custom dimensions.
NUM_DIMENSIONS = 30


def load_gtag():
  """Loads gtag.js."""
  # Note: gtag.js MUST be loaded in the same cell execution as the one doing
  # synthesis. It does NOT persist across cell executions!
  html_code = '''
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-141920863-2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-141920863-2',
       {'referrer': document.referrer.split('?')[0],
        'anonymize_ip': true,
        'page_title': '',
        'page_referrer': ''});
</script>
'''
  IPython.display.display(IPython.display.HTML(html_code))


def create_gtag_js_string(action, category, label=None, logging_dict=None):
  """Creates JavaScript code that sends info to Google Analytics."""
  custom_dimensions_js = ''
  if logging_dict is not None:
    unescaped_text = serialization.serialize(logging_dict)
    dimension_index = 0
    while unescaped_text:
      dimension_index += 1
      this_dimension_text = repr(unescaped_text[:CHARS_PER_DIMENSION])
      unescaped_text = unescaped_text[CHARS_PER_DIMENSION:]
      custom_dimensions_js += "'dimension{}': {}, ".format(
          dimension_index, this_dimension_text)
    if dimension_index > NUM_DIMENSIONS:
      return None

  maybe_label = ("'event_label': '{}', ".format(label)
                 if label is not None else '')
  return ("gtag('event', '{action}', {{'event_category': '{category}', "
          "{maybe_label}{custom_dimensions_js}}})").format(
              action=action,
              category=category,
              maybe_label=maybe_label,
              custom_dimensions_js=custom_dimensions_js)


def run_js(js_string):
  """Runs a JavaScript string."""
  if js_string:
    IPython.display.display(IPython.display.Javascript(js_string))


def get_problem_logging_dict(inputs, output, constants, description, settings,
                             include_in_dataset, problem_id):
  return {
      'inputs': inputs,
      'output': output,
      'constants': constants,
      'description': description,
      'settings': settings.as_dict(),
      'include_in_dataset': include_in_dataset,
      'problem_id': problem_id,
      'session_id': SESSION_ID,
      'tf_coder_version': tf_coder_version.__version__,
      'logging_version': logging_version.__version__,
  }


def log_problem(inputs, output, constants, description, settings,
                include_in_dataset, problem_id, label=None):
  """Logs a TF-Coder problem through Google Analytics."""
  # gtag.js must be loaded during this cell execution.
  load_gtag()

  logging_dict = get_problem_logging_dict(
      inputs=inputs,
      output=output,
      constants=constants,
      description=description,
      settings=settings,
      include_in_dataset=include_in_dataset,
      problem_id=problem_id)
  run_js(create_gtag_js_string(
      'problem', category='tf_coder_problem', label=label,
      logging_dict=logging_dict))


def log_result(results, include_in_dataset, problem_id, label=None):
  """Logs a problem along with TF-Coder's results."""
  benchmark = results.benchmark
  success = bool(results.solutions)
  # Info about the problem.
  logging_dict = get_problem_logging_dict(
      inputs=benchmark.examples[0].inputs,
      output=benchmark.examples[0].output,
      constants=benchmark.constants,
      description=benchmark.description,
      settings=results.settings,
      include_in_dataset=include_in_dataset,
      problem_id=problem_id)
  # Info about TF-Coder's results.
  logging_dict.update({
      'success': success,
      'solution_expressions': [s.expression for s in results.solutions],
      'solution_weights': [s.weight for s in results.solutions],
      'solution_times': [s.time for s in results.solutions],
      'total_time': results.total_time,
      'value_set_size': len(results.value_set),
  })
  run_js(create_gtag_js_string(
      'result', category='tf_coder_result', label=label,
      logging_dict=logging_dict))

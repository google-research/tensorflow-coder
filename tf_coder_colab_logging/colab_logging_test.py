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
"""Tests for colab_logging.py."""

from absl.testing import absltest
from tf_coder import version as tf_coder_version
from tf_coder_colab_logging import colab_logging
from tf_coder_colab_logging import version as logging_version
from tf_coder.value_search import colab_interface
from tf_coder.value_search import value_search_settings as settings_module


class ColabLoggingTest(absltest.TestCase):

  def setUp(self):
    super(ColabLoggingTest, self).setUp()
    self.inputs = [[1, 2, 3],
                   [10, 20, 30, 40]]
    self.output = [[11, 21, 31, 41],
                   [12, 22, 32, 42],
                   [13, 23, 33, 43]]
    self.constants = [12, 34, (56, 78)]
    self.description = 'add two vectors with broadcasting to get a matrix'
    self.settings = settings_module.from_dict({
        'timeout': 60,
        'only_minimal_solutions': False,
        'max_solutions': 2,
    })
    colab_logging._IPYTHON_MOCK.reset_mock()

  def test_get_uuid_returns_different_results(self):
    # This test will fail with probability 2^-64.
    self.assertNotEqual(colab_logging.get_uuid(), colab_logging.get_uuid())

  def test_runs(self):
    problem_id = colab_logging.get_uuid()
    colab_logging.log_problem(
        self.inputs, self.output, self.constants, self.description,
        self.settings, include_in_dataset=True, problem_id=problem_id)

    results = colab_interface.run_value_search_from_colab(
        self.inputs, self.output, self.constants, self.description,
        self.settings)

    colab_logging.log_result(
        results, include_in_dataset=True, problem_id=problem_id)

    colab_logging._IPYTHON_MOCK.display.HTML.assert_called_once()
    colab_logging._IPYTHON_MOCK.display.Javascript.assert_called()
    js_strings = [
        call_args[0] for call_args in (
            colab_logging._IPYTHON_MOCK.display.Javascript.call_args_list)
    ]
    self.assertTrue(all(js_strings))  # None of them are empty or None.

  def test_too_large_logging_dict(self):
    # Pretend that there are very few dimensions, so that the serialized logging
    # dict is too large to send to GA.
    old_num_dimensions = colab_logging.NUM_DIMENSIONS
    colab_logging.NUM_DIMENSIONS = 5

    problem_id = colab_logging.get_uuid()
    colab_logging.log_problem(
        self.inputs, self.output, self.constants, self.description,
        self.settings, include_in_dataset=True, problem_id=problem_id)

    results = colab_interface.run_value_search_from_colab(
        self.inputs, self.output, self.constants, self.description,
        self.settings)

    colab_logging.log_result(
        results, include_in_dataset=True, problem_id=problem_id)

    colab_logging._IPYTHON_MOCK.display.HTML.assert_called_once()
    colab_logging._IPYTHON_MOCK.display.Javascript.assert_not_called()

    colab_logging.NUM_DIMENSIONS = old_num_dimensions

  def test_version(self):
    inputs = {
        'rows': [10, 20, 30],
        'cols': [1, 2, 3, 4],
    }
    output = [[11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34]]
    constants = []
    description = 'add two vectors with broadcasting to get a matrix'
    logging_dict = colab_logging.get_problem_logging_dict(
        inputs=inputs,
        output=output,
        constants=constants,
        description=description,
        settings=settings_module.from_dict({'timeout': 99}),
        include_in_dataset=False,
        problem_id=123)
    self.assertEqual(logging_dict['tf_coder_version'],
                     tf_coder_version.__version__)
    self.assertEqual(logging_dict['logging_version'],
                     logging_version.__version__)


if __name__ == '__main__':
  absltest.main()

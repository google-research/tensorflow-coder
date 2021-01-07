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
"""Tests for value_search.py."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tf_coder.value_search import value_search_main


class ValueSearchMainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('all', 'ALL', True),
      ('stackoverflow_01', 'stackoverflow_01', True),
      ('nonexistent_benchmark', 'no_benchmark_has_this_name', False))
  def test_benchmark_name_validator(self, benchmark_name, expected_result):
    self.assertEqual(
        value_search_main.benchmark_name_validator(benchmark_name),
        expected_result)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

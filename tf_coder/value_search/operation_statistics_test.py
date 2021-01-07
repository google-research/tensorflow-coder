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
"""Tests for operation_statistics.py."""

from absl.testing import absltest
from tf_coder.value_search import operation_statistics


class OperationStatisticsTest(absltest.TestCase):

  def test_update(self):
    statistics = operation_statistics.OperationStatistics()
    statistics.update('a', count=10, successes=1, time=1.5)
    statistics.update('b', count=80, successes=60, time=40.0)
    statistics.update('a', count=100, successes=10, time=8.5)
    statistics.update('c', count=0, successes=0, time=0.0)
    self.assertEqual(statistics.total_apply_count, 190)
    self.assertEqual(statistics.total_apply_successes, 71)
    self.assertEqual(statistics.operation_apply_time,
                     {'a': 10.0, 'b': 40.0, 'c': 0.0})
    self.assertEqual(statistics.operation_apply_count,
                     {'a': 110, 'b': 80, 'c': 0})
    self.assertEqual(statistics.operation_apply_successes,
                     {'a': 11, 'b': 60, 'c': 0})
    self.assertEqual(statistics.all_operation_names, {'a', 'b', 'c'})

  def test_get_total_time(self):
    statistics = operation_statistics.OperationStatistics()
    statistics.update('a', count=10, successes=1, time=1.5)
    statistics.update('b', count=80, successes=60, time=40.0)
    statistics.update('a', count=100, successes=10, time=8.5)
    statistics.update('c', count=0, successes=0, time=0.0)
    self.assertEqual(statistics.get_total_time(), 50.0)

  def test_statistics_as_string(self):
    statistics = operation_statistics.OperationStatistics()
    statistics.update('a', count=10, successes=1, time=1.5)
    statistics.update('b', count=80, successes=60, time=40.0)
    statistics.update('a', count=100, successes=10, time=8.5)
    statistics.update('c', count=0, successes=0, time=0.0)
    statistics.update('do_not_output', count=100, successes=10, time=1.23)

    result = statistics.statistics_as_string(operation_names=['a', 'b', 'c'],
                                             num_unique_values=77,
                                             elapsed_time=100.0)

    a_row = operation_statistics._ROW_FORMAT_STR.format(
        name='a', eps=11.0, sps=1.1, executions=110, successes=11, rate=0.1,
        time=10.0, time_frac=0.2)
    self.assertIn(a_row, result)
    b_row = operation_statistics._ROW_FORMAT_STR.format(
        name='b', eps=2.0, sps=1.5, executions=80, successes=60, rate=0.75,
        time=40.0, time_frac=0.8)
    self.assertIn(b_row, result)
    nan = float('NaN')
    c_row = operation_statistics._ROW_FORMAT_STR.format(
        name='c', eps=nan, sps=nan, executions=0, successes=0, rate=nan,
        time=0.0, time_frac=0.0)
    self.assertIn(c_row, result)
    self.assertNotIn('do_not_output', result)

    self.assertIn('Number of evaluations: 290', result)
    self.assertIn('Number of successful evaluations: 81', result)
    self.assertIn('Total time applying operations: 51.23 sec', result)
    self.assertIn('Number of unique values: 77', result)
    self.assertIn('Executions per second: 2.9', result)

  def test_statistics_as_string_sorted_by_time(self):
    statistics = operation_statistics.OperationStatistics()
    statistics.update('fast_op', count=10, successes=1, time=1.5)
    statistics.update('slow_op', count=10, successes=1, time=2.5)
    result = statistics.statistics_as_string(sort_by_time=False)
    self.assertLess(result.index('fast_op'), result.index('slow_op'))
    result = statistics.statistics_as_string(sort_by_time=True)
    self.assertLess(result.index('slow_op'), result.index('fast_op'))

if __name__ == '__main__':
  absltest.main()

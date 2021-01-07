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
"""Defines the OperationStatistics class.

An OperationStatistics instance tracks how often operations are applied, how
many of those applications are successful, and how long they take.
"""

import collections

# Format strings for the statistics table.
_HEADER_FORMAT_STR = ('{name:64}{eps:>12}{sps:>12}{executions:>13}'
                      '{successes:>12}{rate:>15}{time:>13}{time_frac:>11}')
_ROW_FORMAT_STR = ('{name:64}{eps:12.1f}{sps:12.1f}{executions:13}'
                   '{successes:12}{rate:15.2%}{time:13.3f}{time_frac:11.1%}')


def _nan_div(numerator, denominator):
  """Performs division, returning NaN if the denominator is zero."""
  return numerator / denominator if denominator != 0 else float('NaN')


class OperationStatistics(object):
  """Tracks statistics about applications of Operations.

  An OperationStatistics instance tracks how often Operation objects are
  applied, how many applications are successful, and how long they take.

  Attributes:
    total_apply_count: The total number of Operation applications.
    total_apply_successes: The total number of successful Operation
      applications.
    operation_apply_time: A dict mapping Operation names to the total time spent
      applying that Operation.
    operation_apply_count: A dict mapping Operation names to the number of
      applications of that Operation.
    operation_apply_successes: A dict mapping Operation names to the number of
      successful applications of that Operation.
    all_operation_names: A set of recorded Operation names.
  """

  def __init__(self):
    """Initializes the attributes."""
    self.total_apply_count = 0
    self.total_apply_successes = 0
    self.operation_apply_time = collections.defaultdict(float)
    self.operation_apply_count = collections.Counter()
    self.operation_apply_successes = collections.Counter()
    self.all_operation_names = set()

  def update(self, operation_name, count, successes, time):
    """Updates the statistics with the given statistics for one operation."""
    self.total_apply_count += count
    self.total_apply_successes += successes
    self.operation_apply_time[operation_name] += time
    self.operation_apply_count[operation_name] += count
    self.operation_apply_successes[operation_name] += successes
    self.all_operation_names.add(operation_name)

  def get_total_time(self):
    """Returns the total time spent applying operations."""
    return sum(self.operation_apply_time.values())

  def statistics_as_string(self, operation_names=None, num_unique_values=None,
                           elapsed_time=None, sort_by_time=True):
    """Returns a printable string with statistics for the given operations.

    Args:
      operation_names: A list of operation names. If None, all recorded
        operations will be logged.
      num_unique_values: The number of unique values found by value search. This
        is optional; if provided, it will be logged.
      elapsed_time: The total time used by value search (not just for Operation
        applications). This is optional; if provided, the operation applications
        per second will be logged.
      sort_by_time: Whether to sort the table by decreasing time. If False
        (default), then the table will be sorted by name alphabetically.
    """
    string_parts = []

    if operation_names is None:
      operation_names = list(self.all_operation_names)

    header = _HEADER_FORMAT_STR.format(
        name='Operation name', eps='Exec./sec', sps='Succ./sec',
        executions='Executions', successes='Successes', rate='Success rate',
        time='Time (sec)', time_frac='Time (%)')
    string_parts.append(header)
    string_parts.append('-' * len(header))
    total_time = sum(self.operation_apply_time[name]
                     for name in operation_names)
    if sort_by_time:
      sorted_names = sorted(operation_names,
                            key=lambda n: -self.operation_apply_time[n])
    else:
      sorted_names = sorted(operation_names)
    for name in sorted_names:
      apply_time = self.operation_apply_time[name]
      apply_count = self.operation_apply_count[name]
      apply_successes = self.operation_apply_successes[name]
      string_parts.append(_ROW_FORMAT_STR.format(
          name=name,
          eps=_nan_div(apply_count, apply_time),
          sps=_nan_div(apply_successes, apply_time),
          executions=apply_count,
          successes=apply_successes,
          rate=_nan_div(apply_successes, apply_count),
          time=apply_time,
          time_frac=_nan_div(apply_time, total_time)))

    string_parts.append('\nNumber of evaluations: {}\n'
                        'Number of successful evaluations: {}\n'
                        'Total time applying operations: {:.2f} sec\n'.format(
                            self.total_apply_count,
                            self.total_apply_successes,
                            self.get_total_time()))
    if num_unique_values is not None:
      string_parts.append('Number of unique values: {}'.format(
          num_unique_values))
    if elapsed_time is not None:
      string_parts.append('Executions per second: {:.1f}'.format(
          _nan_div(self.total_apply_count, elapsed_time)))

    return '\n'.join(string_parts)

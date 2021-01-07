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

"""Tests for operation_base.py."""

import timeit

from absl.testing import absltest
from tf_coder.value_search import filtered_values_cache
from tf_coder.value_search import operation_base
from tf_coder.value_search import operation_statistics
from tf_coder.value_search import value
from tf_coder.value_search import value_search_settings as settings_module


class StrangeAdditionOperation(operation_base.Operation):
  """A simple Operation subclass for testing purposes.

  This strange Operation can compute X + Y, but only if X < Y, and one of the
  following holds:

    * X is divisible by 2 and Y is divisible by 3, or
    * X is divisible by 4 and Y is divisible by 5.
  """

  def __init__(self):
    metadata = operation_base.OperationMetadata(docstring='test docstring')
    super(StrangeAdditionOperation, self).__init__(2, weight=5,
                                                   metadata=metadata)
    self.add_value_filters([lambda arg_value: arg_value.value % 2 == 0,
                            lambda arg_value: arg_value.value % 3 == 0])
    self.add_value_filters([lambda arg_value: arg_value.value % 4 == 0,
                            lambda arg_value: arg_value.value % 5 == 0])
    self.set_apply_filter(
        lambda arg_values: arg_values[0].value < arg_values[1].value)

  def _compute_name(self):
    return 'strange_addition'

  def apply(self, arg_values, settings):
    """See base class."""
    return value.OperationValue(arg_values[0].value + arg_values[1].value,
                                self, arg_values)

  def reconstruct_expression_from_strings(self, arg_strings):
    """See base class."""
    return '(' + arg_strings[0] + ' + ' + arg_strings[1] + ')'


def _value(wrapped_value):
  """A simple utility to create Value objects."""
  return value.ConstantValue(wrapped_value)


class OperationBaseTest(absltest.TestCase):

  def setUp(self):
    super(OperationBaseTest, self).setUp()
    self.operation = StrangeAdditionOperation()
    self.settings = settings_module.default_settings()

  def test_name(self):
    self.assertEqual(self.operation.name, 'strange_addition')
    self.assertEqual(self.operation.name, 'strange_addition')  # Cached.

  def test_metadata(self):
    self.assertEqual(self.operation.metadata.docstring, 'test docstring')

  def test_add_value_filters_raises_for_wrong_number_of_filters(self):
    with self.assertRaises(ValueError):
      self.operation.add_value_filters([lambda _: True])

  def test_apply(self):
    result_value = self.operation.apply([_value(2), _value(9)], self.settings)
    self.assertEqual(result_value.value, 11)

  def test_enumerate_values_with_weight(self):
    values_by_weight = [
        [],  # Weight 0.
        [_value(1), _value(4), _value(9), _value(15)],  # Weight 1.
        [_value(2), _value(6), _value(20), _value(60)],  # Weight 2.
        [_value(10), _value(12)],  # Weight 3.
    ]
    statistics = operation_statistics.OperationStatistics()

    filter_cache = filtered_values_cache.FilteredValuesCache()
    actual_results = self.operation.enumerate_values_with_weight(
        9, values_by_weight, filter_cache,
        end_time=float('inf'), settings=self.settings, statistics=statistics)

    # The operation itself has weight 5. We can get a total weight of 9 by
    # adding arguments of weight 1 and 3, or 2 and 2.
    expected_results = [
        # Weight 1 + weight 3.

        # X divisible by 2, Y divisible by 3, and X < Y.
        _value(16),  # 4 + 12.

        # X divisible by 4, Y divisible by 5, and X < Y.
        _value(14),  # 4 + 10.

        ################################################

        # Weight 2 + weight 2.

        # X divisible by 2, Y divisible by 3, and X < Y.
        _value(8),  # 2 + 6.
        _value(62),  # 2 + 60.
        _value(66),  # 6 + 60.
        _value(80),  # 20 + 60.

        # X divisible by 4, Y divisible by 5, and X < Y.
        _value(80),  # 20 + 60.

        ################################################

        # Weight 3 + weight 1.

        # X divisible by 2, Y divisible by 3, and X < Y.
        _value(25),  # 10 + 15.
        _value(27),  # 12 + 15.

        # X divisible by 4, Y divisible by 5, and X < Y.
        _value(27),  # 12 + 15.
    ]
    self.assertCountEqual(actual_results, expected_results)

    self.assertEqual(statistics.total_apply_count, 10)
    self.assertEqual(statistics.operation_apply_successes,
                     {'strange_addition': 10})

  def test_enumerate_values_with_weight_with_immediate_timeout(self):
    values_by_weight = [
        [],  # Weight 0.
        [_value(1), _value(4), _value(9), _value(15)],  # Weight 1.
        [_value(2), _value(6), _value(20), _value(60)],  # Weight 2.
        [_value(10), _value(12)],  # Weight 3.
    ]
    filter_cache = filtered_values_cache.FilteredValuesCache()
    # The time cutoff is before the function starts executing.
    actual_results = self.operation.enumerate_values_with_weight(
        9, values_by_weight, filter_cache,
        end_time=timeit.default_timer() - 0.1, settings=self.settings)
    self.assertEmpty(actual_results)

  def test_reconstruct_expression(self):
    left_term = self.operation.apply([_value(12), _value(34)], self.settings)
    middle_term = self.operation.apply([_value(5), _value(6)], self.settings)
    right_term = self.operation.apply([middle_term, _value(7)], self.settings)
    result_value = self.operation.apply([left_term, right_term], self.settings)
    self.assertEqual(result_value.reconstruct_expression(),
                     '((12 + 34) + ((5 + 6) + 7))')

if __name__ == '__main__':
  absltest.main()

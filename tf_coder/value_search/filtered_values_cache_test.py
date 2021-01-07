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
"""Tests for filtered_values_cache.py."""

import collections

from absl.testing import absltest
from tf_coder.value_search import filtered_values_cache
from tf_coder.value_search import value


def _value(wrapped_value):
  """A simple utility to create Value objects."""
  return value.ConstantValue(wrapped_value)


def _self_mapping(items):
  """Turns a list into an identity mapping dict."""
  return collections.OrderedDict(zip(items, items))


class FilteredValuesCacheTest(absltest.TestCase):

  def test_filter_values(self):
    filter_cache = filtered_values_cache.FilteredValuesCache()
    values_iterable = _self_mapping([_value(2), _value(3), _value(4)])
    even_filter = lambda arg_value: arg_value.value % 2 == 0

    self.assertEqual(
        filter_cache.filter_values(even_filter, 1, values_iterable),
        [_value(2), _value(4)])
    self.assertEqual(
        filter_cache.filter_values(None, 1, values_iterable),
        [_value(2), _value(3), _value(4)])

    # Check that the cache is used: even if the collection of values is
    # different, the cached results remain unchanged.
    self.assertEqual(
        filter_cache.filter_values(even_filter, 1, None),
        [_value(2), _value(4)])
    self.assertEqual(
        filter_cache.filter_values(None, 1, None),
        [_value(2), _value(3), _value(4)])


if __name__ == '__main__':
  absltest.main()

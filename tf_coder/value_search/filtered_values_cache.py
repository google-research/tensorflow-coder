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
"""A cache for operation filtering."""

import collections
from typing import Callable, Dict, Iterable, List, Optional

from tf_coder.value_search import value


class FilteredValuesCache(object):
  """A cache for operation filtering."""

  def __init__(self) -> None:
    # A map from filter functions and weights to associated values.
    self._cache = collections.defaultdict(dict)  # type: Dict[Callable[[value.Value], bool], Dict[int, List[value.Value]]]

  def filter_values(self,
                    filter_function: Optional[Callable[[value.Value], bool]],
                    weight: int,
                    values: Iterable[value.Value]) -> List[value.Value]:
    """Filters values using a filter function, caching the results.

    Args:
      filter_function: A filter function function (Value -> bool). If None, this
        means skip filtering, e.g., equivalent to filter_function being
        `lambda _: True`, but faster.
      weight: The weight of the candidate values. This will be used as part of
        the cache key.
      values: A list of values with the given weight, to be filtered.

    Returns:
      A list of values that pass the filter. The returned list should not be
      modified.
    """
    cached_weights = self._cache[filter_function]
    if weight in cached_weights:
      return cached_weights[weight]
    else:
      filtered_values = list(filter(filter_function, values)
                             if filter_function is not None
                             else values)
      cached_weights[weight] = filtered_values
      return filtered_values

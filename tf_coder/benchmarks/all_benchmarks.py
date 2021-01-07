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
"""Collects all known TF-Coder benchmarks."""

import inspect

from tf_coder.benchmarks import autopandas_benchmarks
from tf_coder.benchmarks import google_benchmarks
from tf_coder.benchmarks import simple_benchmarks
from tf_coder.benchmarks import stackoverflow_benchmarks


_ALL_BENCHMARK_MODULES = [google_benchmarks, simple_benchmarks,
                          stackoverflow_benchmarks, autopandas_benchmarks]


def all_benchmarks(include_ignored=False, modules=None):
  """Returns a list of all benchmarks.

  Args:
    include_ignored: A boolean indicating whether the search should include
      ignored benchmarks.
    modules: A list of module objects to inspect for benchmark functions. If
      None, all known relevant modules are used.

  Returns:
    A list of benchmark.Benchmark objects.
  """
  if modules is None:
    modules = _ALL_BENCHMARK_MODULES
  members = sum((inspect.getmembers(benchmark_module, inspect.isfunction)
                 for benchmark_module in modules), [])
  benchmark_list = []
  for unused_name, benchmark_function in members:
    benchmark = benchmark_function()
    if include_ignored or not benchmark.should_ignore:
      benchmark_list.append(benchmark)
  return benchmark_list


def find_benchmark_with_name(benchmark_name, include_ignored=False,
                             modules=None):
  """Returns a benchmark with the given name.

  Args:
    benchmark_name: A name (string) to search for.
    include_ignored: A boolean, used as described in all_benchmarks().
    modules: A list of module objects, used as described in all_benchmarks(). If
      None, all known relevant modules are used.

  Returns:
    A benchmark.Benchmark with the given name, if there is exactly one such
    benchmark. If there are zero or multiple such benchmarks, None is returned.
  """
  benchmark_list = all_benchmarks(include_ignored=include_ignored,
                                  modules=modules)
  matching_benchmarks = [benchmark
                         for benchmark in benchmark_list
                         if benchmark.name == benchmark_name]
  if len(matching_benchmarks) == 1:
    return matching_benchmarks[0]
  return None


def get_chosen_benchmarks(benchmark_name, include_ignored=False, modules=None):
  """Returns benchmarks according to the benchmark_name argument.

  Args:
    benchmark_name: The string name of a desired benchmark, or "ALL".
    include_ignored: A boolean, used as described in all_benchmarks().
    modules: A list of module objects, used as described in all_benchmarks(). If
      None, all known relevant modules are used.

  Returns:
    A list of benchmark.Benchmark objects.
  """
  if benchmark_name == 'ALL':
    return all_benchmarks(modules=modules)
  benchmark = find_benchmark_with_name(
      benchmark_name, include_ignored=include_ignored, modules=modules)
  if benchmark is None:
    return []
  return [benchmark]

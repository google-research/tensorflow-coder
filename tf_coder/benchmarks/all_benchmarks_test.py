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
"""Tests for all_benchmarks.py."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from tf_coder import tf_coder_utils
from tf_coder.benchmarks import all_benchmarks
from tf_coder.benchmarks import stackoverflow_benchmarks
from tf_coder.benchmarks import test_benchmarks
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search


def _parameterize_all_benchmarks():
  """Creates parameterized test cases for all benchmarks.

  This is useful so that tests can run on each benchmark individually, instead
  of having a single test loop over all benchmarks. In this way, issues with
  multiple benchmarks can be identified in one round of testing, and it is
  clearer which of the benchmarks need further attention.

  Returns:
    A list of tuples (test_case_name, benchmark, use_eager) for all benchmarks
    and both values of use_eager (True and False).
  """
  parameterized_tuples = []
  for index, benchmark in enumerate(all_benchmarks.all_benchmarks()):
    # The index ensures all test cases have distinct names, even if multiple
    # benchmarks have the same name.
    test_case_name = '{index}_{name}'.format(index=index, name=benchmark.name)
    parameterized_tuples.append((test_case_name, benchmark))
  return parameterized_tuples


class AllBenchmarksTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('including_ignored', True, ['test_add', 'test_cast',
                                   'inconsistent_target_program', 'test_add']),
      ('not_including_ignored', False, ['test_add', 'test_cast']))
  def test_all_benchmarks_finds_correct_functions(self, include_ignored,
                                                  expected_names):
    benchmarks = all_benchmarks.all_benchmarks(modules=[test_benchmarks],
                                               include_ignored=include_ignored)
    benchmark_names = [benchmark.name for benchmark in benchmarks]
    self.assertCountEqual(benchmark_names, expected_names)

  def test_all_benchmarks_have_unique_names(self):
    names = []
    for benchmark in all_benchmarks.all_benchmarks(include_ignored=False):
      name = benchmark.name
      self.assertIsNotNone(name)
      names.append(name)
    self.assertLen(set(names), len(names))

  def test_all_benchmarks_have_source(self):
    for benchmark in all_benchmarks.all_benchmarks(include_ignored=False):
      self.assertIsNotNone(benchmark.source)

  def test_all_benchmarks_have_description(self):
    for benchmark in all_benchmarks.all_benchmarks(include_ignored=False):
      if benchmark.name.startswith('autopandas_'):
        continue  # AutoPandas benchmarks don't have descriptions.
      self.assertTrue(benchmark.description)

  def test_all_benchmarks_have_reason_if_ignored(self):
    for benchmark in all_benchmarks.all_benchmarks(include_ignored=True):
      if benchmark.should_ignore:
        self.assertIsNotNone(benchmark.ignore_reason)

  @parameterized.named_parameters(
      ('zero_occurrences', 'nonexistent_name', True),
      ('one_occurrence', 'test_cast', False),
      ('two_occurrences', 'test_add', True))
  def test_find_benchmark_with_name(self, benchmark_name, expect_none):
    result = all_benchmarks.find_benchmark_with_name(
        benchmark_name, include_ignored=True, modules=[test_benchmarks])
    if expect_none:
      self.assertIsNone(result)
    else:
      self.assertIsNotNone(result)
      self.assertEqual(result.name, benchmark_name)

  def test_get_chosen_benchmarks_single_success(self):
    result = all_benchmarks.get_chosen_benchmarks(
        'test_add', include_ignored=False, modules=[test_benchmarks])
    self.assertLen(result, 1)
    self.assertEqual(result[0].name, 'test_add')

  def test_get_chosen_benchmarks_all_success(self):
    result = all_benchmarks.get_chosen_benchmarks(
        'ALL', modules=[test_benchmarks])
    self.assertGreater(len(result), 1)

  def test_get_chosen_benchmarks_duplicate_name(self):
    result = all_benchmarks.get_chosen_benchmarks(
        'test_add', include_ignored=True, modules=[test_benchmarks])
    self.assertEmpty(result)


class StackOverflowBenchmarksTest(absltest.TestCase):

  def test_all_stackoverflow_benchmarks_have_unique_source(self):
    all_sources = set()
    for benchmark in all_benchmarks.all_benchmarks(
        modules=[stackoverflow_benchmarks], include_ignored=False):
      self.assertNotIn(benchmark.source, all_sources)
      all_sources.add(benchmark.source)


class TargetProgramTest(parameterized.TestCase):

  def _check_target_program(self, benchmark):
    """Checks that a benchmark's target program is consistent with its examples.

    Args:
      benchmark: A Benchmark to verify.
    """
    self.assertIsNotNone(benchmark.target_program)

    for example in benchmark.examples:
      # Turn inputs into constant tensors and assign them to variables using a
      # new global namespace.
      global_namespace = {'tf': tf}
      input_names_to_objects = value_search._input_names_to_objects(
          example.inputs)
      for input_name, input_object in input_names_to_objects.items():
        input_value = value_module.InputValue(input_object, name='dummy_name')
        global_namespace[input_name] = input_value.value

      # Evaluate the target program, which uses the canonical variables.
      target_program_output = eval(benchmark.target_program, global_namespace)  # pylint: disable=eval-used

      # Check that the two outputs have equal string representation.
      expected_output = tf_coder_utils.convert_to_tensor(example.output)
      self.assertEqual(
          tf_coder_utils.object_to_string(expected_output),
          tf_coder_utils.object_to_string(target_program_output))

  def test_check_target_program_passes_for_good_benchmarks(self):
    good_benchmarks = [test_benchmarks.test_add(), test_benchmarks.test_cast()]
    for good_benchmark in good_benchmarks:
      self._check_target_program(good_benchmark)

  def test_check_target_program_fails_for_bad_benchmarks(self):
    bad_benchmark = test_benchmarks.inconsistent_target_program()
    with self.assertRaises(AssertionError):
      self._check_target_program(bad_benchmark)

  @parameterized.named_parameters(_parameterize_all_benchmarks())
  def test_all_benchmarks_have_correct_target_programs(self, benchmark):
    # Some benchmarks don't have target programs because we don't know how to
    # solve them ourselves.
    if benchmark.target_program is not None:
      self._check_target_program(benchmark)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)

  absltest.main()

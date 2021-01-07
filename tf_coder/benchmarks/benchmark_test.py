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
"""Tests for benchmark.py."""

from absl.testing import absltest
from absl.testing import parameterized
from tf_coder.benchmarks import benchmark
from tf_coder.benchmarks import test_benchmarks


class BenchmarkTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_examples', []),
      ('none_output', [benchmark.Example(inputs=[[1]], output=None)]),
      ('inconsistent_num_inputs', [
          benchmark.Example(inputs=[[1]], output=[1]),  # 1 input.
          benchmark.Example(inputs=[[1], [2]], output=[3]),  # 2 inputs.
      ]))
  def test_benchmark_init_raises(self, examples):
    with self.assertRaises(ValueError):
      benchmark.Benchmark(examples, name='test')

  def test_benchmark_init_computes_num_inputs(self):
    examples = [
        benchmark.Example(
            inputs=[
                [1],
                [2],
            ],
            output=[1, 2]
        ),
        benchmark.Example(
            inputs=[
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
            output=[[1, 2], [3, 4], [5, 6], [7, 8]]
        ),
    ]
    self.assertEqual(benchmark.Benchmark(examples, name='test').num_inputs, 2)

  def test_benchmark_init_assigns_name_if_not_given(self):
    benchmark_1 = benchmark.Benchmark([benchmark.Example(inputs=[1], output=1)])
    benchmark_2 = benchmark.Benchmark([benchmark.Example(inputs=[2], output=2)])
    self.assertEqual(benchmark_1.name, 'unnamed_benchmark_1')
    self.assertEqual(benchmark_2.name, 'unnamed_benchmark_2')

  def test_benchmark_not_ignored_by_default(self):
    not_ignored_list = [test_benchmarks.test_add(), test_benchmarks.test_cast()]
    for not_ignored in not_ignored_list:
      self.assertFalse(not_ignored.should_ignore)
      self.assertIsNone(not_ignored.ignore_reason)

  def test_ignore_decorator(self):
    ignored_benchmark = test_benchmarks.inconsistent_target_program()
    self.assertTrue(ignored_benchmark.should_ignore)
    self.assertIsNotNone(ignored_benchmark.ignore_reason)


if __name__ == '__main__':
  absltest.main()

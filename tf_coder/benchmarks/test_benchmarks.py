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
"""Benchmarks used only for testing purposes."""

from tf_coder.benchmarks import benchmark


def test_add():
  """Returns a simple benchmark.Benchmark for an addition task."""
  examples = [
      benchmark.Example(
          inputs=[
              [10],
              [20],
          ],
          output=[30],
      ),
  ]
  constants = [0]
  description = 'Add elementwise'
  target_program = 'tf.add(in1, in2)'
  source = 'test'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='test_add')


def test_cast():
  """Returns a simple benchmark.Benchmark for a casting task."""
  examples = [
      benchmark.Example(
          inputs=[
              [1, 0, 1, 1, 0],
          ],
          output=[True, False, True, True, False],
      ),
  ]
  constants = []
  description = 'cast a tensor'
  target_program = 'tf.cast(in1, tf.bool)'
  source = 'test'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='test_cast')


@benchmark.ignore('Target program is inconsistent with the example.')
def inconsistent_target_program():
  """Returns a benchmark.Benchmark with an inconsistent target program."""
  examples = [
      benchmark.Example(
          inputs=[
              [10],
              [20],
          ],
          output=[40],  # Should be 30.
      ),
  ]
  constants = [0]
  description = 'add elementwise'
  target_program = 'tf.add(in1, in2)'
  source = 'test'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='inconsistent_target_program')


@benchmark.ignore('Intentionally duplicate benchmark.')
def duplicate_test_add():
  """Returns a duplicate benchmark.Benchmark."""
  examples = [
      benchmark.Example(
          inputs=[
              [10],
              [20],
          ],
          output=[30],
      ),
  ]
  constants = [0]
  description = 'Add elementwise'
  target_program = 'tf.add(in1, in2)'
  source = 'test'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='test_add')

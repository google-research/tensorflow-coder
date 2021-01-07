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
"""A Benchmark class for TF-Coder."""

import collections


# A list of input tensors along with the expected output tensor.
Example = collections.namedtuple('Example', ['inputs', 'output'])


# The number of unnamed Benchmark objects created so far (a global variable).
#
# This is used to assign distinct names to unnamed benchmarks. However, the
# assigned name can clash with another explicitly-provided name. Unique names
# are preferable for easier testing and debugging. However, uniqueness is not
# required or enforceable, since any name can be provided to the constructor.
_num_unnamed_benchmarks = 0


class Benchmark(object):
  """A class for specifying the benchmarks (example problems) for TF-Coder.

  Attributes:
    examples: A nonempty list of Example namedtuples.
    constants: A list of constants.
    description: A natural-language description of the task.
    target_program: A string representing the desired program, if known.
    source: A string describing the source of the problem, preferably a
      hyperlink if available.
    name: A name (string) for the benchmark.
    num_inputs: The number of inputs in each example.
    should_ignore: A boolean indicating whether the benchmark should be ignored,
      perhaps because it is unsuitable or unreasonable for synthesis.
    ignore_reason: A string describing the reason for ignoring the benchmark, if
      should_ignore is True.
  """

  def __init__(self, examples, constants=None, description=None,
               target_program=None, source=None, name=None):
    """Initializes a Benchmark.

    Args:
      examples: A nonempty list of user-provided Example namedtuples.
      constants: A list of user-provided constants.
      description: A natural-language description of the task.
      target_program: A string representing the desired program, if known.
      source: A string describing the source of the problem, preferably a
        hyperlink if available.
      name: A name (string) for the benchmark. If not provided, a name will be
        assigned.

    Raises:
      ValueError: If there are no examples, or an example has no output, or the
      examples have inconsistent numbers of inputs.
    """
    if not examples:
      raise ValueError('A benchmark must have at least 1 example.')

    self.examples = examples
    self.constants = [] if constants is None else constants
    self.description = '' if description is None else description
    self.target_program = target_program
    self.source = source

    if name is None:
      global _num_unnamed_benchmarks
      _num_unnamed_benchmarks += 1
      self.name = 'unnamed_benchmark_' + str(_num_unnamed_benchmarks)
    else:
      self.name = name

    # Examples must have the same number of inputs and non-None output.
    self.num_inputs = len(examples[0].inputs)
    for example in examples:
      if example.output is None:
        raise ValueError('Examples must have non-None output.')
      if len(example.inputs) != self.num_inputs:
        raise ValueError('The examples in a bechmark must have the same number '
                         'of inputs.')

    # Set by the @ignore decorator.
    self.should_ignore = False
    self.ignore_reason = None


def ignore(ignore_reason):
  """A decorator to be applied to benchmark-creating functions.

  The benchmark returned by the decorated function will have should_ignore set
  to True and ignore_reason set as the decorator's argument.

  Args:
    ignore_reason: A string describing the reason for ignoring the benchmark.

  Returns:
    A decorator that adds the supplied ignore_reason to the benchmark.
  """
  def add_ignore_reason(benchmark_function):
    def wrapper():
      benchmark = benchmark_function()
      benchmark.should_ignore = True
      benchmark.ignore_reason = ignore_reason
      return benchmark
    return wrapper
  return add_ignore_reason

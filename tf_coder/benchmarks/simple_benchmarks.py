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
"""Simple benchmarks that TF-Coder should always solve quickly."""

# Avoid wrapping URLs and target programs to ease clicking and copying.
# pylint: disable=line-too-long

import tensorflow as tf
from tf_coder.benchmarks import benchmark


def simple_broadcasted_add():
  """Adding two tensors with broadcasting."""
  examples = [
      benchmark.Example(
          inputs=[
              [3, 4, 5],
              [10, 20, 30],
          ],
          output=[[13, 14, 15], [23, 24, 25], [33, 34, 35]],
      ),
  ]
  constants = []
  description = 'Add two tensors with broadcasting'
  target_program = 'tf.add(in1, tf.expand_dims(in2, 1))'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_broadcasted_add')


def simple_with_input_names():
  """A benchmark using names for input tensors."""
  examples = [
      benchmark.Example(
          inputs={
              'tensor_x': [3, 4, 5],
              'tensor_y': [10, 20, 30],
          },
          output=[[13, 14, 15], [23, 24, 25], [33, 34, 35]],
      ),
  ]
  constants = []
  description = 'Add two tensors with broadcasting'
  target_program = 'tf.add(tensor_x, tf.expand_dims(tensor_y, 1))'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_with_input_names')


def simple_cast():
  """Casting an int tensor into a float tensor."""
  examples = [
      benchmark.Example(
          inputs=[
              [12, 34, 56]
          ],
          output=[12.0, 34.0, 56.0],
      ),
  ]
  constants = []
  description = 'Cast an int tensor into a float tensor'
  target_program = 'tf.cast(in1, tf.float32)'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_cast')


def simple_index():
  """Indexing within a tensor."""
  examples = [
      benchmark.Example(
          inputs=[
              [12, 34, 56, 78],
              -2,
          ],
          output=56,
      ),
  ]
  constants = []
  description = 'Index into a tensor'
  target_program = 'in1[in2]'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_index')


def simple_slice():
  """Slicing within a tensor."""
  examples = [
      benchmark.Example(
          inputs=[
              [[12, 34, 56, 78], [-1, -2, -3, -4]],
              -1,
          ],
          output=[[34, 56], [-2, -3]],
      ),
  ]
  constants = []
  description = 'Slice a tensor'
  target_program = 'in1[:, 1:in2]'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_slice')


def simple_sparse_add():
  """Adding a sparse and a dense tensor to produce a sparse tensor."""
  examples = [
      benchmark.Example(
          inputs=[
              tf.SparseTensor(indices=[[0, 0], [0, 1]],
                              values=[12, 34],
                              dense_shape=[2, 2]),
              [[-3, 0], [-5, 0]],
          ],
          output=tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]],
                                 values=[9, 34, -5],
                                 dense_shape=[2, 2]),
      ),
  ]
  constants = []
  description = 'Add sparse tensor with dense tensor'
  target_program = 'tf.sparse.add(in1, tf.sparse.from_dense(in2))'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_sparse_add')


def simple_add_big_tensors():
  """Broadcasted addition of "big" tensors."""
  examples = [
      benchmark.Example(
          inputs=[
              list(range(100)),
              [1000, 2000, 3000, 4000, 5000],
          ],
          output=[list(range(start, start + 100))
                  for start in range(1000, 5001, 1000)]
      ),
  ]
  constants = []
  description = 'Add two tensors'
  target_program = 'tf.add(in1, tf.expand_dims(in2, 1))'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_add_big_tensors')


def simple_using_constant():
  """A benchmark that requires a user-provided constant."""
  examples = [
      benchmark.Example(
          inputs=[
              [1, 2, 3],
          ],
          output=[101, 102, 103]
      ),
  ]
  constants = [100]
  description = 'Add 100 to every element'
  target_program = 'tf.add(in1, tf.constant(100))'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_using_constant')


def simple_using_output_shape():
  """A benchmark that requires a constant obtained from the output shape."""
  examples = [
      benchmark.Example(
          inputs=[
              tf.constant(7.0),
          ],
          output=[[7.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 7.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 7.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 7.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 7.0]]
      ),
  ]
  constants = []
  description = 'Multiply with the identity matrix'
  target_program = 'tf.multiply(in1, tf.eye(5))'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_using_output_shape')


def simple_using_output_shape_tuple():
  """A benchmark that requires the output shape as a constant."""
  examples = [
      benchmark.Example(
          inputs=[],
          output=tf.zeros(shape=(2, 3, 4, 5)),
      ),
  ]
  constants = []
  description = 'Construct a 4D zeros tensor'
  target_program = 'tf.zeros(shape=(2, 3, 4, 5))'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_using_output_shape_tuple')


def simple_using_boolean_constant():
  """A benchmark that requires using a boolean constant."""
  examples = [
      benchmark.Example(
          inputs=[
              tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 1]],
                              values=[12, 34, 56],
                              dense_shape=[2, 2]),
          ],
          output=tf.SparseTensor(indices=[[0], [1]],
                                 values=[46, 56],
                                 dense_shape=[2]),
      ),
  ]
  constants = []
  description = 'Reduce sum on a sparse tensor'
  target_program = 'tf.sparse.reduce_sum(in1, axis=1, output_is_sparse=True)'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_using_boolean_constant')


def simple_using_constant_kwarg():
  """A benchmark that requires using a constant kwarg."""
  examples = [
      benchmark.Example(
          inputs=[
              [40, 20, 60, 50, 10, 20, 50, 50],
          ],
          output=[2, 3, 6, 7, 0, 1, 5, 4],
      ),
  ]
  constants = []
  description = 'Indices sorted in reverse order'
  target_program = "tf.argsort(in1, axis=0, direction='DESCENDING', stable=True)"
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_using_constant_kwarg')


def simple_using_primitive_input():
  """A benchmark with a Python primitive as an input."""
  examples = [
      benchmark.Example(
          inputs=[
              123,
              tf.constant(45),
          ],
          output=tf.constant(168),
      ),
  ]
  constants = []
  description = 'Add primitive int and scalar int tensor'
  target_program = 'tf.add(tf.constant(in1), in2)'
  source = 'handwritten task'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='simple_using_primitive_input')

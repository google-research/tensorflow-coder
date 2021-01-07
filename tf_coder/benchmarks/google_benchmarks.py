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
"""Benchmarks from real questions and tasks encountered by Googlers."""

# Avoid wrapping URLs and target programs to ease clicking and copying.
# pylint: disable=line-too-long

# Every function in this module takes no arguments and creates a benchmark.
# pylint: disable=missing-docstring

import tensorflow as tf
from tf_coder.benchmarks import benchmark


def google_01():
  examples = [
      benchmark.Example(
          inputs=[
              [0, 0, 0, 1, 3, 3],
          ],
          output=[[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [3, 1]]
      ),
  ]
  constants = []
  description = 'Convert index tensor into pairs for SparseTensor indexing'
  target_program = 'tf.cast(tf.where(tf.sequence_mask(tf.math.bincount(in1))), tf.int32)'
  source = 'From an internal Google chat room, 09/07/2018'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_01')


def google_02():
  examples = [
      benchmark.Example(
          inputs=[
              [[0.0, 1.0, 0.0, 0.0],
               [0.0, 1.0, 1.0, 0.0],
               [1.0, 1.0, 1.0, 1.0]],
          ],
          output=[[0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.5, 0.5, 0.0],
                  [0.25, 0.25, 0.25, 0.25]]
      ),
  ]
  constants = []
  description = 'Divide each row by the sum of that row'
  target_program = 'tf.divide(in1, tf.expand_dims(tf.reduce_sum(in1, axis=1), 1))'
  source = 'Real task encountered by Googler, 11/01/2018'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_02')


def google_03():
  examples = [
      benchmark.Example(
          inputs=[
              tf.SparseTensor(
                  indices=[[0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 2]],
                  values=[1., 1., 1., 1.],
                  dense_shape=[2, 2, 800]),
          ],
          output=tf.SparseTensor(
              indices=[[0, 0, 0], [0, 1, 1]],
              values=[1., 1.],
              dense_shape=[1, 2, 800])
      ),
  ]
  constants = []
  description = 'Slice the first dimension of a SparseTensor'

  # Original approach, which can't be written as one expression.
  """
  reduced_sp_tensor = tf.sparse_retain(in1, tf.equal(in1.indices[:, 0], 0))
  output = tf.SparseTensor(indices=reduced_sp_tensor.indices,
                           values=reduced_sp_tensor.values,
                           dense_shape=tf.concat([tf.ones(1, dtype=tf.int64),
                                                  in1.dense_shape[1:]], 0))
  """  # pylint: disable=pointless-string-statement

  target_program = 'tf.sparse.slice(in1, tf.zeros(3, dtype=tf.int64), tf.concat([[1], in1.dense_shape[1:]], 0))'
  source = 'Real task encountered by Googler, 10/16/2018'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_03')


def google_04():
  examples = [
      benchmark.Example(
          inputs=[
              [111, 112, 121, 122, 131, 132, 211, 212, 221, 222, 231, 232,
               311, 312, 321, 322, 331, 332, 411, 412, 421, 422, 431, 432],
          ],
          output=[[[111, 112], [121, 122], [131, 132]],
                  [[211, 212], [221, 222], [231, 232]],
                  [[311, 312], [321, 322], [331, 332]],
                  [[411, 412], [421, 422], [431, 432]]]
      ),
  ]
  constants = []
  description = 'Reshape a flat array into a rank 3 tensor'
  target_program = 'tf.reshape(in1, shape=(4, 3, 2))'
  source = 'Real task encountered by Googler, 3/21/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_04')


def google_05():
  examples = [
      benchmark.Example(
          inputs=[
              [[1, 2, 3, 4],
               [5, 6, 7, 8]],
          ],
          output=[[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                   [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                  [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8],
                   [5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]]]
      ),
  ]
  constants = [6]
  description = 'Repeat each input entry 6 times'
  target_program = 'tf.tile(tf.expand_dims(in1, 1), (1, 6, 1))'
  source = 'Real task encountered by Googler, 3/22/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_05')


def google_06():
  examples = [
      benchmark.Example(
          inputs=[
              [0, 1, 1, 2, 3, 3, 3, 3],
              [1, 3, 4, 5, 10, 8, 9, 4],
          ],
          output=[1, 4, 5, 10]
      ),
  ]
  constants = []
  description = 'Take the max from each group of elements'
  target_program = 'tf.math.segment_max(in2, in1)'
  source = 'Real task encountered by Googler, 3/28/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_06')


def google_07():
  examples = [
      benchmark.Example(
          inputs=[
              [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 5, 5],
              [4, 1, 8, 2, 5, 7, 9, 3, 7, 3, 1, 42, 1, 2, 4, 0],
          ],
          output=[2, 3, 1, 0, 1, 0]
      ),
  ]
  constants = []
  description = 'Take the argmax of each group of elements'
  target_program = 'tf.cast(tf.argmax((tf.sequence_mask(tf.cumsum(tf.math.bincount(in1)), dtype=tf.int32) - tf.sequence_mask(tf.cumsum(tf.math.bincount(in1), exclusive=True), dtype=tf.int32, maxlen=in1.shape[0])) * in2, axis=1), tf.int32) - tf.cumsum(tf.math.bincount(in1), exclusive=True)'
  # Original question asker's solution below.
  """
  segment_lens = tf.math.bincount(in1)
  segment_ends = tf.scan(lambda a, x: a + x, segment_lens)
  segment_infos = tf.cast(tf.transpose(tf.stack([segment_lens, segment_ends])), dtype=tf.int64)
  def segment_argmax_fn(r):
    return tf.argmax(in2[r[1] - r[0]:r[1]])
  result = tf.cast(tf.map_fn(segment_argmax_fn, segment_infos), dtype=tf.int32)
  """  # pylint: disable=pointless-string-statement
  # Another Googler's solution below. Only works if in2 is positive.
  """
  segment_lens = tf.math.bincount(in1)
  cumulative_inclusive = tf.cumsum(segment_lens)
  cumulative_exclusive = tf.cumsum(segment_lens, exclusive=True)
  mask_inclusive = tf.sequence_mask(cumulative_inclusive, dtype=tf.int32)
  mask_exclusive = tf.sequence_mask(cumulative_exclusive, dtype=tf.int32, maxlen=in1.shape[0])
  masked_data = (mask_inclusive - mask_exclusive) * in2
  output = tf.cast(tf.argmax(masked_data, axis=1), tf.int32) - cumulative_exclusive
  """  # pylint: disable=pointless-string-statement
  source = 'Real task encountered by Googler, 3/29/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_07')


def google_08():
  examples = [
      benchmark.Example(
          inputs=[
              [3, 4, 2, 1],
          ],
          output=[[1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 0],
                  [1, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0]]
      ),
  ]
  constants = [5]
  description = 'create a mask for sequences of the given lengths'
  target_program = 'tf.cast(tf.greater(tf.expand_dims(in1, 1), tf.range(5)), tf.int32)'
  source = 'Real task encountered by Googler, 4/11/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_08')


def google_09():
  examples = [
      benchmark.Example(
          inputs=[
              [1, 1, 1, 0, 0, 2],
              [10, 20, 30, 14, 15, 26],
          ],
          output=[14, 15, 10, 20, 30, 26]
      ),
  ]
  constants = []
  description = 'sort the segments'
  target_program = 'tf.gather(in2, tf.argsort(in1, stable=True))'
  source = 'Real task encountered by Googler, 8/9/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_09')


def google_10():
  examples = [
      benchmark.Example(
          inputs=[
              [10, 20, 0, 40, 0, 30],
              [1, 1, 0, 1, 0, 1],
          ],
          output=[10, 20, 40, 30]
      ),
  ]
  constants = []
  description = 'gather the marked elements'
  target_program = 'tf.boolean_mask(in1, tf.cast(in2, tf.bool))'
  source = ('Proposed by Googler at an internal demo on 8/13/2019, '
            'simplified slightly')
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_10')


def google_11():
  examples = [
      benchmark.Example(
          inputs=[
              [[1.0, 0.3, -4.2, 0.0, 2.1, 0.4],
               [-0.1, 0.0, 1.4, -1.0, 0.4, 0.0],
               [0.1, 0.0, 0.7, -0.3, 0.5, -0.1],
               [1.4, 2.5, 0.3, 0.01, 0.0, 1.2]],
          ],
          output=[4, 2, 3, 5]
      ),
  ]
  constants = []
  description = 'count the number of elements greater than 0 in each row'
  target_program = 'tf.reduce_sum(tf.cast(tf.greater(in1, 0), tf.int32), axis=1)'
  source = 'Real task encountered by Googler, 8/26/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_11')


def google_12():
  examples = [
      benchmark.Example(
          inputs=[
              [[1.0, 0.3, -4.2, 0.0, 2.1],
               [-0.1, 0.0, 1.4, -1.0, 0.4],
               [0.1, 0.0, 0.7, -0.3, 0.5],
               [1.4, 2.5, 0.3, -0.1, 0.0]],
          ],
          output=[[1, 1, 0, 1, 0],
                  [0, 1, 0, 0, 1],
                  [1, 1, 1, 0, 1],
                  [0, 0, 1, 0, 1]]
      ),
  ]
  constants = []
  description = 'identify elements between 0 and 1'
  target_program = 'tf.cast(tf.logical_and(0 <= in1, in1 <= 1), tf.int32)'
  source = 'Real task encountered by Googler, 8/26/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_12')


def google_13():
  examples = [
      benchmark.Example(
          inputs=[
              [[1, 2], [10, 20]],
              [[3, 4, 5], [30, 40, 50]],
          ],
          output=[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
      ),
  ]
  constants = []
  description = 'Concatenate batches of sequences'
  target_program = 'tf.concat([in1, in2], axis=1)'
  source = 'Real task encountered by Googler, 9/13/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_13')


def google_14():
  examples = [
      benchmark.Example(
          inputs=[
              [[1, 3, 2, 0, 0], [4, 6, 5, 0, 0], [8, 7, 9, 0, 0]],
          ],
          output=[[0, 1, 3, 2, 0], [0, 4, 6, 5, 0], [0, 8, 7, 9, 0]]
      ),
  ]
  constants = []
  description = 'circular buffer'
  target_program = 'tf.roll(in1, 1, 1)'
  source = 'From internal Colab, 10/17/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_14')


def google_15():
  examples = [
      benchmark.Example(
          inputs=[
              [[1, 3, 5, 7], [2, 4, 6, 8]],
          ],
          output=[[1, 3, 5, 7, 0], [2, 4, 6, 8, 0]]
      ),
  ]
  constants = []
  description = 'pad a zero column'
  target_program = "tf.pad(in1, [[0, 0], [0, 1]], 'CONSTANT')"
  source = 'Real task encountered by Googler, 10/23/2019'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_15')


def google_16():
  examples = [
      benchmark.Example(
          inputs=[
              [1, 2, 0, 3],
              [2, 1, 2, 3],
          ],
          output=[1, 1, 2, 0, 0, 3, 3, 3]
      ),
  ]
  constants = []
  description = 'replicate elements a given number of times'
  target_program = 'tf.gather(in1, tf.where(tf.sequence_mask(in2))[:, 0])'
  source = 'From an internal Google forum'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_16')


def google_17():
  examples = [
      benchmark.Example(
          inputs=[
              [True, False, False, True, False],
              [1, 2, 3, 4, 5],
          ],
          output=[1, -20, -30, 4, -50]
      ),
  ]
  constants = [-10]
  description = 'use bool tensor as condition'
  target_program = 'tf.where(in1, in2, tf.multiply(in2, -10))'
  source = 'From an internal Google forum'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_17')


def google_18():
  examples = [
      benchmark.Example(
          inputs=[
              [5, 7, -12, 10, 20],
              [1, 2, 3, 1, 2],
          ],
          output=[15, 27, -12, 15, 27]
      ),
  ]
  constants = []
  description = ('sum of elements in the first tensor but partitioned by the '
                 'second tensor')
  target_program = 'tf.linalg.matvec(tf.cast(tf.equal(in2[:, None], in2), tf.int32), in1)'
  source = 'From an internal Google forum'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_18')


def google_19():
  examples = [
      benchmark.Example(
          inputs=[
              [[11, 12, 13],
               [30, 20, 10],
               [77, 88, 99]],
              [[2, 0, 1],
               [1, 0, 2],
               [0, 2, 1]],
          ],
          output=[[12, 13, 11],
                  [20, 30, 10],
                  [77, 99, 88]]
      ),
  ]
  constants = []
  description = 'scatter a 2-D tensor with indices'
  target_program = 'tf.gather(in1, tf.argsort(in2, axis=1), batch_dims=1)'
  source = 'From an internal Google forum'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_19')


def google_20():
  examples = [
      benchmark.Example(
          inputs=[
              [10, 7, 4, 3, 2, 8],
          ],
          output=[5, 3, 2, 1, 0, 4]
      ),
  ]
  constants = []
  description = 'sort a tensor and return sorted index in original order'
  target_program = 'tf.cast(tf.argsort(tf.argsort(in1)), tf.int32)'
  source = 'From an internal Google forum'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_20')


def google_21():
  examples = [
      benchmark.Example(
          inputs={
              'tensor': [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10, 11]],
              'indices': [[0, 0], [0, 2], [1, 1], [1, 3], [2, 2], [2, 4]],
              'updates': [[0, -2], [-1, -3], [-2, -4]],
          },
          output=[[0, 2, -2, 4, 5], [4, -1, 6, -3, 8], [7, 8, -2, 10, -4]]
      ),
  ]
  constants = []
  description = 'update a tensor at the given indices'
  target_program = 'tf.tensor_scatter_nd_update(tensor, indices, tf.reshape(updates, (-1,)))'
  source = 'Real task encountered by Googler, 12/15/2020'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_21')


def google_22():
  examples = [
      benchmark.Example(
          inputs=[
              [[0, 2], [1, 3], [2, 4]],
          ],
          output=[[0, 0], [0, 2], [1, 1], [1, 3], [2, 2], [2, 4]]
      ),
  ]
  constants = []
  description = 'pair with row index'
  target_program = 'tf.cast(tf.where(tf.reduce_max(tf.one_hot(in1, tf.reduce_max(in1) + 1), axis=1)), tf.int32)'
  source = 'Real task encountered by Googler, 12/15/2020'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_22')


# A template for easy copy/pasting. Copying an existing benchmark and replacing
# parts of it will lead to a state where the benchmark is half-correct, but not
# obviously so. Copy this template instead when creating new benchmarks.
"""

def google_NUMBER():
  examples = [
      benchmark.Example(
          inputs=[
              INPUT_1,
              INPUT_2,
          ],
          output=OUTPUT
      ),
  ]
  constants = [CONSTANTS]
  description = 'DESCRIPTION'
  target_program = 'SOLUTION_PROGRAM'
  source = 'PROBLEM_SOURCE'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='google_NUMBER')

"""  # pylint: disable=pointless-string-statement

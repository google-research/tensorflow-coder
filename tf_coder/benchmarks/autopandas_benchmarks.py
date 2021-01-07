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
"""Benchmarks adapted from AutoPandas's benchmarks.

Autopandas benchmarks are at:
https://github.com/rbavishi/autopandas/blob/master/autopandas_v2/evaluation/benchmarks/stackoverflow.py
"""

# Avoid wrapping URLs and target programs to ease clicking and copying.
# pylint: disable=line-too-long

# Every function in this module takes no arguments and creates a benchmark.
# pylint: disable=missing-docstring,g-doc-return-or-yield

from tf_coder.benchmarks import benchmark


def autopandas_01():
  # Turned the desired indices [0, 2, 4] into an input.
  examples = [
      benchmark.Example(
          inputs=[
              [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
              [0, 2, 4],
          ],
          output=[[5, 7, 9], [10, 12, 14]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.gather(in1, in2, axis=1)'
  source = 'SO_11881165_depth1'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_01')


def autopandas_02():
  """AutoPandas benchmark.

  The DataFrame input:
                 value1  value2
  group1 group2
  a      c          1.1     7.1
         c          2.0     8.0
         d          3.0     9.0
  b      d          4.0    10.0
         d          5.0    11.0
         e          6.0    12.0

  The DataFrame output:
          value1  value2
  group2
  c          1.1     7.1
  c          2.0     8.0
  d          3.0     9.0

  Notice that "c", "d", and "e" are just treated as data, so we'll replace them
  with 1, 2, and 3, respectively. "group1" acts as a third dimension, so our
  input tensor will be 3D. We'll also make "group1" the innermost axis, to make
  the problem not super trivial in TensorFlow.

  Finally, I changed some numbers to rule out tf.reduce_min(axis=2).
  """
  examples = [
      benchmark.Example(
          inputs=[
              [[[1, 2], [1, 2], [2, 3]],  # group2
               [[1.1, 4], [2, 1], [3, 6]],  # value1
               [[7.1, 10], [8, 11], [9, 7]]],  # value2
          ],
          output=[[1, 1, 2],
                  [1.1, 2, 3],
                  [7.1, 8, 9]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  # Note that we don't have ops for indexing/slicing axis 2. But, TF-Coder will
  # find a workaround.
  target_program = 'in1[:, :, 0]'
  source = 'SO_11941492_depth1'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_02')


def autopandas_03():
  # The "series" and "step" columns only serve to identify where the data should
  # move to. The number of unique "series" and "step" values determine the
  # output's shape, so we provide them (3 and 5) as constants. The real data is
  # in the "value" column, so it's a 1D tensor. This is simply a tensor reshape
  # and transpose.
  examples = [
      benchmark.Example(
          inputs=[
              [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
               1011, 1012, 1013, 1014],
          ],
          output=[[1000, 1003, 1006, 1009, 1012],
                  [1001, 1004, 1007, 1010, 1013],
                  [1002, 1005, 1008, 1011, 1014]]
      ),
  ]
  constants = [3, 5]
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.transpose(tf.reshape(in1, (5, 3)))'
  source = 'SO_13647222_depth1'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_03')


def autopandas_04():
  # This problem boils down to "select rows of the table where row['line_race']
  # is nonzero". We use a simpler example to describe the same problem idea.
  examples = [
      benchmark.Example(
          inputs=[
              [[1, 2, 3, 4, 5],
               [9, 8, 7, 6, 5],
               [3, 0, 2, 5, 8],
               [8, 8, 6, 3, 2],
               [2, 0, 7, 7, 3],
               [9, 0, 3, 2, 7],
               [1, 3, 8, 9, 4]],
          ],
          output=[[1, 2, 3, 4, 5],
                  [9, 8, 7, 6, 5],
                  [8, 8, 6, 3, 2],
                  [1, 3, 8, 9, 4]],
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.boolean_mask(in1, tf.cast(in1[:, 1], tf.bool))'
  source = 'SO_18172851_depth1'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_04')


def autopandas_05():
  # The problem is to sort a table using values in one column. Their code that
  # constructs the output actually sorts by 3 columns, but their example has
  # unique values for the first column, so the other two columns don't affect
  # the result at all. Instead of having dates and strings and numbers, we just
  # use numbers. Sort by column 0 in increasing order.
  examples = [
      benchmark.Example(
          inputs=[
              [[6, 3, 7, 8, 4],
               [8, 9, 4, 5, 3],
               [1, 5, 3, 6, 9],
               [2, 1, 4, 3, 2],
               [7, 9, 6, 2, 7],
               [5, 8, 0, 4, 2]],
          ],
          output=[[1, 5, 3, 6, 9],
                  [2, 1, 4, 3, 2],
                  [5, 8, 0, 4, 2],
                  [6, 3, 7, 8, 4],
                  [7, 9, 6, 2, 7],
                  [8, 9, 4, 5, 3]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.gather(in1, tf.argsort(in1[:, 0], stable=True))'
  source = 'SO_49583055_depth1'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_05')


# SO_49592930_depth1 is out of scope. It involves taking the union of two
# mappings from keys to values, where one such mapping takes precedence over the
# other in case of overlapping keys. TensorFlow is not designed to handle
# mappings, and I (Kensen) cannot think of a way to do this using TensorFlow.

# SO_49572546_depth1 is exactly the same as the one above, except for the
# ordering of two inputs and the data. It's out of scope for the same reason.


@benchmark.ignore("This isn't in AutoPandas's table of results, idk why")
def autopandas_06_ignored():
  """AutoPandas benchmark.

  The input DataFrame:
      X   Y   Z
  4  X1  Y2  Z3
  5  X1  Y1  Z1
  6  X1  Y1  Z1
  7  X1  Y1  Z2

  The output DataFrame:
  Z    Z1   Z2   Z3
  Y
  Y1  1.0  1.0  NaN
  Y2  NaN  NaN  1.0

  Basically, the Ys are row indices and the Zs are column indices. The X1 does
  not really matter, we just want a boolean output table (1.0=True, NaN=False)
  that identifies which Y and Z pairs appeared in the input table.

  Their example is tiny so I expanded it.

  This task also doesn't appear in their table of results?
  """
  examples = [
      benchmark.Example(
          inputs=[
              # Y  Z
              [[1, 2],
               [0, 0],
               [0, 0],
               [0, 1],
               [4, 2],
               [4, 3],
               [4, 2],
               [2, 1]],
          ],
          output=[[True, True, False, False],
                  [False, False, True, False],
                  [False, True, False, False],
                  [False, False, False, False],
                  [False, False, True, True]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  # TODO(kshi): Note that tf.scatter_nd is not yet supported by TF-Coder!!
  target_program = 'tf.cast(tf.scatter_nd(in1, updates=tf.ones(8), shape=[5, 4]), tf.bool)'
  source = 'SO_12860421_depth1'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_06_ignored')


def autopandas_06():
  # This is very similar to autopandas_03, they're both pivot tables.
  examples = [
      benchmark.Example(
          inputs=[
              [4, 5, 6, 7],
          ],
          output=[[4, 5], [6, 7]]
      ),
  ]
  constants = [2]
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.reshape(in1, (2, 2))'
  source = 'SO_13261175_depth1'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_06')


# SO_13793321_depth1 is out of scope. It's merging two tables, keeping the rows
# with matching values in a particular column. This isn't something TensorFlow
# was designed for.


def autopandas_07():
  # This is exactly the same as autopandas_05 (sort rows based on a column), but
  # with different data. Like in the other task, their code for creating the
  # output actually sorts using multiple columns, but their data contains unique
  # values. I'm just going to mash my keyboard again to get different data.
  examples = [
      benchmark.Example(
          inputs=[
              [[8, 5, 9, 3],
               [3, 6, 6, 8],
               [1, 2, 3, 4],
               [7, 6, 5, 4],
               [4, 7, 2, 6]]
          ],
          output=[[1, 2, 3, 4],
                  [3, 6, 6, 8],
                  [4, 7, 2, 6],
                  [7, 6, 5, 4],
                  [8, 5, 9, 3]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.gather(in1, tf.argsort(in1[:, 0], stable=True))'
  source = 'SO_14085517_depth1'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_07')


def autopandas_08():
  # I added 3 extra rows so that the solution isn't just slicing, and chose the
  # numbers to test the boundary of the condition row[0] > 1. The boundary (1)
  # is given as a constant.
  examples = [
      benchmark.Example(
          inputs=[
              [[5, 7], [6, 8], [-1, 9], [-2, 10], [2, 11], [1, 12], [3, -3]],
          ],
          output=[[5, 7], [6, 8], [2, 11], [3, -3]]
      ),
  ]
  constants = [1]
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.boolean_mask(in1, tf.greater(in1[:, 0], 1))'
  source = 'SO_11418192_depth2'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_08')


# SO_49567723_depth2 is out of scope. It's another table merge.

# SO_49987108_depth2 is out of scope. It uses DataFrame.fillna(method='ffill'),
# which has no equivalent (afaik) in TensorFlow.


def autopandas_09():
  # This is yet another sort. But this time, the data actually does require
  # sorting by both columns! I have constructed the example to reflect this.
  # Also, this problem (sort by 2 columns) is the same as stackoverflow_19.
  examples = [
      benchmark.Example(
          inputs=[
              [[8, 5, 9, 3],
               [3, 6, 6, 8],
               [7, 9, 3, 4],
               [7, 6, 5, 4],
               [3, 7, 2, 6]]
          ],
          output=[[3, 6, 6, 8],
                  [3, 7, 2, 6],
                  [7, 6, 5, 4],
                  [7, 9, 3, 4],
                  [8, 5, 9, 3]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.gather(tf.gather(in1, tf.argsort(in1[:, 1], stable=True)), tf.argsort(tf.gather(in1, tf.argsort(in1[:, 1], stable=True))[:, 0], stable=True))'
  source = 'SO_13261691_depth2'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_09')


# SO_13659881_depth2 is out of scope. It involves counting how many times each
# row appears, and then appending that count to the deduplicated rows.
# TensorFlow is not designed for deduplicating rows.


def autopandas_10():
  # Drop NaN. NaN is given as a constant.
  examples = [
      benchmark.Example(
          inputs=[
              [float('nan'), 11, 12, float('nan'), 16, 18],
          ],
          output=[11, 12, 16, 18]
      ),
  ]
  constants = [float('nan')]
  description = ''  # No description for AutoPandas benchmarks.
  # TODO(kshi): We don't support tf.math.is_nan() and tf.math.logical_not()!!
  # Once again, TF-Coder is better than me. TF-Coder replaces
  # `tf.math.logical_not(tf.math.is_nan(in1))` with `tf.equal(in1, in1)` because
  # NaN is the only thing that's not equal to itself.
  target_program = 'tf.cast(tf.boolean_mask(in1, tf.math.logical_not(tf.math.is_nan(in1))), tf.int32)'
  source = 'SO_13807758_depth2'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_10')


# SO_34365578_depth2 is out of scope. It involves grouping values and doing a
# sum aggregation for each group. TensorFlow is not designed for grouping values
# like this.

# SO_10982266_depth3 is also out of scope, involving a grouping and then mean
# aggregation for each group.


def autopandas_11():
  # Transpose and prepend the row index. I changed the data to not have such
  # obvious patterns.
  examples = [
      benchmark.Example(
          inputs=[
              [[1, 4, 2, 7, 6], [20, 10, 50, 40, 30]],
          ],
          output=[[0, 1, 20],
                  [1, 4, 10],
                  [2, 2, 50],
                  [3, 7, 40],
                  [4, 6, 30]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.transpose(tf.concat((tf.expand_dims(tf.range(5), axis=0), in1), axis=0))'
  source = 'SO_11811392_depth3'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_11')


def autopandas_12():
  # For each pair, compute pair[1] / (pair[0] + pair[1]).
  # I added some numbers to avoid in1[:, :, 1] == tf.reduce_max(in1, axis=-1).
  examples = [
      benchmark.Example(
          inputs=[
              [[[2, 8], [2, 6], [6, 2]],
               [[0, 2], [1, 1], [2, 0]]],
          ],
          output=[[0.8, 0.75, 0.25],
                  [1.0, 0.5, 0.0]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.cast(tf.divide(in1[:, :, 1], tf.reduce_sum(in1, axis=2)), tf.float32)'
  source = 'SO_49581206_depth3'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_12')


def autopandas_13():
  # Select rows where row[1] is in some set. I'm matching their data where
  # possible.
  examples = [
      benchmark.Example(
          inputs=[
              [[101, 0, 11, 0],
               [102, 1, 12, 4],
               [103, 2, 13, 2],
               [104, 3, 14, 8],
               [105, 4, 15, 4],
               [106, 5, 16, 5],
               [107, 6, 17, 4],
               [108, 7, 18, 7],
               [109, 8, 19, 7],
               [110, 9, 20, 4]],
              [4, 2, 6],
          ],
          output=[[103, 2, 13, 2],
                  [105, 4, 15, 4],
                  [107, 6, 17, 4]]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.boolean_mask(in1, tf.reduce_any(tf.equal(in1[:, 1], tf.expand_dims(in2, axis=1)), axis=0))'
  source = 'SO_12065885_depth3'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_13')


def autopandas_14():
  # The intended solution is to do a merge, but the example actually has nothing
  # to merge, so all that happens is an extra column is created filled with NaN.
  # Why are there so many problems where the example completely underspecifies
  # the intended transformation? Instead of ignoring the problem because it has
  # a merge (out of scope), let's just try to append a NaN column.
  examples = [
      benchmark.Example(
          inputs=[
              [[1, 0, 1, 2],
               [1, 1, 3, 4],
               [2, 0, 1, 2],
               [2, 1, 3, 4]],
              # They have another input that is used to provide the column
              # header for the new NaN-filled column. Tensors don't have column
              # headers so this input is useless. Omit it, or else TF-Coder will
              # be required to use it.
          ],
          output=[[1, 0, 1, 2, float('nan')],
                  [1, 1, 3, 4, float('nan')],
                  [2, 0, 1, 2, float('nan')],
                  [2, 1, 3, 4, float('nan')]]
      ),
  ]
  constants = [float('nan')]
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.concat((tf.cast(in1, tf.float32), tf.fill([4, 1], float("nan"))), axis=1)'
  source = 'SO_13576164_depth3'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_14')


# SO_14023037_depth3 is out of scope. It uses DataFrame.fillna(method='bfill'),
# which has no equivalent (afaik) in TensorFlow.


def autopandas_15():
  # The task is to group by everything except one numeric column, and then
  # cumsum over that numeric column. Grouping is out-of-scope, and the groups
  # essentially become row headers anyway. The only part of this that is doable
  # in TensorFlow is the cumsum part. I changed the example because it was too
  # simple.
  examples = [
      benchmark.Example(
          inputs=[
              [1, 1, 2, 1, 3, 2],
          ],
          output=[1, 2, 4, 5, 8, 10]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.cumsum(in1)'
  source = 'SO_53762029_depth3'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_15')


# SO_21982987_depth3 is out of scope, involving a grouping and then mean
# aggregation for each group.


def autopandas_16():
  # Reduce mean across columns, and then ignore column 0.
  examples = [
      benchmark.Example(
          inputs=[
              [[0, 6, 0],
               [3, 101, 14],
               [0, 91, 6],
               [5, 15, 0]],
          ],
          output=[53.25, 5.00]
      ),
  ]
  constants = []
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'tf.reduce_mean(tf.cast(in1, tf.float32), axis=0)[1:]'
  source = 'SO_39656670_depth3'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_16')


# SO_23321300_depth3 is out of scope, involving a grouping and then mean
# aggregation for each group.


# A template for easy copy/pasting. Copying an existing benchmark and replacing
# parts of it will lead to a state where the benchmark is half-correct, but not
# obviously so. Copy this template instead when creating new benchmarks.
"""

def autopandas_NUMBER():
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
  description = ''  # No description for AutoPandas benchmarks.
  target_program = 'SOLUTION_PROGRAM'
  source = 'PROBLEM_SOURCE'
  return benchmark.Benchmark(examples=examples,
                             constants=constants,
                             description=description,
                             target_program=target_program,
                             source=source,
                             name='autopandas_NUMBER')

"""  # pylint: disable=pointless-string-statement

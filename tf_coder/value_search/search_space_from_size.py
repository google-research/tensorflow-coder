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
"""Computes the size of value search's search space.

Found 134 operations.
# operations with arity 0: 0
# operations with arity 1: 37
# operations with arity 2: 76
# operations with arity 3: 20
# operations with arity 4: 1

Num leaf nodes: 13

There are 2.1364033304069425e+18 expressions using exactly 5 ops and 11 nodes.
There are 2.2841608421687201e+18 expressions using at most 5 ops and 11 nodes.
"""

import itertools

from absl import app
from absl import flags
import numpy as np
from tf_coder import tf_coder_utils as utils
from tf_coder.value_search import all_operations


FLAGS = flags.FLAGS

flags.DEFINE_integer('num_ops', 5, 'The target number of operations.')
flags.DEFINE_integer('num_nodes', 11,
                     'The target number of nodes in the expression tree.')
flags.DEFINE_integer('num_leaf_choices', 13,
                     'The number of inputs and constants available.')


def compute_search_space_size(num_ops, num_nodes, num_leaf_choices):
  """Computes and prints the size of the search space.

  This counts the total number of expressions with exactly the given number of
  operations and nodes in the expression tree. Distinct expressions will be
  counted separately even if they evaluate to the same value, unlike in
  TF-Coder's value_search algorithm which does value-based pruning.

  Args:
    num_ops: The target number of operations.
    num_nodes: A target number of nodes in the expression tree.
    num_leaf_choices: The number of distinct inputs and constants available to
      form the leaves of the expression tree.

  Returns:
    The DP table, where dp[i][j] is the answer for i ops and j nodes.
  """
  operations = all_operations.get_operations(include_sparse_operations=True)
  max_arity = max(op.num_args for op in operations)
  arity_counts = [0] * (max_arity + 1)

  print('Found {} operations.'.format(len(operations)))
  for arity in range(max_arity + 1):
    arity_counts[arity] = sum(1 for op in operations if op.num_args == arity)
    print('# operations with arity {}: {}'.format(arity, arity_counts[arity]))
  print('\nNum leaf nodes: {}'.format(num_leaf_choices))

  # dp[i][j] = the number of expressions using exactly i ops and j nodes.
  dp = np.zeros((num_ops + 1, num_nodes + 1))

  # The only expressions using 0 ops are single-node leaves.
  dp[0][1] = num_leaf_choices

  for ops in range(1, num_ops + 1):
    for nodes in range(1, num_nodes + 1):
      # The running total number of ways to satisfy # ops and # nodes.
      total = 0

      for arity in range(1, max_arity + 1):
        # The running total number of ways to fill the arguments.
        args_total = 0

        # The ways to allocate remaining ops and nodes to the arguments.
        ops_partitions = utils.generate_partitions(num_elements=ops - 1,
                                                   num_parts=arity)
        nodes_partitions = utils.generate_partitions(num_elements=nodes - 1,
                                                     num_parts=arity)

        for ops_partition, nodes_partition in itertools.product(
            ops_partitions, nodes_partitions):
          # The i-th argument must have ops_partition[i] ops and
          # nodes_partition[i] nodes. Look up the number of ways in the DP
          # table.
          args_total += np.prod([dp[ops_partition[i]][nodes_partition[i]]
                                 for i in range(arity)])

        # There are arity_counts[arity] choices for the outermost operation.
        total += args_total * arity_counts[arity]

      dp[ops][nodes] = total

  print('\nThere are {} expressions using exactly {} ops and {} nodes.'.format(
      dp[num_ops][num_nodes], num_ops, num_nodes))
  print('There are {} expressions using at most {} ops and {} nodes.'.format(
      np.sum(dp), num_ops, num_nodes))
  return dp


def main(unused_argv):
  compute_search_space_size(num_ops=FLAGS.num_ops,
                            num_nodes=FLAGS.num_nodes,
                            num_leaf_choices=FLAGS.num_leaf_choices)


if __name__ == '__main__':
  app.run(main)

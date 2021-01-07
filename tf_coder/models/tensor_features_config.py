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
"""Config for the tensor features model."""

import tensorflow as tf
from tf_coder import tf_coder_utils
from tf_coder.datasets import collect_tensor_data
from tf_coder.value_search import all_operations


def get_config():
  """Gets the default config for the tensor features model."""
  cfg = {}

  # Model size.
  cfg['feedforward_size'] = 2048
  cfg['num_feedforward_layers'] = 2

  # Training hyperparams.
  cfg['batch_size'] = 1024
  cfg['learning_rate'] = 1e-5  # 1e-5 is better, 1e-4 is faster for debugging.
  cfg['global_norm_clip'] = 5
  cfg['num_epochs'] = 3

  # Loss function and options.

  # sigmoid_ce: Average the sigmoid cross entropy loss over all operations.
  # f_beta: Use a differentiable version of F_beta score as the loss.
  cfg['loss'] = 'f_beta'
  # If using f_beta loss, this is the value of beta. Because TF-Coder cares more
  # about recall than precision, this value should be >= 1.
  cfg['beta'] = 2
  # Whether to use weights to "balance" the prevalence of different ops. Let
  # `c_i` be the number of training examples operation `op_i` appears in. If
  # this option is True, then if `op_i` is used in a given example, we treat it
  # as actually appearing ((mean_j c_j) / c_i) times. This effectively upweights
  # rare operations and downweights common operations, such that each operation
  # now appears an equal (mean_j c_j) times across the dataset. Furthermore,
  # the total number of operations used across the dataset remains the same,
  # namely (sum_j c_j).
  cfg['weighted_ops'] = True
  # If 'mean', use (mean_j c_j) as the weight numerator. If 'max', use
  # (max_j c_j) instead, so that no operations are downweighted.
  cfg['weight_numerator'] = 'mean'
  # If using weights to balance the occurrences of ops, this is the maximum
  # weight. Weights that are too large may increase training instability.
  # However, we do have gradient clipping in place, so this may be unnecessary.
  cfg['max_weight'] = 10000

  # Details about the training data.
  operations = all_operations.get_operations(include_sparse_operations=True)
  cfg['num_ops'] = len(operations)
  cfg['num_kinds'] = 5
  cfg['num_dtypes'] = len(tf_coder_utils.INT_DTYPES +
                          tf_coder_utils.FLOAT_DTYPES + (tf.bool,))
  cfg['max_rank'] = collect_tensor_data.MAX_RANK
  cfg['num_shape_buckets'] = len(collect_tensor_data.COUNT_BOUNDARIES)
  cfg['num_float_buckets'] = len(collect_tensor_data.FLOAT_BOUNDARIES)
  cfg['num_count_buckets'] = len(collect_tensor_data.COUNT_BOUNDARIES)
  cfg['max_num_inputs'] = 3
  # Store the names of all operations used at the time of model training, so we
  # know how to interpret the model's predictions at test time, even if new
  # operations were added in between.
  cfg['operation_names'] = [op.name for op in operations]
  cfg['op_counts'] = [
      261791, 3219730, 49984, 468220, 70588, 671391, 304105, 76203, 361,
      12293742, 223803, 208334, 4469035, 0, 5228323, 395630, 514399, 8810328,
      175664, 982, 48877, 24769, 1429728, 15448, 124548, 675, 788727, 180186,
      770114, 198687, 7687, 26592, 773463, 443414, 26209, 183343, 397296,
      580331, 257051, 22749, 24187, 1229, 1727, 8011, 28719, 316277, 229433,
      54625, 5881, 23486, 2548, 33794, 466112, 1906853, 590050, 2780775, 29209,
      2393112, 4041, 342637, 39436, 9667, 1308, 2748, 752650, 17687, 5663,
      1246906, 1327261, 486676, 421095, 126426, 184234, 134355, 2565768,
      2124113, 5845, 49484, 338106, 63409, 6227, 9410, 953093, 124307, 878241,
      126966, 211238, 96981, 185479, 4384591, 800148, 119813, 188229, 5031788,
      1761827, 244750, 2493965, 2, 339585, 1145115, 4293436, 149049, 1058,
      342232, 687, 33192, 451, 31190, 1145201, 2621, 1642, 128519, 142314,
      22508, 16, 3660, 499, 48583, 359399, 118306, 3940, 9318, 3614, 543957,
      809717, 9070582, 2120948, 273953, 719889, 603656, 170092, 517624, 481419,
      2997663
  ]

  # Evaluation settings.
  cfg['eval_batch_size'] = 100000  # This is actually the entire eval set.
  cfg['eval_step_frequency'] = 100  # A multiple of summary_step_frequency.

  # Saving summaries and checkpoints.
  cfg['summary_step_frequency'] = 100
  cfg['save_step_frequency'] = 100
  cfg['keep_checkpoint_every_n_hours'] = 0.5
  cfg['allow_restore'] = False
  cfg['log_device_placement'] = False

  return cfg

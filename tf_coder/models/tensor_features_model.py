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
"""A model that predicts operations from features of the I/O tensors."""

import collections
import json
import os
import timeit
from typing import Any, Dict, Text, Tuple

from absl import logging
import six
import tensorflow as tf
import termcolor
from tf_coder.datasets import collect_tensor_data


ConfigType = Dict[Text, Any]

Model = collections.namedtuple('Model', [
    'optimizer',

    'kind_embeddings',  # num_kinds, embedding_size
    'dtype_embeddings',  # num_dtypes, embedding_size
    'rank_embeddings',  # max_rank + 1, embedding_size
    'shape_buckets_embeddings',  # num_shape_buckets, embedding_size
    'float_buckets_embeddings',  # num_float_buckets, embedding_size
    'count_buckets_embeddings',  # num_count_buckets, embedding_size
    'booleans_embeddings',  # 2, embedding_size
    'io_comparisons_embeddings',  # 3, embedding_size
    'io_booleans_embeddings',  # 2, embedding_size
    'io_count_buckets_embeddings',  # num_count_buckets, embedding_size
    'num_inputs_embeddings',  # max_num_inputs, embedding_size

    'feedforward_layers',  # each (batch_size, example_len or feedforward_size
                           #       -> batch_size, feedforward_size)
    'output_layer',  # batch_size, feedforward_size -> batch_size, num_ops
])
Result = collections.namedtuple('Result', [
    'operation_logits',  # batch_size, num_ops
])
Example = collections.namedtuple('Example',  # pylint: disable=invalid-name
                                 list(collect_tensor_data.FEATURE_NAME_TO_TYPE))


def parse_example_proto(serialized_example: Text) -> Dict[Text, tf.Tensor]:
  """Parses the serialized tf.train.Example proto."""
  features = {}
  for feature_name, feature_type in six.iteritems(
      collect_tensor_data.FEATURE_NAME_TO_TYPE):
    dtype = (
        tf.int64 if feature_type == collect_tensor_data.FeatureType.INT else  # pylint: disable=g-long-ternary
        tf.float32 if feature_type == collect_tensor_data.FeatureType.FLOAT else
        tf.string if feature_type == collect_tensor_data.FeatureType.STRING else
        None)
    assert dtype is not None
    features[feature_name] = tf.io.VarLenFeature(dtype)
  parsed = tf.io.parse_single_example(serialized_example, features)
  for key in parsed:
    parsed[key] = tf.sparse.to_dense(parsed[key])
  return parsed


def get_model(config: ConfigType) -> Model:
  """Gets a namedtuple with all the variables and layers of the model."""
  initializer = tf.initializers.GlorotUniform()
  optimizer = tf.keras.optimizers.Adam(config['learning_rate'])
  optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

  return Model(
      optimizer=optimizer,

      kind_embeddings=tf.Variable(
          name='kind_embeddings',
          initial_value=initializer(
              shape=[config['num_kinds']] * 2,
              dtype=tf.float32)),
      dtype_embeddings=tf.Variable(
          name='dtype_embeddings',
          initial_value=initializer(
              shape=[config['num_dtypes']] * 2,
              dtype=tf.float32)),
      rank_embeddings=tf.Variable(
          name='rank_embeddings',
          initial_value=initializer(
              shape=[config['max_rank'] + 1] * 2,
              dtype=tf.float32)),
      shape_buckets_embeddings=tf.Variable(
          name='shape_buckets_embeddings',
          initial_value=initializer(
              shape=[config['num_shape_buckets']] * 2,
              dtype=tf.float32)),
      float_buckets_embeddings=tf.Variable(
          name='float_buckets_embeddings',
          initial_value=initializer(
              shape=[config['num_float_buckets']] * 2,
              dtype=tf.float32)),
      count_buckets_embeddings=tf.Variable(
          name='count_buckets_embeddings',
          initial_value=initializer(
              shape=[config['num_count_buckets']] * 2,
              dtype=tf.float32)),
      booleans_embeddings=tf.Variable(
          name='booleans_embeddings',
          initial_value=initializer(
              shape=[2] * 2,
              dtype=tf.float32)),
      io_comparisons_embeddings=tf.Variable(
          name='io_comparisons_embeddings',
          initial_value=initializer(
              shape=[3] * 2,
              dtype=tf.float32)),
      io_booleans_embeddings=tf.Variable(
          name='io_booleans_embeddings',
          initial_value=initializer(
              shape=[2] * 2,
              dtype=tf.float32)),
      io_count_buckets_embeddings=tf.Variable(
          name='io_count_buckets_embeddings',
          initial_value=initializer(
              shape=[config['num_count_buckets']] * 2,
              dtype=tf.float32)),
      num_inputs_embeddings=tf.Variable(
          name='num_inputs_embeddings',
          initial_value=initializer(
              shape=[config['max_num_inputs'] + 1] * 2,
              dtype=tf.float32)),

      feedforward_layers=[tf.keras.layers.Dense(config['feedforward_size'])
                          for _ in range(config['num_feedforward_layers'])],
      output_layer=tf.keras.layers.Dense(config['num_ops']),
  )


@tf.function
def predict(model: Model, example: Example) -> Result:
  """Predict which ops will be used.

  Args:
    model: The Model (a namedtuple containing the variables used by the model).
    example: The Example to do prediction for. An example represents an I/O
      tensor pair.

  Returns:
    A Result object with the prediction results.
  """
  embedding_pairs = [
      (model.kind_embeddings, example.kind),
      (model.dtype_embeddings, example.dtype),
      (model.rank_embeddings, example.rank),
      (model.shape_buckets_embeddings, example.shape_buckets),
      (model.float_buckets_embeddings, example.float_buckets),
      (model.count_buckets_embeddings, example.count_buckets),
      (model.booleans_embeddings, example.booleans),
      (model.io_comparisons_embeddings, example.io_comparisons),
      (model.io_booleans_embeddings, example.io_booleans),
      (model.io_count_buckets_embeddings, example.io_count_buckets),
      (model.num_inputs_embeddings, example.num_inputs),
  ]

  embeddings = [tf.nn.embedding_lookup(embeddings, item)
                for embeddings, item in embedding_pairs]

  unembedded = [
      example.shape,
      # example.floats,  # May include inf and nan! Use the buckets only.
      example.counts,
      example.fractions,
      example.io_counts,
      example.io_fractions,
  ]

  # Some features are single numbers, and some are sequences of numbers. Some
  # features are embedded, and some are left unembedded. In any case,
  # concatenate everything into one long input vector.
  batch_size = example.kind.shape[0]
  flattened_embeddings = [tf.reshape(embedding, (batch_size, -1))
                          for embedding in embeddings]
  to_concat = [tf.cast(tensor, tf.float32)
               for tensor in flattened_embeddings + unembedded]
  batched_inputs = tf.concat(to_concat, axis=1)

  layer_output = batched_inputs
  for feedforward_layer in model.feedforward_layers:
    layer_output = feedforward_layer(layer_output)
  operation_logits = model.output_layer(layer_output)

  return Result(operation_logits=operation_logits)


@tf.function
def f_beta_loss(ground_truth, predictions, beta):
  """Computes F_beta loss (1 minus F_beta score) in a differentiable way."""
  # Inspired by:
  # https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
  true_positives = tf.reduce_sum(ground_truth * predictions)
  false_positives = tf.reduce_sum(
      tf.where(tf.equal(ground_truth, 0.0), 1.0, 0.0) * predictions)
  false_negatives = tf.reduce_sum(ground_truth * (1 - predictions))

  epsilon = 0.0001
  precision = true_positives / (true_positives + false_positives + epsilon)
  recall = true_positives / (true_positives + false_negatives + epsilon)

  beta_squared = beta ** 2
  f_beta_score = ((1 + beta_squared) * precision * recall /
                  (beta_squared * precision + recall + epsilon))
  return 1 - f_beta_score


@tf.function
def get_loss(model: Model,
             config: ConfigType,
             result: Result,
             example: Example) -> float:
  """Computes the loss for the model."""
  del model  # Unused.

  # The ground truth.
  operations = tf.clip_by_value(tf.cast(example.operations, tf.float32), 0, 1)

  if config['weighted_ops']:
    op_counts = tf.cast(config['op_counts'], tf.float32)
    if config['weight_numerator'] == 'mean':
      weight_numerator = tf.reduce_mean(op_counts)
    elif config['weight_numerator'] == 'max':
      weight_numerator = tf.reduce_max(op_counts)
    else:
      raise ValueError("Unknown config['weight_numerator']: {}".format(
          config['weight_numerator']))
    weight_if_op_used = tf.minimum(
        tf.cast(config['max_weight'], tf.float32),
        weight_numerator / op_counts)
    weights = tf.where(tf.cast(operations, tf.bool),  # [batch_size, num_ops]
                       weight_if_op_used,  # [num_ops]
                       1.0)

  loss_type = config['loss']
  if loss_type == 'sigmoid_ce':
    loss_per_op = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=operations, logits=result.operation_logits)
    if config['weighted_ops']:
      loss_per_op *= weights
    return tf.reduce_mean(loss_per_op)
  elif loss_type == 'f_beta':
    if config['weighted_ops']:
      # `operations` now contains 0 if the op is not used, otherwise a real
      # number representing the number of times we should treat the op as used.
      operations *= weights
    return f_beta_loss(ground_truth=operations,
                       predictions=tf.sigmoid(result.operation_logits),
                       beta=config['beta'])
  else:
    raise ValueError("Unhandled config['loss']: {}".format(loss_type))


@tf.function
def get_accuracy(model: Model,
                 result: Result,
                 example: Example) -> Tuple[float, float, float, float, float]:
  """Returns a tuple containing various accuracy metrics."""
  del model  # Unused.

  # TODO(kshi): Refactor to reuse the simpler code in f_beta_loss.

  predictions = tf.cast(tf.greater(tf.sigmoid(result.operation_logits), 0.5),
                        tf.int32)
  operations = tf.clip_by_value(tf.cast(example.operations, tf.int32), 0, 1)

  correct = tf.equal(predictions, operations)
  num_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
  accuracy = num_correct / tf.size(correct)

  # Precision.
  # Among ops that the model predicted were used, how often was it correct?
  predicted_positive_indices = tf.where(tf.equal(predictions, 1))
  predicted_positive_truth = tf.gather_nd(operations,
                                          predicted_positive_indices)
  num_correct = tf.reduce_sum(predicted_positive_truth)
  precision = num_correct / tf.size(predicted_positive_truth)

  # Recall, i.e., true positive rate.
  # Among ops that were actually used, how often was the model correct?
  actually_positive_indices = tf.where(tf.equal(operations, 1))
  actually_positive_predictions = tf.gather_nd(predictions,
                                               actually_positive_indices)
  num_correct = tf.reduce_sum(actually_positive_predictions)
  recall = num_correct / tf.size(actually_positive_predictions)

  # True negative rate.
  # Among ops that were actually not used, how often was the model correct?
  actually_negative_indices = tf.where(tf.equal(operations, 0))
  actually_negative_predictions = tf.gather_nd(predictions,
                                               actually_negative_indices)
  num_correct = tf.reduce_sum(1 - actually_negative_predictions)
  true_negative_rate = num_correct / tf.size(actually_negative_predictions)

  # F1 score.
  f1_score = 2 * precision * recall / (precision + recall)

  return accuracy, precision, recall, true_negative_rate, f1_score


def get_trainable_variables(model: Model):
  return [
      model.kind_embeddings,
      model.dtype_embeddings,
      model.rank_embeddings,
      model.shape_buckets_embeddings,
      model.float_buckets_embeddings,
      model.count_buckets_embeddings,
      model.booleans_embeddings,
      model.io_comparisons_embeddings,
      model.io_booleans_embeddings,
      model.io_count_buckets_embeddings,
      model.num_inputs_embeddings,
  ] + (sum([layer.variables for layer in model.feedforward_layers], []) +
       model.output_layer.variables)


def save_config(config, filepath):
  json_string = json.dumps(config, sort_keys=True, indent=4,
                           separators=(',', ': '))
  with tf.io.gfile.GFile(filepath, mode='w') as f:
    f.write(json_string)


def load_config(filepath):
  with tf.io.gfile.GFile(filepath, mode='r') as f:
    return json.load(f)


def create_checkpoint(model, global_step=None):
  """Creates a Checkpoint object."""
  trackables = model._asdict()
  if global_step is not None:
    trackables['global_step'] = global_step
  # Unpack the layers in the list of feedforward layers, since a list of layers
  # is not trackable.
  for i, layer in enumerate(trackables['feedforward_layers']):
    trackables['feedforward_layer_{}'.format(i + 1)] = layer
  del trackables['feedforward_layers']
  return tf.train.Checkpoint(**trackables)


def train(model: Model, train_dataset, eval_dataset, config, work_unit_dir):
  """Trains a model that predicts operations from I/O tensor features."""
  global_step = model.optimizer.iterations
  checkpoint_dir = os.path.join(work_unit_dir, 'checkpoints')

  config_filepath = os.path.join(work_unit_dir, 'config.json')
  save_config(config, config_filepath)

  log_dir = os.path.join(work_unit_dir, 'train_logs')
  writer = tf.summary.create_file_writer(log_dir, flush_millis=10000)

  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  checkpoint = create_checkpoint(model, global_step)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, directory=checkpoint_dir, max_to_keep=1,
      keep_checkpoint_every_n_hours=config['keep_checkpoint_every_n_hours'])

  skip_epoch_checkpoint = False
  if checkpoint_path is not None:
    if config['allow_restore']:
      logging.info(termcolor.colored('Restoring from checkpoint: %s',
                                     color='red', attrs=['bold']),
                   checkpoint_path)
      checkpoint.restore(checkpoint_path)
      skip_epoch_checkpoint = True
    else:
      raise RuntimeError(
          'A preexisting checkpoint was found in the run directory. Aborting.')

  # Get the first batch from the eval dataset to use as the eval batch.
  eval_batch = Example(**next(iter(eval_dataset)))

  trainable_variables = None
  last_batch_end_time = timeit.default_timer()
  with writer.as_default():
    for epoch in range(1, 1 + config['num_epochs']):
      logging.info(
          termcolor.colored('Epoch %d', color='green', attrs=['bold']),
          epoch)

      if skip_epoch_checkpoint:
        skip_epoch_checkpoint = False
      else:
        checkpoint_manager.save()

      for batch_index, example_dict in enumerate(train_dataset):
        example = Example(**example_dict)
        if epoch == 1 and batch_index == 0:
          logging.info(example)
        with tf.GradientTape() as tape:
          result = predict(model, example)
          loss = get_loss(model, config, result, example)
        accuracy, precision, recall, true_negative_rate, f1_score = (
            get_accuracy(model, result, example))

        if trainable_variables is None:
          # The variables only become available after the first forward pass.
          trainable_variables = get_trainable_variables(model)

        grads = tape.gradient(loss, trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(grads,
                                                    config['global_norm_clip'])
        model.optimizer.apply_gradients(zip(grads, trainable_variables))

        if tf.equal(global_step % config['save_step_frequency'], 0):
          checkpoint_manager.save()

        evaluate = tf.equal(global_step % config['eval_step_frequency'], 0)
        if evaluate:
          eval_result = predict(model, eval_batch)
          eval_loss = get_loss(model, config, eval_result, eval_batch)
        else:
          eval_loss = None

        write_summaries = tf.equal(
            global_step % config['summary_step_frequency'], 0)
        if write_summaries:
          tf.summary.experimental.set_step(global_step)
          tf.summary.scalar('loss', loss)
          tf.summary.scalar('global_norm', global_norm)
          tf.summary.scalar('accuracy', accuracy)
          tf.summary.scalar('precision', precision)
          tf.summary.scalar('recall', recall)
          tf.summary.scalar('true_negative_rate', true_negative_rate)
          tf.summary.scalar('f1_score', f1_score)
          if eval_loss is not None:
            tf.summary.scalar('eval_loss', eval_loss)
          # for variable in trainable_variables:
          #   tf.summary.histogram(variable.name.rstrip(':0'), variable)

        logging.info(termcolor.colored('Epoch %s, batch %s, global_step %s',
                                       color='green', attrs=['bold']),
                     epoch, batch_index + 1, int(global_step))
        logging.info(termcolor.colored('Loss: %s', color='magenta'),
                     loss.numpy())
        if eval_loss is not None:
          logging.info(termcolor.colored('Eval loss: %s', color='magenta',
                                         attrs=['bold']),
                       eval_loss.numpy())
        logging.info(termcolor.colored('Global norm: %s', color='magenta'),
                     global_norm.numpy())
        logging.info(termcolor.colored('Accuracy: %.3f%%', color='blue'),
                     accuracy.numpy() * 100)
        logging.info(termcolor.colored('Precision: %.3f%%', color='cyan'),
                     precision.numpy() * 100)
        logging.info(termcolor.colored('Recall: %.3f%%', color='cyan'),
                     recall.numpy() * 100)
        logging.info(termcolor.colored('True negative rate: %.3f%%',
                                       color='blue'),
                     true_negative_rate.numpy() * 100)
        logging.info(termcolor.colored('F1 score: %.3f%%',
                                       color='blue'),
                     f1_score.numpy() * 100)
        batch_end_time = timeit.default_timer()
        logging.info(termcolor.colored('Batch time: %.2f sec',
                                       color='blue'),
                     batch_end_time - last_batch_end_time)
        last_batch_end_time = batch_end_time


def eval_single_example(model: Model, serialized_example: Text) -> Result:
  """Runs the model on one example."""
  example_dict = parse_example_proto(serialized_example)
  for key in example_dict:
    example_dict[key] = tf.expand_dims(example_dict[key], axis=0)
  example = Example(**example_dict)
  return predict(model, example)

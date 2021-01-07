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
"""Loads extracted tf_function data from disk."""

import os
from typing import Dict, List, Text

import tensorflow as tf
from tf_coder.datasets.github import tokenizer

ADD_OPERATION_NAMES = True
ADD_OPERATION_DOCSTRINGS = False

DEFAULT_DATA_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data', '')


def parse_example_proto(serialized_example: Text) -> Dict[Text, tf.Tensor]:
  """Parses a serialized tensorflow Example into its component tensors.

  Executed in TensorFlow graph mode by tf.data.Dataset.map.

  Args:
    serialized_example: A single tf.Example, serialized as bytes. The output
      of the extract_tf_functions Beam pipeline.
  Returns:
    A dict mapping keys to string tensors for a single example.
  """
  features = {
      'docstring': tf.io.VarLenFeature(tf.string),
      'names': tf.io.VarLenFeature(tf.string),
      'comments': tf.io.VarLenFeature(tf.string),
      'strings': tf.io.VarLenFeature(tf.string),
      'tf_functions': tf.io.VarLenFeature(tf.string),
  }
  parsed = tf.io.parse_single_example(serialized_example, features)
  for key in parsed:
    parsed[key] = tf.sparse.to_dense(parsed[key])
  return parsed


def _as_text_list(value: tf.Tensor) -> List[Text]:
  return [b.decode('utf-8') for b in value.numpy().tolist()]


def _as_python_example(
    example: Dict[Text, tf.Tensor]) -> Dict[Text, List[Text]]:
  return {
      key: _as_text_list(value)
      for key, value in example.items()
  }


def load_data(prefix) -> List[Dict[Text, List[Text]]]:
  filenames = tf.io.gfile.glob(prefix + '*')
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(parse_example_proto)
  return [_as_python_example(example) for example in dataset]


def get_operations(example: Dict[Text, List[Text]]) -> Text:
  return ' '.join(
      [tf_function[3:] if tf_function.startswith('tf.') else tf_function
       for tf_function in example['tf_functions']]
  )


def get_context(example: Dict[Text, List[Text]]) -> Text:
  """Gets the textual context provided in a single example."""
  docstring = example['docstring'][0]
  comments = example['comments']
  names = example['names']
  strings = example['strings']

  tokens = (
      tokenizer.tokenize(docstring)
      + tokenizer.tokens_from_text_list(comments)
      + tokenizer.tokens_from_text_list(names)
      + tokenizer.tokens_from_text_list(strings)
  )
  return ' '.join(tokens)


def get_full_context(example):
  context = get_context(example)
  if ADD_OPERATION_NAMES:
    context += ' ' + get_operations(example)
  if ADD_OPERATION_DOCSTRINGS:
    raise NotImplementedError()
  return ' '.join(tokenizer.tokenize(context))


def uses_operation(example: Dict[Text, List[Text]], tf_function: Text) -> bool:
  return tf_function in example['tf_functions']

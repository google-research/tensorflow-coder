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
"""Serialization of objects relevant to TF-Coder.

This module will be used to send information from the public Colab notebook to
Google Analytics, in string form. Using BigQuery we can extract the strings that
were sent, and then parse the strings back into the Python objects that they
represent. The information we want to log includes:

  * Input/output objects. Usually these are multidimensional lists, Tensors, or
    SparseTensors, but in principle these can be anything that value search
    supports (e.g., primitives, dtypes, tuples of Tensors etc.).
  * Constants. Usually these are Python primitives, but again they may be
    anything value search supports (e.g., a shape tuple).
  * Natural language description. This should be a string and may contain tricky
    characters like Unicode or quotes.
  * Settings for the TF-Coder tool. These may use standard Python collections,
    i.e. lists/tuples/sets/dicts. This category of information should be treated
    generally to be future-proof.
  * Results of the TF-Coder tool. These would include timestamps and solution
    expressions.
  * Other metadata, e.g., session/problem IDs, and whether the data can be
    released in a dataset.
"""

import ast
from typing import Any, List, Text

import numpy as np
import tensorflow as tf

# Constant strings for dict representations of objects.
_KIND_KEY = 'kind'
_DTYPE_KIND = 'DType'
_TENSOR_KIND = 'Tensor'
_SPARSE_TENSOR_KIND = 'SparseTensor'
_DICT_KIND = 'Dict'


def _object_to_literal(to_serialize: Any, container_stack: List[Any]) -> Any:
  """Turns a supported object into a Python literal."""
  if isinstance(to_serialize, (int, float, bool, str, bytes, type(None))):
    return to_serialize

  elif isinstance(to_serialize, tf.DType):
    dtype_string = repr(to_serialize)
    assert dtype_string.startswith('tf.')
    dtype_string = dtype_string[len('tf.'):]
    return {_KIND_KEY: _DTYPE_KIND,
            'dtype': dtype_string}

  elif isinstance(to_serialize, tf.Tensor):
    tensor_content = to_serialize.numpy()
    # Sometimes tensor_content is a numpy type, and sometimes it's a normal
    # Python type.
    if type(tensor_content).__module__ == np.__name__:
      tensor_content = tensor_content.tolist()
    return {_KIND_KEY: _TENSOR_KIND,
            'content': tensor_content,
            'dtype': _object_to_literal(to_serialize.dtype, container_stack)}

  elif isinstance(to_serialize, tf.SparseTensor):
    return {_KIND_KEY: _SPARSE_TENSOR_KIND,
            'indices': _object_to_literal(to_serialize.indices,
                                          container_stack),
            'values': _object_to_literal(to_serialize.values, container_stack),
            'dense_shape': _object_to_literal(to_serialize.dense_shape,
                                              container_stack)}

  elif isinstance(to_serialize, dict):
    if any(to_serialize is seen for seen in container_stack):
      raise ValueError('Cycle detected in object dependencies.')
    container_stack.append(to_serialize)
    result = {_object_to_literal(key, container_stack):
                  _object_to_literal(value, container_stack)
              for key, value in to_serialize.items()}
    container_stack.pop()
    return {_KIND_KEY: _DICT_KIND,
            'dict': result}

  elif isinstance(to_serialize, (list, tuple, set)):
    if any(to_serialize is seen for seen in container_stack):
      raise ValueError('Cycle detected in object dependencies.')
    container_stack.append(to_serialize)
    generator = (_object_to_literal(x, container_stack) for x in to_serialize)
    container_type = type(to_serialize)
    result = container_type(generator)
    container_stack.pop()
    return result

  else:
    raise TypeError('Cannot convert object {} with type {} to a literal.'
                    .format(to_serialize, type(to_serialize)))


def _literal_to_object(literal: Any) -> Any:
  """Turns a literal created by _object_to_literal back into the object."""
  if isinstance(literal, (int, float, bool, str, bytes, type(None))):
    return literal

  elif isinstance(literal, dict):
    # If the dict was not created by _object_to_literal, we may throw KeyError.
    kind = literal[_KIND_KEY]
    if kind == _DTYPE_KIND:
      return getattr(tf, literal['dtype'])
    elif kind == _TENSOR_KIND:
      return tf.constant(literal['content'],
                         dtype=_literal_to_object(literal['dtype']))
    elif kind == _SPARSE_TENSOR_KIND:
      return tf.SparseTensor(
          indices=_literal_to_object(literal['indices']),
          values=_literal_to_object(literal['values']),
          dense_shape=_literal_to_object(literal['dense_shape']))
    elif kind == _DICT_KIND:
      return {_literal_to_object(key): _literal_to_object(value)
              for key, value in literal['dict'].items()}
    else:
      raise ValueError('Unsupported kind in dict: {}'.format(kind))

  elif isinstance(literal, (list, tuple, set)):
    generator = (_literal_to_object(x) for x in literal)
    container_type = type(literal)
    return container_type(generator)

  else:
    raise TypeError('Cannot convert literal {} with type {} to an object.'
                    .format(literal, type(literal)))


def serialize(to_serialize: Any) -> Text:
  """Serializes an object into a string.

  Note: This does not work in Python 2 because its ast.literal_eval does not
  support sets.

  Args:
    to_serialize: The object to serialize. This may be a Python literal (int,
      float, boolean, string, or None), Tensor, SparseTensor, or
      possibly-nested lists/tuples/sets/dicts of these.

  Returns:
    A string representation of the object.
  """
  return repr(_object_to_literal(to_serialize, container_stack=[]))


def parse(serialized: Text) -> Any:
  """Unparses a string into an object (the inverse of serialize_object)."""
  literal = ast.literal_eval(serialized)
  return _literal_to_object(literal)

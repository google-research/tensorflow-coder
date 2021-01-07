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
"""Manages all Operation objects used by value search."""

import inspect
from typing import List, Optional, Text

from tf_coder import tf_functions
from tf_coder.value_search import function_operation
from tf_coder.value_search import operation_base
from tf_coder.value_search import python_operations


def get_python_operations() -> List[operation_base.Operation]:
  """Returns a list of Operation objects from the python_operations module."""
  operation_classes = inspect.getmembers(
      python_operations,
      lambda x: inspect.isclass(x) and not inspect.isabstract(x))
  return [operation_class()
          for unused_name, operation_class in operation_classes]


def get_tf_operations() -> List[operation_base.Operation]:
  """Returns a list of Operation objects for dense TensorFlow operations."""
  return [function_operation.FunctionOperation(function_info)
          for function_info in tf_functions.TF_FUNCTIONS]


def get_sparse_operations() -> List[operation_base.Operation]:
  """Returns a list of Operation objects for sparse operations."""
  return [function_operation.FunctionOperation(function_info)
          for function_info in tf_functions.SPARSE_FUNCTIONS]


def get_operations(
    include_sparse_operations: bool = False) -> List[operation_base.Operation]:
  """Returns a list of Operation objects that value search should use."""
  operations = []
  operations.extend(get_tf_operations())
  if include_sparse_operations:
    operations.extend(get_sparse_operations())
  operations.extend(get_python_operations())
  return operations


def find_operation_with_name(
    operation_name: Text,
    operation_list: Optional[List[operation_base.Operation]] = None
) -> operation_base.Operation:
  """Finds an operation with the given name, optionally within a given list."""
  if operation_list is None:
    operation_list = get_operations(include_sparse_operations=True)
  matching_operations = [op for op in operation_list
                         if op.name == operation_name]
  if len(matching_operations) == 1:
    return matching_operations[0]

  raise ValueError('Found {} operations matching the name {}'.format(
      len(matching_operations), operation_name))

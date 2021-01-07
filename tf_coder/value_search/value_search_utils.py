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
"""Utilities for value search."""

from tf_coder import tensor_limits as limits


def check_tensor_size(tensor_value):
  """Returns whether the tensor is an acceptable size for value search."""
  tensor = tensor_value.value
  return (
      # Check the number of elements.
      tensor_value.num_elements() <= limits.MAX_TENSOR_ELEMENTS and

      # Check the maximum length of a dimension.

      # Note that a TensorShape object for zero dimensions does NOT implicitly
      # convert to False! We really want to check `len(tensor.shape) == 0`,
      # which is NOT equivalent to `not tensor.shape` as suggested by the
      # linter and style guide.
      (not len(tensor.shape) or  # pylint: disable=g-explicit-length-test
       max(tensor.shape) <= limits.MAX_DIMENSION_LENGTH) and

      # Check the number of dimensions.
      len(tensor.shape) <= limits.MAX_NUM_DIMENSIONS)


def check_sparse_tensor_size(sparse_tensor_value):
  """Returns whether the tensor is an acceptable size for value search."""
  sparse_tensor = sparse_tensor_value.value
  return (
      # Check the number of non-default elements.
      sparse_tensor_value.num_elements() <= limits.MAX_TENSOR_ELEMENTS and

      # Check the number of dimensions.
      len(sparse_tensor.dense_shape) <= limits.MAX_NUM_DIMENSIONS)

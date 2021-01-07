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
"""Limits on tensor size and decimal representation."""

# Maximum allowable number of elements in a tensor.
MAX_TENSOR_ELEMENTS = 1000

# Maximum allowable number of dimensions for a tensor.
MAX_NUM_DIMENSIONS = 4

# Maximum length of one dimension of a tensor.
MAX_DIMENSION_LENGTH = 100

# The number of floating-point decimal places to consider.
NUM_DECIMALS = 5

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
"""Tests for random_inputs.py."""

from absl.testing import absltest
from tf_coder.datasets import random_inputs
from tf_coder.value_search import value as value_module


class SeedInputsTest(absltest.TestCase):

  def test_generate_random_input(self):
    for i in range(1000):
      # Just check that generation completes and produces valid Values.
      random_input = random_inputs.generate_random_input(random_seed=i)
      value = value_module.InputValue(random_input, name='dummy_name')
      self.assertIsNotNone(value)

  def test_generate_random_input_uses_seed(self):
    random_input_1a = random_inputs.generate_random_input(random_seed=1)
    random_input_1b = random_inputs.generate_random_input(random_seed=1)
    random_input_2 = random_inputs.generate_random_input(random_seed=2)

    self.assertEqual(str(random_input_1a), str(random_input_1b))
    self.assertNotEqual(str(random_input_1a), str(random_input_2))


if __name__ == '__main__':
  absltest.main()

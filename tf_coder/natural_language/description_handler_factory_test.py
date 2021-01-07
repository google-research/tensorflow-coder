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
"""Tests for description_handler_factory.py."""

from absl.testing import absltest
from tf_coder.natural_language import description_handler_factory


class DescriptionHandlerFactoryTest(absltest.TestCase):

  def test_handler_string_list(self):
    string_list = description_handler_factory.handler_string_list()
    self.assertIn('no_change', string_list)

  def test_create_handler(self):
    handler = description_handler_factory.create_handler('no_change')
    self.assertEqual(repr(handler), 'NoChangeDescriptionHandler')

  def test_create_handler_raises(self):
    with self.assertRaises(ValueError):
      description_handler_factory.create_handler('nonexistent_handler')

if __name__ == '__main__':
  absltest.main()

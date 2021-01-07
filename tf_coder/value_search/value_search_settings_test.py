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
"""Tests for settings_module.py."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from tf_coder.value_search import value_search_settings as settings_module


class ValueSearchSettingsTest(parameterized.TestCase):

  def test_set(self):
    settings = settings_module.Settings()
    settings.set('timeout', 12345)
    settings.set('printing.verbose', True)
    settings.set('tensor_model_prioritize_threshold', 0.9)
    self.assertEqual(settings.timeout, 12345)
    self.assertTrue(settings.printing.verbose)
    self.assertEqual(settings.tensor_model.prioritize_threshold, 0.9)

  def test_set_raises(self):
    settings = settings_module.Settings()
    with self.assertRaises(ValueError):
      settings.set('unknown_setting', True)
    with self.assertRaises(ValueError):
      settings.set('printing.unknown_setting', True)

  def test_from_dict(self):
    overrides = {
        'timeout': 12345,
        'printing.verbose': True,
        'tensor_model_prioritize_threshold': 0.9,
    }
    settings = settings_module.from_dict(overrides)
    self.assertEqual(settings.timeout, 12345)
    self.assertTrue(settings.printing.verbose)
    self.assertEqual(settings.tensor_model.prioritize_threshold, 0.9)

  def test_from_list(self):
    overrides = [
        'timeout=12345',
        'printing.verbose=True',
        'tensor_model_prioritize_threshold=0.9',
    ]
    settings = settings_module.from_list(overrides)
    self.assertEqual(settings.timeout, 12345)
    self.assertTrue(settings.printing.verbose)
    self.assertEqual(settings.tensor_model.prioritize_threshold, 0.9)

  @parameterized.named_parameters(
      ('no_equals', 'timeout: 12345'),
      ('too_many_equals', 'timeout==12345'),
      ('spaces', 'timeout = 12345'),
      ('bad_name', 'unknown_setting=12345'),
      ('bad_value', 'timeout=foo()'))
  def test_from_list_raises(self, override_string):
    with self.assertRaises(ValueError):
      settings_module.from_list([override_string])

  def test_as_dict_default_works_with_json(self):
    settings = settings_module.default_settings()
    as_dict = settings.as_dict()
    self.assertFalse(as_dict['printing.verbose'])
    # Test that all defaults are JSON-serializable.
    self.assertNotEmpty(json.dumps(as_dict))


if __name__ == '__main__':
  absltest.main()

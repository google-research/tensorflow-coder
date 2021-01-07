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
"""Tests for tf_functions.py."""

from absl.testing import absltest
from absl.testing import parameterized
import funcsigs
from tf_coder import filter_group
from tf_coder import tf_coder_utils
from tf_coder import tf_functions

# FunctionInfo names should match this.
FUNCTION_INFO_NAME_REGEX = r'\w+(\.\w+)*\(\w+(, \w+(=[^,=()]+)?)*\)'


class TfFunctionsTest(parameterized.TestCase):

  def _check_function(self, function_name, usable_args, constant_kwargs):
    """Checks for errors in one entry of tf_functions.TF_FUNCTIONS."""
    try:
      func_obj = tf_coder_utils.get_tf_function(function_name)
    except ValueError:
      func_obj = None
    self.assertIsNotNone(func_obj,
                         'Could not find function {}.'.format(function_name))

    self.assertLen(set(usable_args), len(usable_args),
                   'Function {} has duplicate usable arguments.'.format(
                       function_name))

    parameters = funcsigs.signature(func_obj).parameters
    for param_name in parameters:
      param = parameters[param_name]
      if param.default is param.empty:
        self.assertIn(param_name, usable_args,
                      "Function {} is missing required argument '{}'.".format(
                          function_name, param_name))

    ordered_param_names = list(parameters)
    last_index = -1
    for arg_name in usable_args:
      self.assertIn(arg_name, ordered_param_names,
                    "Function {} has invalid argument '{}'.".format(
                        function_name, arg_name))
      cur_index = ordered_param_names.index(arg_name)
      self.assertGreater(cur_index, last_index,
                         "Function {} has argument '{}' out of order.".format(
                             function_name, arg_name))
      last_index = cur_index

    for kwarg_name in constant_kwargs:
      self.assertIn(kwarg_name, ordered_param_names,
                    "Function {} has invalid kwarg '{}'.".format(
                        function_name, kwarg_name))
      self.assertNotIn(kwarg_name, usable_args,
                       "Function {} has repeated argument '{}'.".format(
                           function_name, kwarg_name))

  def test_check_function_passes(self):
    self._check_function('tf.argsort', ['values', 'axis'],
                         {'direction': 'DESCENDING', 'stable': True})

  @parameterized.named_parameters(
      ('bad_function', 'tf.this_function_does_not_exist', ['x', 'y'], {}),
      ('duplicate_args', 'tf.add', ['x', 'y', 'x'], {}),
      ('missing_arg', 'tf.add', ['x'], {}),
      ('invalid_arg', 'tf.add', ['x', 'y', 'z'], {}),
      ('out_of_order', 'tf.add', ['y', 'x'], {}),
      ('duplicate_kwarg', 'tf.argsort', ['values', 'axis'], {'axis': -1}),
      ('invalid_kwarg', 'tf.argsort', ['values', 'axis'], {'invalid': True}))
  def test_check_function_fails(self, function_name, usable_args,
                                constant_kwargs):
    with self.assertRaises(AssertionError):
      self._check_function(function_name, usable_args, constant_kwargs)

  @parameterized.named_parameters(
      ('not_tf', 'np.foo(axis)'),
      ('nested_modules', 'tf.nn.foo(tensor, axis)'),
      ('no_module', 'foo(tensor, axis)'),
      ('string_literal', 'foo(tensor, axis, baz="a constant string")'),
      ('boolean_literal', 'foo(tensor, axis, baz=False)'),
      ('two_literals', 'foo(tensor, bar=[], baz=1.0)'))
  def test_function_info_name_regex_passes(self, good_name):
    self.assertRegex(good_name, FUNCTION_INFO_NAME_REGEX)

  @parameterized.named_parameters(
      ('bad_characters', 'tf.foo(axis=1)'),
      ('extra_spaces_1', 'tf.foo(tensor,  axis)'),
      ('extra_spaces_2', 'tf.foo( tensor, axis)'),
      ('extra_spaces_3', 'tf.foo (tensor, axis)'),
      ('missing_space', 'tf.foo(tensor,axis)'),
      ('missing_middle_arg', 'tf.foo(tensor, , axis)'),
      ('missing_last_arg', 'tf.foo(tensor, )'),
      ('no_args', 'tf.foo()'),
      ('no_parens', 'tf.foo'),
      ('empty_literal', 'tf.foo(a, x=)'),
      ('literal_with_bad_char', 'tf.foo(a, x=",")'))
  def test_function_info_name_regex_fails(self, bad_name):
    self.assertNotRegex(bad_name, FUNCTION_INFO_NAME_REGEX)

  @parameterized.named_parameters(
      ('tf_functions', tf_functions.TF_FUNCTIONS),
      ('sparse_functions', tf_functions.SPARSE_FUNCTIONS))
  def test_function_lists(self, function_list):
    for function_info in function_list:
      self.assertRegex(function_info.name, FUNCTION_INFO_NAME_REGEX)
      self.assertIsInstance(function_info.filter_group,
                            filter_group.FilterGroup)
      self.assertIsInstance(function_info.weight, int)

      function_name, usable_args, constant_kwargs = (
          tf_functions.parse_function_info_name(function_info))
      self._check_function(function_name, usable_args, constant_kwargs)

  def test_parse_function_info_name(self):
    function_info = tf_functions.FunctionInfo(
        name='tf.foo.bar(tensor, axis, baz=True)',
        filter_group=filter_group.FilterGroup.NONE,
        weight=1)
    self.assertEqual(tf_functions.parse_function_info_name(function_info),
                     ('tf.foo.bar', ['tensor', 'axis'], {'baz': True}))

  @parameterized.named_parameters(
      ('no_open_paren', 'tf.foo.bar tensor, axis)'),
      ('multiple_open_parens', 'tf.foo.bar((tensor, axis)'),
      ('no_close_paren', 'tf.foo.bar(tensor, axis'),
      ('close_paren_not_at_end', 'tf.foo.bar(tensor, axis) '))
  def test_parse_function_info_name_fails_for_bad_name(self, bad_name):
    function_info = tf_functions.FunctionInfo(
        name=bad_name,
        filter_group=filter_group.FilterGroup.NONE,
        weight=1)
    with self.assertRaises(ValueError):
      tf_functions.parse_function_info_name(function_info)


if __name__ == '__main__':
  absltest.main()

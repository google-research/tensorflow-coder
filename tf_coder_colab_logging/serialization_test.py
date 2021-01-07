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
"""Tests for serialization.py."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from tf_coder_colab_logging import serialization


class SerializationTest(parameterized.TestCase):

  def test_round_trip(self):
    to_serialize = [
        [1, 2, 3],
        [[1.2, 3.4], [5.6, 7.8]],
        'Description with \'quotes" and unicode \u1234',
        tf.constant([1, 2, 3], dtype=tf.int16),
        tf.constant(4.5),
        tf.constant('tf-coder'),
        tf.float16,
        tf.string,
        {'my': 'dict', 1: 2},
        {'my', 'set', 1, 0, (True, False)},
        (True, False),
        None,
        tf.sparse.from_dense(tf.constant([[0, 1, 0], [2, 0, 0], [0, 3, 4]])),
    ]
    serialized = serialization.serialize(to_serialize)
    parsed = serialization.parse(serialized)

    self.assertLen(to_serialize, len(parsed))
    for before, after in zip(to_serialize, parsed):
      self.assertEqual(type(before), type(after))
      if isinstance(before, (dict, set)):
        # Do not use string comparison because the order of elements may change.
        # Make sure that the test dict and set do not contain tensors, as they
        # will cause the == comparison to fail!
        self.assertEqual(before, after)
      else:
        # Two "identical" tensors with different IDs will not be equal (==). So,
        # compare str representations. The repr of a tensor contains its id, but
        # the str representation does not.
        self.assertEqual(str(before), str(after))

  @parameterized.named_parameters(
      ('tf_function', tf.reduce_max),
      ('tf_module', tf))
  def test_serialize_raises_for_unsupported_objects(self, unsupported_object):
    # Weird objects result in TypeError.
    with self.assertRaises(TypeError):
      serialization.serialize(unsupported_object)

  def test_serialize_detects_cycles(self):
    outer = [1]
    inner = [2]
    outer.append(inner)
    outer.append(inner)
    # No cycle yet, even though `inner` is used multiple times.
    self.assertEqual(serialization.serialize(outer), '[1, [2], [2]]')

    # Introduce cycle.
    inner.append(outer)
    with self.assertRaises(ValueError):
      serialization.serialize(outer)

if __name__ == '__main__':
  absltest.main()

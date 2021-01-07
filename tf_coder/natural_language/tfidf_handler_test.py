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
"""Tests for tfidf_handler.py."""

from absl.testing import absltest
from absl.testing import parameterized
import mock
import numpy as np
from tf_coder.natural_language import tfidf_handler
from tf_coder.value_search import value_search_settings as settings_module


def _create_operation(name, docstring):
  """Creates a dummy operation with the given name and docstring."""
  # For the purposes of this test, an Operation is any object with `.name` and
  # `.metadata.docstring` attributes.
  operation = mock.Mock()
  operation.name = name
  operation.metadata = mock.Mock()
  operation.metadata.docstring = docstring
  return operation


class TfidfHandlerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('tile', 'Tile a tensor multiple times', 'tf.tile(input, multiples)'),
      ('tensordot', 'Something like np.tensordot', 'tf.tensordot(a, b, axes)'))
  def test_score_description_best_operation(self, description, best_operation):
    handler = tfidf_handler.TfidfDescriptionHandler()
    score_dict = handler.score_description(description)
    self.assertTrue(all(0 <= value <= 1 for value in score_dict.values()))
    max_score = max(score_dict.values())
    self.assertEqual(score_dict[best_operation], max_score)

  def test_score_description_exact_values(self):
    # This example is taken from the following link:
    # https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
    counts = [[3, 0, 1],
              [2, 0, 0],
              [3, 0, 0],
              [4, 0, 0],
              [3, 2, 0],
              [3, 0, 2]]
    docstrings = [' '.join(['apple'] * i + ['banana'] * j + ['clementine'] * k)
                  for i, j, k in counts]
    operations = [_create_operation('operation_{}'.format(i), docstring)
                  for i, docstring in enumerate(docstrings)]
    handler = tfidf_handler.TfidfDescriptionHandler(operations=operations)

    # Same counts as docstrings[0]. "dragonfruit" is not in the vocabulary.
    description = 'apple clementine apple dragonfruit apple'

    scores = handler.score_description(description)
    sorted_scores = [scores[name] for name in sorted(scores.keys())]

    # Keep the formatting as in the link above. pylint: disable=bad-whitespace
    expected_term_document_matrix = [[0.85151335, 0.        , 0.52433293],
                                     [1.        , 0.        , 0.        ],
                                     [1.        , 0.        , 0.        ],
                                     [1.        , 0.        , 0.        ],
                                     [0.55422893, 0.83236428, 0.        ],
                                     [0.63035731, 0.        , 0.77630514]]
    # pylint: enable=bad-whitespace

    expected_scores = [np.dot(expected_term_document_matrix[0], row)
                       for row in expected_term_document_matrix]

    self.assertLen(sorted_scores, len(expected_scores))
    for actual_score, expected_score in zip(sorted_scores, expected_scores):
      self.assertAlmostEqual(actual_score, expected_score)

  @parameterized.named_parameters(
      ('0', 0), ('1', 1), ('2', 2), ('5', 5), ('10', 10))
  def test_get_operation_multiplier_respects_max_num_prioritized(
      self, max_num_prioritized):
    handler = tfidf_handler.TfidfDescriptionHandler(
        max_num_prioritized=max_num_prioritized,
        min_tfidf_score=0)

    multipliers = handler.get_operation_multipliers(
        'Tile a tensor multiple times',
        settings=settings_module.default_settings())

    actual_num_prioritized = sum(multiplier < 1
                                 for multiplier in multipliers.values())
    self.assertEqual(actual_num_prioritized, max_num_prioritized)
    # All multipliers must be in (0, 1] (no operation is deprioritized).
    self.assertTrue(all(0 < multiplier <= 1
                        for multiplier in multipliers.values()))

  @parameterized.named_parameters(
      ('0', 0.0), ('0_1', 0.1), ('0_2', 0.2), ('1_0', 1.0))
  def test_get_operation_multipliers_respects_min_tfidf_score(
      self, min_tfidf_score):
    handler = tfidf_handler.TfidfDescriptionHandler(
        max_num_prioritized=1000000,
        min_tfidf_score=min_tfidf_score)

    description = 'Tile a tensor multiple times'
    scores = handler.score_description(description)
    multipliers = handler.get_operation_multipliers(
        description, settings=settings_module.default_settings())

    prioritized_names = [name for name in multipliers.keys()
                         if multipliers[name] < 1]
    expected_prioritized_names = [name for name in scores.keys()
                                  if scores[name] >= min_tfidf_score]
    self.assertCountEqual(prioritized_names, expected_prioritized_names)
    # All multipliers must be in (0, 1] (no operation is deprioritized).
    self.assertTrue(all(0 < multiplier <= 1
                        for multiplier in multipliers.values()))

  def test_repr(self):
    handler = tfidf_handler.TfidfDescriptionHandler(max_num_prioritized=12,
                                                    min_tfidf_score=0.34,
                                                    multiplier=0.75)
    self.assertEqual(repr(handler),
                     'TfidfDescriptionHandler(max_num_prioritized=12, '
                     'min_tfidf_score=0.34, multiplier=0.75)')

if __name__ == '__main__':
  absltest.main()

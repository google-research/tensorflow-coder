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
"""TF-IDF for similarity between task descriptions and operation docstrings."""

from typing import Dict, List, Optional, Text

from sklearn.feature_extraction import text as sklearn_text
from sklearn.metrics import pairwise as sklearn_pairwise
from tf_coder.natural_language import description_handler
from tf_coder.value_search import operation_base
from tf_coder.value_search import value_search_settings as settings_module


class TfidfDescriptionHandler(description_handler.DescriptionHandler):
  """Computes similarity between task descriptions and operation docstrings."""

  def __init__(self,
               operations: Optional[List[operation_base.Operation]] = None,
               max_num_prioritized: int = 5,
               min_tfidf_score: float = 0.15,
               multiplier: float = 0.75):
    """Creates a TfidfDescriptionHandler that can score the given operations.

    Args:
      operations: A list of operations that the scorer should handle. Exposed
        for testing.
      max_num_prioritized: The maximum number of operations to prioritize (these
        are assigned a delta of -1).
      min_tfidf_score: The minimum TF-IDF score for an operation to be
        prioritized.
      multiplier: The multiplier applied to an operation's weight if it is
        prioritized.

    Raises:
      ValueError: If there are duplicate operation names.
    """
    super(TfidfDescriptionHandler, self).__init__(operations)
    self.max_num_prioritized = max_num_prioritized
    self.min_tfidf_score = min_tfidf_score
    self.multiplier = multiplier

    docstrings = [operation.metadata.docstring
                  for operation in self.operations]
    self._vectorizer = sklearn_text.TfidfVectorizer(stop_words='english')
    self._term_document_matrix = self._vectorizer.fit_transform(docstrings)

  def score_description(self, description: Text) -> Dict[Text, float]:
    """Returns a map from operation names to their similarity to `description`.

    Similarity scores will be in the range [0, 1], with higher scores indicating
    more relevance.

    Args:
      description: An English description of a task.
    """
    description_vector = self._vectorizer.transform([description])
    scores = sklearn_pairwise.cosine_similarity(
        description_vector, self._term_document_matrix).flatten()
    return dict(zip(self.all_names, scores))

  def get_operation_multipliers(
      self,
      description: Text,
      settings: settings_module.Settings) -> Dict[Text, float]:
    """See base class."""
    scores = self.score_description(description)
    # Sorted in order of decreasing TF-IDF score.
    sorted_names = sorted(self.all_names,
                          key=lambda name: scores[name], reverse=True)
    prioritized_set = set(name
                          for name in sorted_names[:self.max_num_prioritized]
                          if scores[name] >= self.min_tfidf_score)
    if settings.printing.prioritized_operations:
      for name in prioritized_set:
        print('TF-IDF handler prioritized {}, score={:.3f}'.format(
            name, scores[name]))

    return dict((name, self.multiplier) for name in prioritized_set)

  def __repr__(self) -> Text:
    """See base class."""
    return ('{}(max_num_prioritized={}, min_tfidf_score={}, multiplier={})'
            .format(self.__class__.__name__,
                    self.max_num_prioritized,
                    self.min_tfidf_score,
                    self.multiplier))

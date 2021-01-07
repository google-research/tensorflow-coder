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
"""Creates description handlers from strings."""

import collections
import functools
from typing import Callable, Dict, List, Text

from tf_coder.natural_language import bag_of_words_handlers
from tf_coder.natural_language import description_handler
from tf_coder.natural_language import tfidf_handler


# Use lambdas to avoid instantiating handlers until they're used.
# pylint: disable=g-long-lambda
DESCRIPTION_HANDLER_FNS = collections.OrderedDict([
    ('no_change',
     description_handler.NoChangeDescriptionHandler),
    ('function_name',
     description_handler.FunctionNameDescriptionHandler),
    ('tfidf',
     tfidf_handler.TfidfDescriptionHandler),  # Use its defaults.
    ('naive_bayes',
     bag_of_words_handlers.NaiveBayesDescriptionHandler),  # Use its defaults.
    ('svm',
     bag_of_words_handlers.LinearSVCDescriptionHandler),  # Use its defaults.

    ('tfidf_3_0.1',
     functools.partial(tfidf_handler.TfidfDescriptionHandler,
                       max_num_prioritized=3,
                       min_tfidf_score=0.1)),
    ('tfidf_3_0.15',
     functools.partial(tfidf_handler.TfidfDescriptionHandler,
                       max_num_prioritized=3,
                       min_tfidf_score=0.15)),
    ('naive_bayes_3_0.5',
     functools.partial(bag_of_words_handlers.NaiveBayesDescriptionHandler,
                       max_num_prioritized=3,
                       min_probability=0.5)),
    ('naive_bayes_3_0.75',
     functools.partial(bag_of_words_handlers.NaiveBayesDescriptionHandler,
                       max_num_prioritized=3,
                       min_probability=0.75)),

    ('tfidf_5_0.1',
     functools.partial(tfidf_handler.TfidfDescriptionHandler,
                       max_num_prioritized=5,
                       min_tfidf_score=0.1)),
    ('tfidf_5_0.15',
     functools.partial(tfidf_handler.TfidfDescriptionHandler,
                       max_num_prioritized=5,
                       min_tfidf_score=0.15)),
    ('naive_bayes_5_0.5',
     functools.partial(bag_of_words_handlers.NaiveBayesDescriptionHandler,
                       max_num_prioritized=5,
                       min_probability=0.5)),
    ('naive_bayes_5_0.75',
     functools.partial(bag_of_words_handlers.NaiveBayesDescriptionHandler,
                       max_num_prioritized=5,
                       min_probability=0.75)),

    ('tfidf_10_0.1',
     functools.partial(tfidf_handler.TfidfDescriptionHandler,
                       max_num_prioritized=10,
                       min_tfidf_score=0.1)),
    ('tfidf_10_0.15',
     functools.partial(tfidf_handler.TfidfDescriptionHandler,
                       max_num_prioritized=10,
                       min_tfidf_score=0.15)),
    ('naive_bayes_10_0.5',
     functools.partial(bag_of_words_handlers.NaiveBayesDescriptionHandler,
                       max_num_prioritized=10,
                       min_probability=0.5)),
    ('naive_bayes_10_0.75',
     functools.partial(bag_of_words_handlers.NaiveBayesDescriptionHandler,
                       max_num_prioritized=10,
                       min_probability=0.75)),
])  # type: Dict[Text, Callable[[], description_handler.DescriptionHandler]]
# pylint: enable=g-long-lambda


def handler_string_list() -> List[Text]:
  """Returns a list of available handler strings."""
  return list(DESCRIPTION_HANDLER_FNS.keys())


def create_handler(
    handler_string: Text) -> description_handler.DescriptionHandler:
  """Returns a DescriptionHandler corresponding to the given handler string."""
  if handler_string not in DESCRIPTION_HANDLER_FNS:
    raise ValueError('Unknown description handler: {}'
                     .format(handler_string))
  # Evaluate the lambda to get the handler.
  return DESCRIPTION_HANDLER_FNS[handler_string]()

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
"""Bag-of-words description handlers for TF-Coder."""

import pickle
from typing import Dict, List, Optional, Text

from sklearn import ensemble
from sklearn import feature_extraction
from sklearn import naive_bayes
from sklearn import svm
from tf_coder.datasets.github import data_loader
from tf_coder.datasets.github import tokenizer
from tf_coder.natural_language import description_handler
from tf_coder.value_search import operation_base
from tf_coder.value_search import value_search_settings as settings_module

STOPWORDS = 'english'


def _save_model(filepath, vectorizer, classifiers):
  with open(filepath, 'wb') as f:
    pickle.dump([vectorizer, classifiers], f)


def _load_model(filepath):
  with open(filepath, 'rb') as f:
    vectorizer, classifiers = pickle.load(f)
  return vectorizer, classifiers


class BagOfWordsDescriptionHandler(description_handler.DescriptionHandler):
  """Fit a classifier to estimate the probability of an op's usage."""

  def __init__(self,
               operations: Optional[List[operation_base.Operation]] = None,
               max_features=None,
               make_classifier=None,
               lowercase: bool = True,
               sublinear_tf: bool = True,
               max_num_prioritized: int = 3,
               min_probability: float = 0.5,
               use_docstrings_for_tfidf=False,
               use_docstrings_as_examples=True,
               multiplier: float = 0.75,
               path_prefix: Text = None,
               model_filepath=None,
               load: bool = False,
               save: bool = False):
    """Creates a BagOfWordsDescriptionHandler.

    Args:
      operations: A list of operations that the handler should handle.
      max_features: If not None, the max size of the vocabulary to construct in
        the vectorizer.
      make_classifier: A function to produce an untrained classifier, which
        will then be fit to the data.
      lowercase: Whether to lowercase tokens as part of preprocessing.
      sublinear_tf: Whether to use sublinear tfs in the tfidf calculation.
      max_num_prioritized: The maximum number of operations to prioritize.
      min_probability: The minimum score for an operation to be prioritized.
      use_docstrings_for_tfidf: Whether to initialize the tf-idf vectorizer
        using tf operation docstrings.
      use_docstrings_as_examples: Whether to include additional examples, one
        per opperation docstring.
      multiplier: The multiplier applied to an operation's weight if it is
        prioritized.
      path_prefix: The file path prefix from which to load the GitHub data.
      model_filepath: The model filepath to use for saving or loading.
      load: If True, loads a saved model.
      save: If True, pickles the model for later use.
    """
    super(BagOfWordsDescriptionHandler, self).__init__(operations)
    self.path_prefix = path_prefix or data_loader.DEFAULT_DATA_PREFIX
    self.data = data_loader.load_data(self.path_prefix)

    self.max_num_prioritized = max_num_prioritized
    self.min_probability = min_probability
    self.multiplier = multiplier

    self.predict_fns = []
    if load:
      # Load an existing model.
      self.vectorizer, classifiers = _load_model(model_filepath)
      for classifier in classifiers:
        self.predict_fns.append(make_predict_fn(classifier))
      return

    if use_docstrings_as_examples:
      extra_examples = [{  # pylint: disable=g-complex-comprehension
          'docstring': [operation.metadata.docstring],
          'names': [],
          'comments': [],
          'strings': [],
          'tf_functions': [operation.name.split('(')[0]],
      } for operation in self.operations]
      self.data += extra_examples

    # Generate a new model.
    self.vectorizer = feature_extraction.text.TfidfVectorizer(
        stop_words=STOPWORDS,
        max_features=max_features,
        lowercase=lowercase,
        sublinear_tf=sublinear_tf,
    )
    contexts_list = [
        data_loader.get_full_context(example)
        for example in self.data]
    docstrings = [operation.metadata.docstring
                  for operation in self.operations]
    if use_docstrings_for_tfidf:
      self.vectorizer.fit(docstrings)
    else:
      self.vectorizer.fit(contexts_list)
    contexts_features = self.vectorizer.transform(contexts_list)

    self.classifiers = []
    for op_name in self.all_names:
      op = op_name.split('(')[0]
      targets = [data_loader.uses_operation(example, op)
                 for example in self.data]

      classifier = make_classifier()
      try:
        classifier.fit(contexts_features, targets)
      except ValueError:
        # For classes without positive examples, some classifiers cannot be fit.
        classifier = None

      self.classifiers.append(classifier)
      self.predict_fns.append(make_predict_fn(classifier))

    if save:
      _save_model(model_filepath, self.vectorizer, self.classifiers)

  def score_description(self, description: Text) -> Dict[Text, float]:
    description_as_context = ' '.join(tokenizer.tokenize(description))
    vectorized = self.vectorizer.transform([description_as_context])
    probas = {}
    for op_name, predict_fn in zip(self.all_names, self.predict_fns):
      proba = predict_fn(vectorized)
      probas[op_name] = proba
    return probas

  def get_operation_multipliers(
      self,
      description: Text,
      settings: settings_module.Settings) -> Dict[Text, float]:
    """See base class."""
    scores = self.score_description(description)

    sorted_names = sorted(self.all_names,
                          key=lambda name: scores[name], reverse=True)
    prioritized_set = set(name
                          for name in sorted_names[:self.max_num_prioritized]
                          if scores[name] >= self.min_probability)

    if settings.printing.prioritized_operations:
      for name in prioritized_set:
        print('BOW handler prioritized {}, score={:.3f}'.format(
            name, scores[name]))
    return dict((name, self.multiplier) for name in prioritized_set)

  def __repr__(self) -> Text:
    """See base class."""
    return ('{}(path_prefix={}, min_probability={}, max_num_prioritized={}, '
            'multiplier={})').format(
                self.__class__.__name__,
                self.path_prefix,
                self.min_probability,
                self.max_num_prioritized,
                self.multiplier,
            )


def make_predict_fn(classifier):
  """Produces a single-example predict_proba function from a classifier."""
  if not classifier:
    return lambda v: 0.0

  def predict(vectorized):
    try:
      return classifier.predict_proba(vectorized)[0, 1]
    except:  # pylint: disable=bare-except
      return float(classifier.predict(vectorized)[0])
  return predict


# pylint: disable=invalid-name
def NaiveBayesDescriptionHandler(alpha=0.25, max_features=None,
                                 sublinear_tf=True, max_num_prioritized=3,
                                 min_probability=0.5):
  return BagOfWordsDescriptionHandler(
      max_features=max_features,
      sublinear_tf=sublinear_tf,
      make_classifier=
      lambda: naive_bayes.MultinomialNB(alpha=alpha, class_prior=(.5, .5)),
      max_num_prioritized=max_num_prioritized,
      min_probability=min_probability,
  )


def LinearSVCDescriptionHandler(C=0.0625, max_features=None,
                                max_num_prioritized=3,
                                min_probability=0.5):
  return BagOfWordsDescriptionHandler(
      max_features=max_features,
      make_classifier=
      lambda: svm.LinearSVC(C=C, dual=True, max_iter=500),
      max_num_prioritized=max_num_prioritized,
      min_probability=min_probability,
  )


def _make_ensemble(classifiers, voting='hard'):
  return ensemble.VotingClassifier(
      [('c' + str(i), classifier)
       for i, classifier in enumerate(classifiers)],
      voting=voting
  )


def EnsembleLinearSVCDescriptionHandler(C=0.0625, max_features=None, n=10):
  return BagOfWordsDescriptionHandler(
      max_features=max_features,
      make_classifier=
      # pylint: disable=g-long-lambda
      lambda: _make_ensemble([svm.LinearSVC(C=C, dual=True, max_iter=500)
                              for _ in range(n)]),
      # pylint: enable=g-long-lambda
  )


def EnsembleNaiveBayesDescriptionHandler(alpha=0.25, max_features=None, n=10):
  return BagOfWordsDescriptionHandler(
      max_features=max_features,
      make_classifier=
      # pylint: disable=g-long-lambda
      lambda: _make_ensemble([
          naive_bayes.MultinomialNB(alpha=alpha, fit_prior=True)
          for _ in range(n)]),
      # pylint: enable=g-long-lambda
  )
# pylint: enable=invalid-name

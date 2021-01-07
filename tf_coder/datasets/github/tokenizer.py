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
"""Tokenization of docstrings, comments, variable names, and string literals."""

import re
import string
from typing import List, Text


def camel_case_split(text: Text) -> List[Text]:
  matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                        text)
  return [m.group(0) for m in matches]


def normalize(text: Text) -> Text:
  """Normalizes text to a canonical form for use in classifiers."""
  # Strip quotes.
  text = text.replace('"', '').replace("'", '')

  # Remove non-alphanumeric characters.
  text = ''.join([
      char if char in string.ascii_letters + string.digits else ' '
      for char in text
  ])
  tokens = text.split()

  # Split camel-case tokens.
  subtokens = []
  for token in tokens:
    subtokens.extend(camel_case_split(token))

  # Lower case.
  return ' '.join(subtokens).lower()


def tokenize(text: Text) -> List[Text]:
  return normalize(text).split()


def tokens_from_text_list(texts: List[Text]) -> List[Text]:
  tokens = []
  for text in texts:
    tokens.extend(normalize(text).split())
  return tokens

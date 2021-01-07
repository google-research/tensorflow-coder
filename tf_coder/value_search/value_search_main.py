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
"""Exhaustive value search (enumerating by weight of expression)."""

import gc
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Must happen before importing tf.
import sys  # pylint: disable=g-import-not-at-top

from absl import app
from absl import flags
import numpy as np
from scipy.stats import mstats
from tf_coder.benchmarks import all_benchmarks
from tf_coder.benchmarks import autopandas_benchmarks
from tf_coder.benchmarks import google_benchmarks
from tf_coder.benchmarks import stackoverflow_benchmarks
from tf_coder.models import tensor_features_model
from tf_coder.natural_language import description_handler_factory
from tf_coder.value_search import value_search
from tf_coder.value_search import value_search_settings as settings_module


FLAGS = flags.FLAGS

flags.DEFINE_string('benchmark_name', 'ALL',
                    'The name of a benchmark to run, or "ALL".')
flags.DEFINE_multi_string('settings',
                          ['timeout=300'],
                          'Settings to override the defaults.')
flags.DEFINE_string(
    'json_output',
    '',
    'Where the results json file should be written. It will only be written if '
    'all benchmarks are run. Use an empty string to avoid writing this file.')
flags.DEFINE_string(
    'notes',
    '',
    'Any notes to write into the results json file.')


def benchmark_name_validator(benchmark_name):
  """Checks that benchmark_name is "ALL", or refers to exactly one benchmark."""
  return (benchmark_name == 'ALL' or
          all_benchmarks.find_benchmark_with_name(benchmark_name) is not None)

flags.register_validator('benchmark_name', benchmark_name_validator,
                         message=('benchmark_name must be "ALL" or refer to '
                                  'exactly one benchmark.'))




def run_on_all_benchmarks():
  """Runs value search on all benchmarks, printing results to stdout."""

  benchmark_count = 0
  benchmark_success = 0
  unsolved_benchmarks = []
  solution_times = []  # Only including successful tasks.

  settings = settings_module.from_list(FLAGS.settings)

  description_handler = description_handler_factory.create_handler(
      settings.description_handler_name)
  print('Description handler: {!r}\n'.format(description_handler))

  results_json = {
      'benchmark_name': FLAGS.benchmark_name,
      'settings': settings.as_dict(),
      'notes': FLAGS.notes,
      'results': [],
  }

  if (settings.tensor_model.config_path and
      settings.tensor_model.checkpoint_path):
    tensor_config = tensor_features_model.load_config(
        settings.tensor_model.config_path)
    tensor_model = tensor_features_model.get_model(tensor_config)
    checkpoint = tensor_features_model.create_checkpoint(tensor_model)
    checkpoint.restore(settings.tensor_model.checkpoint_path).expect_partial()

    # Warm up. Running the model for the first time takes an extra ~10 seconds.
    print('Warming up the tensor features model...')
    value_search.operation_multipliers_from_tensor_model(
        all_benchmarks.find_benchmark_with_name('simple_cast'),
        tensor_model, tensor_config, settings)
    print('Finished warming up.')
  else:
    tensor_config = None
    tensor_model = None

  print('=' * 80)
  modules = [google_benchmarks, stackoverflow_benchmarks, autopandas_benchmarks]
  for benchmark in all_benchmarks.get_chosen_benchmarks(
      FLAGS.benchmark_name, modules=modules):
    gc.collect()

    print('Performing value search for benchmark {}.\n'
          .format(benchmark.name))
    benchmark_count += 1

    result = value_search.run_value_search(
        benchmark=benchmark,
        settings=settings,
        description_handler=description_handler,
        tensor_model=tensor_model,
        tensor_config=tensor_config)

    if settings.printing.statistics:
      print('\nOperation statistics:\n{}'.format(
          result.statistics.statistics_as_string(
              num_unique_values=len(result.value_set),
              elapsed_time=result.total_time,
              sort_by_time=settings.printing.statistics_sort_by_time)))

    solutions = result.solutions
    if solutions:
      first_solution = solutions[0]
      print('\nBest solution of weight {} found in {:.2f} sec:\n{}'.format(
          first_solution.weight, first_solution.time,
          first_solution.expression))
      benchmark_success += 1
      solution_times.append(first_solution.time)
    else:
      unsolved_benchmarks.append(benchmark)
    print('=' * 80)
    sys.stdout.flush()

    results_json['results'].append({
        'name': benchmark.name,
        'solved': bool(solutions),
        'solution': solutions[0].expression if solutions else None,
        'solution_weight': solutions[0].weight if solutions else None,
        'time': solutions[0].time if solutions else None,
    })

  solve_time_total = sum(solution_times)
  solve_time_mean = np.mean(solution_times)
  solve_time_geometric_mean = mstats.gmean(solution_times)

  results_json['num_benchmarks'] = benchmark_count
  results_json['num_solved'] = benchmark_success
  results_json['solve_time_total'] = solve_time_total
  results_json['solve_time_mean'] = solve_time_mean
  results_json['solve_time_geometric_mean'] = solve_time_geometric_mean

  print('Solved {} out of {} benchmarks in {:.2f} sec.'.format(
      benchmark_success, benchmark_count, solve_time_total))
  print('\n'
        'Arithmetic mean of solve times: {:.2f} sec\n'
        'Geometric mean of solve times: {:.2f} sec\n'.format(
            solve_time_mean, solve_time_geometric_mean))

  print('Unsolved benchmarks:')
  for unsolved in unsolved_benchmarks:
    print('Name: {}, target program: {}'.format(
        unsolved.name, unsolved.target_program))
  print()

  if FLAGS.json_output and FLAGS.benchmark_name == 'ALL':
    with open(FLAGS.json_output, 'w') as json_file:
      json.dump(results_json, json_file,
                indent=4, sort_keys=True, separators=(',', ': '))
      json_file.write('\n')
    print('Wrote JSON results to {}.'.format(FLAGS.json_output))
  else:
    print('Did not write JSON results file.')


def main(unused_argv):

  run_on_all_benchmarks()

if __name__ == '__main__':
  app.run(main)

# Copyright 2018 The TensorFlow Authors.
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

"""Functions for training models with the TensorFlow Estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import collections
import tensorflow as tf

def evaluate(estimator, input_fn, eval_steps=None, eval_name="val"):
    """Runs evaluation on the latest models checkpoint.

    Args:
      estimator: Instance of tf.Estimator.
      input_fn: Input function returning a tuple (features, labels).
      eval_steps: The number of steps for which to evaluate the models. If None,
          evaluates until input_fn raises an end-of-input exception.
      eval_name: Name of the evaluation set, e.g. "train" or "val".

    Returns:
      A dict of metric values from the evaluation. May be empty, e.g. if the
      training job has not yet saved a checkpoint or the checkpoint is deleted by
      the time the TPU worker initializes.
    """
    values = {}  # Default return value if evaluation fails.

    latest_checkpoint = tf.train.latest_checkpoint(estimator.model_dir)
    if not latest_checkpoint:
        # This is expected if the training job has not yet saved a checkpoint.
        return values

    tf.logging.info("Starting evaluation on checkpoint %s", latest_checkpoint)
    try:
        values = estimator.evaluate(input_fn, steps=eval_steps, name=eval_name, checkpoint_path=latest_checkpoint)
    except tf.errors.NotFoundError:
        # Expected under some conditions, e.g. TPU worker does not finish
        # initializing until long after the CPU job tells it to start evaluating
        # and the checkpoint file is deleted already.
        tf.logging.info("Checkpoint %s no longer exists, skipping evaluation",
                        latest_checkpoint)

    return values


def continuous_train_and_eval(estimator,
                              train_input_fn,
                              eval_input_fn,
                              local_eval_frequency=None,
                              train_hooks=None,
                              train_steps=None,
                              eval_steps=None,
                              eval_name="eval"):
    """Alternates training and evaluation.

    Args:
      estimator: Instance of tf.Estimator.
      train_input_fn: Input function returning a tuple (features, labels).
      eval_input_fn: Input function returning a tuple (features, labels).
      local_eval_frequency: The number of training steps between evaluations. If
          None, trains until train_input_fn raises an end-of-input exception.
      train_hooks: List of SessionRunHook subclass instances. Used for callbacks
          inside the training call.
      train_steps: The total number of steps to train the models for.
      eval_steps: The number of steps for which to evaluate the models. If None,
          evaluates until eval_input_fn raises an end-of-input exception.
      eval_name: Name of the evaluation set, e.g. "train" or "val".

    Yields:
      A dict of metric values from each evaluation. May be empty, e.g. if the
      training job has not yet saved a checkpoint or the checkpoint is deleted by
      the time the TPU worker initializes.
    """
    while True:
        # We run evaluation before training in this loop to prevent evaluation from
        # being skipped if the process is interrupted.
        values = evaluate(estimator, eval_input_fn, eval_steps, eval_name)
        yield values

        global_step = values.get("global_step", 0)
        if train_steps and global_step >= train_steps:
            break

        # Decide how many steps before the next evaluation.
        steps = local_eval_frequency
        if train_steps:
            remaining_steps = train_steps - global_step
            steps = min(steps, remaining_steps) if steps else remaining_steps

        tf.logging.info("Starting training at global step %d", global_step)
        estimator.train(train_input_fn, hooks=train_hooks, steps=steps)

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)

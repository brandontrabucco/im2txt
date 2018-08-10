# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Evaluate the model.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time

import json
import pickle as pkl
import numpy as np
import tensorflow as tf
import glove.configuration
import glove

from im2txt import configuration
from im2txt import show_and_tell_model
from pycocoapi.coco import COCO
from pycocoapi.eval import COCOEvalCap

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("wikipedia_file_pattern", "",
                       "File pattern of sharded TFRecord wikipedia files.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", "", 
                       "Directory to write event logs.")
tf.flags.DEFINE_string("annotations_file", "", 
                       "The captions annotations json file.")

tf.flags.DEFINE_integer("eval_interval_secs", 1800,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 10132,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 0,
                        "Minimum global step to run evaluation.")
tf.flags.DEFINE_integer("max_eval_batches", 10,
                        "Maximum number batches to run evaluation.")
tf.flags.DEFINE_integer("style_iterations", 20,
                        "Maximum number batches to run evaluation.")

tf.logging.set_verbosity(tf.logging.INFO)


def coco_get_metrics(time_now, global_step, json_dump):
    """Get the performance metrics on the dataset.
    """
    
    # Output a temporary results file for evaluation
    with open(
            os.path.join(FLAGS.eval_dir, "style.results." + str(time_now) + ".json"), 
            "w") as f:
        json.dump(json_dump, f)
    
    # Evaluate the results file
    coco = COCO(FLAGS.annotations_file)
    cocoRes = coco.loadRes(os.path.join(FLAGS.eval_dir, "style.results." + str(time_now) + ".json"))
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # Dump the results to a metrics file
    with open(
            os.path.join(FLAGS.eval_dir, "style.metrics." + str(time_now) + ".json"),
            "w") as f:
        metrics_dump = {metric: float(np.sum(score)) for metric, score in cocoEval.eval.items()}
        metrics_dump["global_step"] = int(np.sum(global_step))
        json.dump(metrics_dump, f)
        
    return metrics_dump


def ids_to_sentence(word_ids, vocab):
    """Convert sequence of ids to a sentence using thsi vocab.
    """
    generated_caption = ""
    for w in word_ids:
        
        w = str(vocab.id_to_word(w))
        if w == "</S>":
            break
            
        generated_caption += w + " "
        if w == ".":
            break
            
    return generated_caption


def evaluate_model(sess, model, global_step, summary_writer, summary_op):
    """Computes perplexity-per-word over the evaluation dataset.

    Summaries and perplexity-per-word are written out to the eval directory.

    Args:
        sess: Session object.
        model: Instance of ShowAndTellModel; the model to evaluate.
        global_step: Integer; global step of the model checkpoint.
        summary_writer: Instance of FileWriter.
        summary_op: Op for generating model summaries.
    """
    # Log model summaries on a single batch.
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, global_step)

    time_now = int(time.time())
    vocab = glove.load(model.config.config)[0]

    # Compute perplexity over the entire dataset.
    num_eval_batches = min(FLAGS.max_eval_batches, int(
        math.ceil(FLAGS.num_eval_examples / model.config.batch_size)))

    start_time = time.time()
    sum_losses = 0.
    sum_weights = 0.

    unique_image_ids = set()
    json_dump = []
    comparison_dump = []

    for i in range(num_eval_batches):
        
        tf.logging.info("Starting beam_search on batch %d", i)
        (global_step,
            images,
            image_ids,
            target_seqs,
            final_seqs, _) = sess.run([
                model.global_step,
                model.images,
                model.image_ids,
                model.target_seqs,
                model.final_seqs,
                model.assign_initial_states
            ])
        for x in range(FLAGS.style_iterations):
            sess.run(model.descend_style)
        style_seqs = sess.run(model.style_seqs)
        tf.logging.info("Finishing beam_search on batch %d", i)

        # For each element of the batch write to file
        for b in range(model.config.batch_size):

            # Save each element of the batch
            single_global_step = global_step
            single_image = (images[b, :, :, :] - images[b, :, :, :].min())/2.0
            single_image_id = image_ids[b]
            single_target_seq = target_seqs[b, :]
            single_final_seq = final_seqs[b, 0, :]
            single_style_seq = style_seqs[b, 0, :]
            
            comparison_dump.append({"ground_truth": ids_to_sentence(single_target_seq, vocab),
                                    "original": ids_to_sentence(single_final_seq, vocab),
                                    "styled": ids_to_sentence(single_style_seq, vocab)})
            
            if single_image_id not in unique_image_ids:
                # Caption to dump and update image ids
                json_dump.append({"image_id": int(np.sum(single_image_id)), 
                                  "caption": ids_to_sentence(single_style_seq, vocab)})
                
                tf.logging.info("Ground Truth %d of %d: %s", 
                                i * model.config.batch_size + b,
                                model.config.batch_size * num_eval_batches,
                                ids_to_sentence(single_target_seq, vocab))
                tf.logging.info("Original %d of %d: %s", 
                                i * model.config.batch_size + b,
                                model.config.batch_size * num_eval_batches,
                                ids_to_sentence(single_final_seq, vocab))
                tf.logging.info("Styled %d of %d: %s", 
                                i * model.config.batch_size + b,
                                model.config.batch_size * num_eval_batches,
                                ids_to_sentence(single_style_seq, vocab))
                unique_image_ids.add(single_image_id)
                
    # Output a comparison file between generated and ground truth.
    with open(
            os.path.join(FLAGS.eval_dir, "style.comparison." + str(time_now) + ".json"), 
            "w") as f:
        json.dump(comparison_dump, f)
    
    # Evaluate the performance
    metrics = coco_get_metrics(time_now, global_step, json_dump)  
    eval_time = time.time() - start_time

    # Log perplexity to the FileWriter.
    summary = tf.Summary()
    for name, val in metrics.items():
        value = summary.value.add()
        value.simple_value = val
        value.tag = name
    summary_writer.add_summary(summary, global_step)

    # Write the Events file to the eval directory.
    summary_writer.flush()
    tf.logging.info("Finished processing evaluation at global step %d.",
                    global_step)


def run_once(model, saver, summary_writer, summary_op):
    """Evaluates the latest model checkpoint.

    Args:
        model: Instance of ShowAndTellModel; the model to evaluate.
        saver: Instance of tf.train.Saver for restoring model Variables.
        summary_writer: Instance of FileWriter.
        summary_op: Op for generating model summaries.
    """
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if not model_path:
        tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                        FLAGS.checkpoint_dir)
        return

    with tf.Session() as sess:
        # Load model from checkpoint.
        tf.logging.info("Loading model from checkpoint: %s", model_path)
        saver.restore(sess, model_path)
        global_step = tf.train.global_step(sess, model.global_step.name)
        tf.logging.info("Successfully loaded %s at global step = %d.",
                        os.path.basename(model_path), global_step)
        if global_step < FLAGS.min_global_step:
            tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                          FLAGS.min_global_step)
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Run evaluation on the latest checkpoint.
        try:
            evaluate_model(
                sess=sess,
                model=model,
                global_step=global_step,
                summary_writer=summary_writer,
                summary_op=summary_op)
        except Exception as e:  # pylint: disable=broad-except
            tf.logging.error("Evaluation failed.")
            coord.request_stop(e)

        try:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
        except Exception as e:  # pylint: disable=broad-except
            tf.logging.error("Failed to shutdown correctly.")


def run():
    """Runs evaluation in a loop, and logs summaries to TensorBoard."""
    # Create the evaluation directory if it doesn't exist.
    eval_dir = FLAGS.eval_dir
    if not tf.gfile.IsDirectory(eval_dir):
        tf.logging.info("Creating eval directory: %s", eval_dir)
        tf.gfile.MakeDirs(eval_dir)

    g = tf.Graph()
    with g.as_default():
        # Build the model for evaluation.
        model_config = configuration.ModelConfig()
        model_config.input_file_pattern = FLAGS.input_file_pattern
        model_config.wikipedia_file_pattern = FLAGS.wikipedia_file_pattern
        model = show_and_tell_model.ShowAndTellModel(model_config, mode="eval")
        model.use_style = True
        model.build()

        # Create the Saver to restore model Variables.
        saver = tf.train.Saver(var_list=model.model_variables)

        # Create the summary operation and the summary writer.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(eval_dir)

        g.finalize()

        # Run a new evaluation run every eval_interval_secs.
        while True:
            start = time.time()
            tf.logging.info("Starting evaluation at " + time.strftime(
                "%Y-%m-%d-%H:%M:%S", time.localtime()))
            run_once(model, saver, summary_writer, summary_op)
            time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)


def main(unused_argv):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.wikipedia_file_pattern, "--wikipedia_file_pattern is required"
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
    assert FLAGS.eval_dir, "--eval_dir is required"
    assert FLAGS.annotations_file, "--annotations_file is required"
    run()


if __name__ == "__main__":
    tf.app.run()

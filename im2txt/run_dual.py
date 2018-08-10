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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import json

import numpy as np
import tensorflow as tf

from im2txt import configuration
from im2txt import show_and_tell_model


from attrigram import run_attributes


import glove

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("attrigram_checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.flags.DEFINE_integer("divisions", 1000,
                       "Number of samles of the heuristic to take ")

tf.logging.set_verbosity(tf.logging.INFO)


def run_caption(checkpoint_path, attrigram_checkpoint_path, filenames, divisions):
    g = tf.Graph()
    with g.as_default():
        # Build the model for evaluation.
        model_config = configuration.ModelConfig()
        model = show_and_tell_model.ShowAndTellModel(model_config, mode="inference")
        model.use_style = True
        model.build()
        # Create the Saver to restore model Variables.
        saver = tf.train.Saver(var_list=model.model_variables)
        g.finalize()

    def _restore_fn(sess):
        tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
        saver.restore(sess, checkpoint_path)
        tf.logging.info("Successfully loaded checkpoint: %s",
                        os.path.basename(checkpoint_path))
        
    restore_fn = _restore_fn

    # Create the vocabulary.
    vocab = glove.load(model_config.config)[0]

    run_results = []
    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)
        
        # Run the attribute probabilities
        ps = run_attributes.run_attributes(
            attrigram_checkpoint_path, 
            filenames,
            divisions)

        # Prepare the list of image bytes for evaluation.
        images = []
        for filename in filenames:
            with tf.gfile.GFile(filename, "rb") as f:
                images.append(f.read())
        
        captions = []
        for img, p in zip(images, ps):
            input_feed = {"image_feed:0": img,
                model.attribute_probabilities: [p["probabilities"]]}
            sess.run(model.assign_initial_states, 
                feed_dict=input_feed)
            
            # The original caption
            img_progress = [sess.run(model.style_seqs, 
                feed_dict=input_feed)]
            
            # Perform style transfer
            for x in range(10):
                sess.run(model.descend_style, 
                    feed_dict=input_feed)
                caption = sess.run(model.style_seqs, 
                    feed_dict=input_feed)
                img_progress.append(caption)
            captions.append(img_progress)

        for i, caption in enumerate(captions):
            run_results.append({"filename": filenames[i], "captions": []})
            for flow in caption:
                for j in range(flow.shape[0]):
                    # Ignore begin and end words.
                    c = flow[j, 0, :].tolist()
                    sentence = [vocab.id_to_word(w) for w in c[:-1]]
                    sentence = " ".join(sentence)
                    run_results[i]["captions"].append(sentence)
                
    return run_results

def main(_):
        
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        if not checkpoint_path:
            raise ValueError("No checkpoint file found in: %s" % FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
        
    if tf.gfile.IsDirectory(FLAGS.attrigram_checkpoint_path):
        attrigram_checkpoint_path = tf.train.latest_checkpoint(FLAGS.attrigram_checkpoint_path)
        if not checkpoint_path:
            raise ValueError("No checkpoint file found in: %s" % FLAGS.attrigram_checkpoint_path)
    else:
        attrigram_checkpoint_path = FLAGS.attrigram_checkpoint_path

    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)
                
    zero_caption = run_caption(checkpoint_path, attrigram_checkpoint_path, filenames, FLAGS.divisions)
    for single_image in zero_caption:
        print("Captions for image %s:" % os.path.basename(single_image["filename"]))
        for j, caption in enumerate(single_image["captions"]):
            print("  %d) %s " % (j, caption))
            
    with open("captions_styled.json", "w") as f:
        json.dump(zero_caption, f)

if __name__ == "__main__":
      tf.app.run()

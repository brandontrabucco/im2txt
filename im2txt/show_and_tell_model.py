# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os.path
import glove.utils

from im2txt.ops import image_embedding
from im2txt.ops import image_processing
from im2txt.ops import inputs as input_ops
from im2txt.ops import inputs_wikipedia as wikipedia_ops


class ShowAndTellModel(object):
    """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

    "Show and Tell: A Neural Image Caption Generator"
    Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
    """

    def __init__(self, config, mode, train_inception=False):
        """Basic setup.

        Args:
            config: Object containing configuration parameters.
            mode: "train", "eval" or "inference".
            train_inception: Whether the inception submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        # Reader for the input data.
        self.reader = tf.TFRecordReader()
        self.reader_w = tf.TFRecordReader()

        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
                minval=-self.config.initializer_scale,
                maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0):
        """Decodes and processes an image string.

        Args:
            encoded_image: A scalar string Tensor; the encoded image.
            thread_id: Preprocessing thread id used to select the ordering of color
                distortions.

        Returns:
            A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              is_training=self.is_training(),
                                              height=self.config.image_height,
                                              width=self.config.image_width,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)

    def build_wikipedia_inputs(self):
        """Input prefetching preprocessing and batching for wikipedia.
        Training and eval only.

        Outputs:
            self.wikipedia_article_ids
            self.wikipedia_sentence_ids
            self.wikipedia_target_seqs
            self.wikipedia_input_seqs
            self.wikipedia_mask
        """
        if self.mode == "inference":
            # Build wikipedia inputs
            wikipedia_input_seqs = None
            wikipedia_target_seqs = None
            wikipedia_mask = None
            wikipedia_article_ids = None
            wikipedia_sentence_ids = None

        else:
            # Prefetch serialized SequenceExample protos.
            wikipedia_queue = wikipedia_ops.prefetch_input_data(
                    self.reader_w,
                    self.config.wikipedia_file_pattern,
                    is_training=self.is_training(),
                    batch_size=self.config.batch_size,
                    values_per_shard=self.config.values_per_wikipedia_shard,
                    input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                    num_reader_threads=self.config.num_input_reader_threads)

            # Process sentences into a list from SequenceExamples
            assert self.config.num_preprocess_threads % 2 == 0
            sentence_features = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = wikipedia_queue.dequeue()
                w_buffer = wikipedia_ops.parse_sequence_example(
                        serialized_sequence_example,
                        article_id=self.config.article_id_name,
                        sentence_id=self.config.sentence_id_name,
                        title_feature=self.config.title_feature_name,
                        sentence_feature=self.config.sentence_feature_name)
                sentence_features.append(w_buffer)

            queue_capacity = (2 * self.config.num_preprocess_threads *
                                                self.config.batch_size)
            # Build wikipedia inputs
            (wikipedia_article_ids, 
                wikipedia_sentence_ids, 
                wikipedia_input_seqs,
                wikipedia_target_seqs, 
                wikipedia_mask) = (
                        wikipedia_ops.batch_with_dynamic_pad(sentence_features,
                                                             batch_size=self.config.batch_size,
                                                             queue_capacity=queue_capacity))

        self.wikipedia_article_ids = wikipedia_article_ids
        self.wikipedia_sentence_ids = wikipedia_sentence_ids
        self.wikipedia_target_seqs = wikipedia_target_seqs
        self.wikipedia_input_seqs = wikipedia_input_seqs
        self.wikipedia_mask = wikipedia_mask

    def build_mscoco_inputs(self):
        """Input prefetching, preprocessing and batching.

        Outputs:
            self.images
            self.input_seqs
            self.target_seqs (training and eval only)
            self.input_mask (training and eval only)
        """
        if self.mode == "inference":
            # In inference mode, images and inputs are fed via placeholders.
            image_feed = tf.placeholder(dtype=tf.string, 
                                        shape=[], 
                                        name="image_feed")

            # Process images.
            images = self.process_image(image_feed)
            images = tf.expand_dims(images, axis=0)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_seqs = None
            image_ids = None
            input_mask = None
            encoded_image = None
            caption = None

        else:
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                    self.reader,
                    self.config.input_file_pattern,
                    is_training=self.is_training(),
                    batch_size=self.config.batch_size,
                    values_per_shard=self.config.values_per_input_shard,
                    input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                    num_reader_threads=self.config.num_input_reader_threads)

            # Image processing and random distortion. Split across multiple threads
            # with each thread applying a slightly different distortion.
            assert self.config.num_preprocess_threads % 2 == 0
            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                encoded_image, encoded_id, caption = input_ops.parse_sequence_example(
                        serialized_sequence_example,
                        image_feature=self.config.image_feature_name,
                        image_id=self.config.image_id_name,
                        caption_feature=self.config.caption_feature_name)
                image = self.process_image(encoded_image, thread_id=thread_id)
                images_and_captions.append([image, encoded_id, caption])

            # Batch inputs.
            queue_capacity = (2 * self.config.num_preprocess_threads *
                                                self.config.batch_size)
            images, image_ids, input_seqs, target_seqs, input_mask = (
                    input_ops.batch_with_dynamic_pad(images_and_captions,
                                                     batch_size=self.config.batch_size,
                                                     queue_capacity=queue_capacity))

        self.images = images
        self.image_ids = image_ids
        self.encoded_images = encoded_image
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings.

        Inputs:
            self.images

        Outputs:
            self.image_embeddings
        """
        inception_output = image_embedding.inception_v3(
                self.images,
                trainable=self.train_inception,
                is_training=self.is_training())
        self.inception_variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

        # Compute the average pool of the outputs from inception
        context_tensor = tf.reduce_mean(inception_output, axis=[1, 2])

        # Map inception output into embedding space.
        with tf.variable_scope("image_embedding"):

            image_embedding_map = tf.get_variable(
                    name="image_map",
                    shape=[context_tensor.shape[1], self.config.embedding_size],
                    initializer=self.initializer)
            image_embeddings = tf.tensordot(context_tensor, image_embedding_map, 1)

        # Save the embedding size in the graph.
        tf.constant(self.config.embedding_size, name="embedding_size")

        self.inception_output = inception_output
        self.image_embedding_map = image_embedding_map
        self.image_embeddings = image_embeddings

    def build_seq_embeddings(self):
        """Builds the input sequence embeddings.

        Inputs:
            self.input_seqs

        Outputs:
            self.seq_embeddings
        """
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):

            embedding_map = tf.get_variable(
                    name="map",
                    initializer=tf.constant(glove.load(self.config.config)[1], dtype=tf.float32),
                    trainable=self.config.train_embeddings)

            if self.mode == "inference":
                seq_embeddings = None
                wikipedia_embeddings = None
            else:
                seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
                wikipedia_embeddings = tf.nn.embedding_lookup(embedding_map, self.wikipedia_input_seqs)

        self.embedding_map = embedding_map
        self.seq_embeddings = seq_embeddings
        self.wikipedia_embeddings = wikipedia_embeddings

    def build_lstm(self):
        """Builds the model.

        Inputs:
            self.image_embeddings
            self.seq_embeddings
            self.target_seqs (training and eval only)
            self.input_mask (training and eval only)

        Outputs:
            self.total_loss (training and eval only)
            self.target_cross_entropy_losses (training and eval only)
            self.target_cross_entropy_loss_weights (training and eval only)
        """
        # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
        # modified LSTM in the "Show and Tell" paper has no biases and outputs
        # new_c * sigmoid(o).
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.config.num_lstm_units, state_is_tuple=True)
        if self.mode == "train":
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell,
                    input_keep_prob=self.config.lstm_dropout_keep_prob,
                    output_keep_prob=self.config.lstm_dropout_keep_prob)
            
        # Feed the image embeddings to set the initial LSTM state.
        zero_state = lstm_cell.zero_state(
                batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
        _, initial_state = lstm_cell(self.image_embeddings, zero_state)
            
        probabilities_layer = tf.layers.Dense(
            units=self.config.vocab_size,
            kernel_initializer=tf.constant_initializer(
                glove.load(self.config.config)[1].T, dtype=tf.float32))
        
        # Used for adapting thge probabilities of beam search.
        class Critic(tf.layers.Layer):
            
            def __init__(self, probs_layer, heuristic, ratio):
                self.probs_layer = probs_layer
                self.heuristic = heuristic
                self.ratio = ratio
                
            def __call__(self, inputs):
                return ((self.probs_layer(inputs) * self.ratio) 
                        + (self.heuristic * (1 - self.ratio)))
            
            def compute_output_shape(self, input_shape):
                return input_shape

        vocab = glove.load(self.config.config)[0]

        # The beam search decoder used in the graph.
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            lstm_cell, self.embedding_map, 
            tf.fill([self.image_embeddings.get_shape()[0]], vocab.start_id), vocab.end_id,
            tf.contrib.seq2seq.tile_batch(
                initial_state, multiplier=self.config.beam_size), self.config.beam_size,
            output_layer=Critic(probabilities_layer, self.generality_table, 1.0))
            
        (final_outputs, 
            final_state,
            final_sequence_lengths) = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.config.maximum_iterations)
        
        # Transpose outputs into [batch_size, beam_size, sequence_length]
        self.final_seqs = tf.transpose(final_outputs.predicted_ids, [0, 2, 1])
        
        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()
            
            if self.mode == "inference":
                self.mscoco_logits = None
                self.mscoco_outputs = None
                self.wikipedia_logits = None
                self.wikipedia_outputs = None

            else:
                # Run the batch of sequence embeddings through the LSTM.
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                mscoco_hidden, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                     inputs=self.seq_embeddings,
                                                     sequence_length=sequence_length,
                                                     initial_state=initial_state,
                                                     dtype=tf.float32,
                                                     scope=lstm_scope)
                # Stack batches vertically.
                mscoco_hidden = tf.reshape(mscoco_hidden, [-1, lstm_cell.output_size])

                # Run the LSTM on wikipedia.
                wikipedia_length = tf.reduce_sum(self.wikipedia_mask, 1)
                wikipedia_hidden, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                        inputs=self.wikipedia_embeddings,
                                                        sequence_length=wikipedia_length,
                                                        initial_state=zero_state,
                                                        dtype=tf.float32,
                                                        scope=lstm_scope)
                # Stack batches vertically.
                wikipedia_hidden = tf.reshape(wikipedia_hidden, [-1, lstm_cell.output_size])

                # Compute the probabilities.
                with tf.variable_scope("logits"):   

                    # Calculate mscoco logits.
                    self.mscoco_logits = probabilities_layer(mscoco_hidden)
                    self.mscoco_outputs = tf.nn.softmax(self.mscoco_logits, name="mscoco_softmax")

                    self.wikipedia_logits = probabilities_layer(wikipedia_hidden)
                    self.wikipedia_outputs = tf.nn.softmax(self.wikipedia_logits, name="wikipedia_softmax")

    def build_heuristic(self):
        """Builds heuristic calculation from vocabulary.
        
        Outputs:
            self.generality_table
        """
        # Load the generality heiristic table
        if not os.path.isfile(self.config.generality_heuristic_file):
            np.savetxt(self.config.generality_heuristic_file, np.zeros([self.config.vocab_size]))
        generality_table = np.loadtxt(self.config.generality_heuristic_file)
        generality_table = generality_table[np.newaxis, :]

        # Center between [0, 1]
        generality_table = tf.constant(generality_table, dtype=tf.float32)
        generality_table = tf.nn.softmax(generality_table)
            
        self.generality_table = generality_table

    def build_losses(self):
        """Builds the losses on which to optimize the model.
        Inputs:
            self.target_seqs
            self.wikipedia_target_seqs
            self.input_mask
            self.wikipedia_mask
            self.unscaled_logits
            self.softmax_outputs
            self.wikipedia_logits
            self.wikipedia_outputs

        Outputs:
            self.total_loss
            self.target_cross_entropy_losses
            self.target_cross_entropy_loss_weights
        """
        if self.mode != "inference":
            mscoco_targets = tf.reshape(self.target_seqs, [-1])
            wikipedia_targets = tf.reshape(self.wikipedia_target_seqs, [-1])
            mscoco_weights = tf.to_float(tf.reshape(self.input_mask, [-1]))
            wikipedia_weights = tf.to_float(tf.reshape(self.wikipedia_mask, [-1]))

            # Compute losses.
            mscoco_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mscoco_targets,
                                                                                                                            logits=self.mscoco_logits)
            mscoco_loss = tf.div(tf.reduce_sum(tf.multiply(mscoco_losses, mscoco_weights)),
                                                    tf.reduce_sum(mscoco_weights),
                                                    name="batch_loss")
            tf.losses.add_loss(mscoco_loss)

            # Compute loss for wikipedia
            wikipedia_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=wikipedia_targets,
                                                                                                                                                logits=self.wikipedia_logits)
            wikipedia_loss = tf.div(tf.reduce_sum(tf.multiply(wikipedia_losses, wikipedia_weights)),
                                                            tf.reduce_sum(wikipedia_weights),
                                                            name="wikipedia_loss") * self.config.weight_wikipedia
            tf.losses.add_loss(wikipedia_loss)

            # Calculate heuristic reward for sampled words on mscoco
            mscoco_heuristic = -self.config.weight_generality_heuristic * tf.div(
                tf.reduce_sum(tf.reduce_sum(
                    self.mscoco_outputs * self.generality_table,
                    axis=1) * mscoco_weights), tf.reduce_sum(mscoco_weights), name="mscoco_heuristic")
            tf.losses.add_loss(mscoco_heuristic)

            # Calculate heuristic reward for sampled words on wikipedia
            wikipedia_heuristic = -self.config.weight_wikipedia * self.config.weight_generality_heuristic * tf.div(
                tf.reduce_sum(tf.reduce_sum(
                    self.wikipedia_outputs * self.generality_table,
                    axis=1) * wikipedia_weights), tf.reduce_sum(wikipedia_weights), name="wikipedia_heuristic")
            tf.losses.add_loss(wikipedia_heuristic)

            total_loss = tf.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/mscoco_loss", mscoco_loss)
            tf.summary.scalar("losses/wikipedia_loss", wikipedia_loss)
            tf.summary.scalar("losses/mscoco_heuristic", mscoco_heuristic)
            tf.summary.scalar("losses/wikipedia_heuristic", wikipedia_heuristic)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = mscoco_losses    # Used in evaluation.
            self.target_cross_entropy_loss_weights = mscoco_weights    # Used in evaluation.

    def setup_inception_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            # Restore inception variables only.
            saver = tf.train.Saver(self.inception_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring Inception variables from checkpoint file %s",
                                                self.config.inception_checkpoint_file)
                saver.restore(sess, self.config.inception_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_mscoco_inputs()
        self.build_wikipedia_inputs()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_heuristic()
        self.build_lstm()
        self.build_losses()
        self.setup_inception_initializer()
        self.setup_global_step()


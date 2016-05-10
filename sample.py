import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq
from tensorflow.python.platform import gfile
import numpy as np
import sys
import os
import nltk
from six.moves import xrange
import models.chatbot
import util.hyperparamutils as hyper_params
import util.vocabutils as vocab_utils

_buckets = []
convo_lim = 1
max_src_len = 0
max_tr_len = 0

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')
flags.DEFINE_string('data_dir', "data/", "Data storage directory")

def loadModel(session, path):
	global _buckets
	global max_src_len
	global max_tr_len
	global convo_lim

	prms = hyper_params.restoreHyperParams(path)
	bucks = []
	num_buck = prms["num_buckets"]
	max_src_len = prms["max_source_length"]
	max_tr_len = prms["max_target_length"]
	convo_lim = prms["conversation_history"]

	for i in range(num_buck):
		str1 = "bucket_%d_source" % (i)
		str2 = "bucket_%d_target" % (i)

		bucks.append((prms[str1],prms[str2]))
	_buckets = bucks
	model = models.chatbot.ChatbotModel(prms["vocab_size"], _buckets,
		prms["hidden_size"], 1.0, prms["num_layers"], prms["grad_clip"],
		1, prms["learning_rate"], prms["decay_factor"], 512, True)

	checkpt = tf.train.get_checkpoint_state(path)

	if checkpt and gfile.Exists(checkpt.model_checkpoint_path):
		form_str = "Reading model params from %s" % (checkpt.model_checkpoint_path)
		model.saver.restore(session, checkpt.model_checkpoint_path)
	else:
		print "Possible wrong checkpoint directory."
		model = None

	return model

def main():
	with tf.Session() as sess:

		model = loadModel(sess, FLAGS.checkpoint_dir)
		print "Buckets: ",
		print _buckets
		model.batch_size = 1
		vocab = vocab_utils.VocabMapper(FLAGS.data_dir)
		sys.stdout.write(">")
		sys.stdout.flush()
		sent = sys.stdin.readline()
		conv_hist = [sent]

		while sent:
			tok_ids = list(reversed(vocab.tokens2Indices(" ".join(conv_hist))))

			newls = []
			for i in xrange(len(_buckets)):
				if _buckets[i][0] > len(tok_ids):
					newls.append(i)
			buck_id = min(newls)

			enc_in, dec_in, targ_weights = model.get_batch(
				{buck_id: [(tok_ids, [])]}, buck_id)

			_, _, output_logits = model.step(sess, enc_in, dec_in, targ_weights,
				buck_id, True)

			outs = [int(np.argmax(log, axis=1)) for log in output_logits]

			end_id = vocab_utils.EOS_ID

			if end_id in outs:
				outs = outs[:outs.index(end_id)]

			conv = " ".join(vocab.indices2Tokens(outs))

			conv_hist.append(conv)
			print conv
			sys.stdout.write(">")
			sys.stdout.flush()
			sent = sys.stdin.readline()
			conv_hist.append(sent)
			conv_hist = conv_hist[-convo_lim:]


if __name__ == "__main__":
	main()

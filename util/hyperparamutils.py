import os
import pickle

def restoreHyperParams(fp):
	fp = os.path.join(fp, "hyperparams.p")
	with open(fp, 'rb') as newf:
		return pickle.load(newf)


def saveHyperParameters(check, FLAGS, buck, conv_lim):
	params_dict = {}

	params_dict["vocab_size"] = FLAGS.vocab_size
	params_dict["hidden_size"] = FLAGS.hidden_size
	params_dict["dropout"] = FLAGS.dropout
	params_dict["grad_clip"] = FLAGS.grad_clip
	params_dict["num_layers"] = FLAGS.num_layers
	params_dict["learning_rate"] = FLAGS.learning_rate
	params_dict["lr_decay_factor"] = FLAGS.lr_decay_factor
	params_dict["num_buckets"] = len(buck)
	params_dict["max_source_length"] = conv_lim[0]
	params_dict["max_target_length"] = conv_lim[1]
	params_dict["conversation_history"] = conv_lim[2]

	for i in range(len(buck)):
		str1 = "bucket_%d_source" % (i)
		str2 = "bucket_%d_target" % (i)

		params_dict[str1] = buck[i][0]
		params_dict[str2] = buck[i][1]

	check_path = os.path.join(check, "hyperparams.p")
	with open(check_path, 'wb') as file:
		pickle.dumb(params_dict, file)
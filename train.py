
import math, os, random, sys, time
import numpy as numpy

# from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import gfile
import util.hyperparamutils as hp 
from util.vocabutils import VocabMapper, EOS_ID
from util.dataprocessor import DataProcessor
from models.chatbot import ChatbotModel

## CONSTANTS
flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.5, "learning rate")
flags.DEFINE_float("lr_decay_factor", 0.99, "learning rate decay rate")
flags.DEFINE_float("grad_clip", 5.0, "clip gradients to this norm")
flags.DEFINE_float("train_frac", 0.7, "% data to used for training")
flags.DEFINE_float("dropout", 0.5, "probability of hidden inputs being removed")
flags.DEFINE_integer("batch_size", 5, "batch size to use during training.")
flags.DEFINE_integer("max_epoch", 6, "max number of iterations of training set")
flags.DEFINE_integer("hidden_size", 100, "size of each model layer")
flags.DEFINE_integer("num_layers", 1, "number of layers in the model")
flags.DEFINE_integer("vocab_size", 400000, "Max vocabulary size.")
flags.DEFINE_string("data_dir", "data/", "directory containing processed data.")
flags.DEFINE_string("raw_data_dir", "data/raw_data/", "raw text data directory")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "checkpoint dir")
flags.DEFINE_integer("steps_per_checkpoint", 200, "training steps per checkpoint.")
FLAGS = flags.FLAGS

max_num_lines = 1
max_tgt_len = 200
max_src_len = 200

buckets = [(40,10),(50,15),(100,25), (200,200)]
ckpt_path = os.path.join(FLAGS.checkpoint_dir, 'ckpt')
max_loss = 300

def getModel(session, path, vocab_size):
   model = models.chatbot.ChatbotModel(FLAGS.vocab_size, buckets, 
      FLAGS.hidden_sizeFLAGS.dropout, FLAGS.num_layers, FLAGS.grad_clip, 
      LAGS.batch_size, FLAGS.learning_rate, FLAGS.decay_factor)

   conv_lims = [max_src_len, max_tgt_len, max_num_lines]
   hp.saveHyperParameters(path, FLAGS, buckets, conv_lims)

   ckpt = tf.train.get_checkpoint_state(path)
   if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
      print "Restoring Model Parameters from {0}".format(ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
   else:
      print "Creating new model"
      session.run(tf.initialize_all_variables)
   return model

def readData(src_path, tgt_path):
   dataset = [[] for _ in buckets]
   with gfile.GFile(src_path, mode='r') as src_file:
      with gfile.GFile(tgt_path, mode='r') as tgt_file:
         src, tgt = src_file.readline(), tgt_file.readline()
         cnt = 0

         while src and tgt:
            cnt += 1
            if cnt % 100000 == 0:
               print "reading data line %d" % cnt
               sys.stdout.flush()
            src_ids = [int(x) for x in src.split()]
            tgt_ids = [int(x) for x in tgt.split()]
            tgt_ids.append(vocab_utils.EOS_ID)

            for bucket_id, (src_size, tgt_size) in enumerate(buckets):
               if len(src_ids) < src_size and len(tgt_ids) < tgt_size:
                  dataset[bucket_id].append([src_ids, tgt_ids])
                  break
            src, tgt = src_file.readline(), tgt_file.readline()
   return dataset

def main():

   # create checkpoint directory if needed
   if not os.path.exists(FLAGS.checkpoint_dir):
      os.mkdir(FLAGS.checkpoint_dir)

   # run data processor to build training and testing sets
   data_processor = DataProcessor(FLAGS.vocab_size, FLAGS.raw_data_dir, 
      FLAGS.data_dir, FLAGS.train_frac, max_num_lines, max_tgt_len, 
      max_src_len).run()

   print "Data Processed"

   # build vocabulary
   vocab_mapper = vocab.utils.VocabMapper(FLAGS.data_dir)
   vocab_size = vocab_mapper.getVocabSize()

   print "Vocab Processed: {0}".format(vocab_size)

   # start tensorflow
   with tf.Session() as sess:
      
      # logger
      writer = tf.train.SummaryWRite("logs/", sess.graph)
      
      # load model from file, or create new model
      model = createModel(sess, FLAGS.checkpoint_dir, FLAGS.vocab_size)

      print "Using bucket sizes:"
      print buckets

      # set various file paths
      src_train_fp = data_processor.data_source_train
      tgt_train_fp = data_processor.data_target_train
      src_test_fp = data_processor.data_source_test
      tgt_test_fp = data_processor.data_target_test

      # read in training and testing data
      train_set = readData(src_train_fp, tgt_train_fp)
      test_set = readData(src_test_fp, tgt_test_fp)

      # set up buckets
      train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
      train_total_size = float(sum(train_bucket_sizes))
      train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
         for i in range(len(train_bucket_sizes))]

      print "Beginning training"

      # begin training
      step_time, loss = 0.0,0.0
      current_step = 0
      prev_loss = []

      while True:

         # collect random number 
         rnd = np.random.random_sample()
         bucket_ids = min([i for i in range(len(train_buckets_scale))
            if train_buckets_scale[i] > rnd])

         # get results and update loss
         start_time = time.time()
         enc_inputs, dec_inputs, tgt_weights = model.get_batch(train_set, bucket_ids)
         _, step_loss, _ = model.step(sess, enc_inputs, dec_inputs, 
            tgt_weights, bucket_id, False)
         loss += step_loss / FLAGS.steps_per_checkpoint
         current_step += 1

         # print out progress update
         if current_step % FLAGS.steps_per_checkpoint == 0:
            train_liss_summary = tf.summary()
            str_summary_train_loss = train_loss_summary.value.add()
            str_summary_train_loss.simple_value = loss
            str_summary_train_loss.tag = "train_loss"
            writer.add_summary(train_loss_summary, current_step)

            # calculate perplexity, and print results
            perp = math.exp(loss) if loss < max_loss else float('inf')
            print "Checkpoint: %d, Learning Rate: %0.4f,  \t Step-Time: %.2f \t \
               Perplexity: %0.2f" % (model.global_step.eval(), 
               model.learning_rate.eval(), step_time, perp)

            # decrease learning rate if no improvement past 3 iter
            if len(prev_loss) > 2 and loss > max(prev_loss[-3:]):
               sess.run(model.learning_rate_decary_op)

            prev_loss.append(loss)

            model.saver.save(sess, ckpt_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0

            perplexity_summary = tf.Summary()
            eval_loss_summary = tf.Summary()
            for bucket_id in range(len(buckets)):
               if len(test_set[bucket_id]) == 0:
                  print "\tempty bucket %d" % bucket_id
                  continue
               enc_inputs, dec_inputs, tgt_weights = model.get_batch(train_set, bucket_id)
               _, eval_loss, _ = model.step(sess, enc_inputs, dec_inputs, 
                  tgt_weights, bucket_id, True)
               eval_ppx = math.exp(eval_loss) if eval_loss < max_loss else float('inf')
               print("\teval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

               str_summary_ppx = perplexity_summary.value.add()
               str_summary_ppx.simple_value = eval_ppx
               str_summary_ppx.tag = "peplexity_bucket)%d" % bucket_id

               str_summary_eval_loss = eval_loss_summary.value.add()
               str_summary_eval_loss.simple_value = float(eval_loss)
               str_summary_eval_loss.tag = "eval_loss_bucket)%d" % bucket_id
               writer.add_summary(perplexity_summary, current_step)
               writer.add_summary(eval_loss_summary, current_step)

            sys.stdout.flush()

if __name__ == '__main__':
   main()
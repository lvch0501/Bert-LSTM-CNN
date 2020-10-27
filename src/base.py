import tensorflow as tf
import os
class Model(object):


    def __init__(self):
        self.vocab = None
        self.data = None
        self.batch_size = None


    def save(self, checkpoint_dir, dataset_name):
        self.saver = tf.train.Saver()

        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__ or "Reader"
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))


    def load(self, checkpoint_dir, dataset_name):
        self.saver = tf.train.Saver()
        print(" [*] Loading checkpoints...")
        model_name = dataset_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
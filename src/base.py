import tensorflow as tf

class Model(object):


    def __init__(self):
        self.vocab = None
        self.data = None
        self.batch_size = None


    def save(self, checkpoint_dir, dataset_name):
        self.saver = tf.train.Saver()

        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__ or "Reader"

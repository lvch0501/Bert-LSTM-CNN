from utils import load_sentences, word_mapping
import re
import collections
import tensorflow as tf
from tensorflow import pywrap_tensorflow

import tensorflow as tf
hello=tf.constant('hello,world')
sess=tf.Session()
print(sess.run(hello))
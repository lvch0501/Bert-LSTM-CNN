import os
import gensim
import itertools
import re
import numpy as np
from collections import OrderedDict
from read import load_parameter
from utils import load_sentences, update_tag_scheme, word_mapping, augment_with_pretrained
from utils import char_mapping, tag_mapping, pt_mapping, save_mappings, reload_mappings
from utils import prepare_dataset, create_input, evaluate
from GRAMCNN import GRAMCNN
models_path = "../models"

config_para = load_parameter()

parameters = OrderedDict()
#IOB OR IOEB
parameters['padding'] = config_para["padding"] == '1'
parameters['tag_scheme'] = config_para["tag_scheme"]
parameters['lower'] = config_para["lower"] == '1'
parameters['zeros'] = config_para["zeros"] == '1'
parameters['char_dim'] = int(config_para["char_dim"])
parameters['char_lstm_dim'] = int(config_para["char_lstm_dim"])
parameters['char_bidirect'] = config_para["char_bidirect"] == '1'
parameters['word_dim'] = int(config_para["word_dim"])
parameters['word_lstm_dim'] = int(config_para["word_lstm_dim"])
parameters['word_bidirect'] = config_para["word_bidirect"] == '1'
parameters['pre_emb'] = config_para["pre_emb"]
parameters['all_emb'] = config_para["all_emb"] == '1'
parameters['cap_dim'] = int(config_para["cap_dim"])
parameters['crf'] = config_para["crf"] == '1'
parameters['dropout'] = float(config_para["dropout"])
parameters['lr_method'] = config_para["lr_method"]
parameters['use_word'] = config_para["use_word"] == '1'
parameters['use_char'] = config_para["use_char"] == '1'
parameters['hidden_layer'] = int(config_para["hidden_layer"])
parameters['reload'] = config_para["reload"] == '1'
#parameters['kernels'] = [2,3,4,5] if type(config_para.kernel_size) == str else map(lambda x : int(x), opts.kernel_size)
parameters['kernels'] = [2,3,4,5] if config_para["kernels"] == "" else [int(i) for i in config_para["kernels"].split(",")]
#parameters['num_kernels'] = [100,100,100,100] if type(config_para.kernel_num) == str else map(lambda x : int(x), opts.kernel_num)
parameters['num_kernels'] = [100,100,100,100] if config_para["kernel_num"] == "" else [int(i) for i in config_para["kernel_num"].split(",")]
parameters['pts'] = config_para["pts"] == '1'
parameters['epochs'] = int(config_para["epochs"])
parameters['freq_eval'] = int(config_para["freq_eval"])
# model name
model_name = 'use_word' + str(parameters['use_word']) + \
            ' use_char' + str(parameters['use_char']) + \
            ' drop_out' + str(parameters['dropout']) + \
            ' hidden_size' + str(parameters['word_lstm_dim']) + \
            ' hidden_layer' + str(parameters['hidden_layer']) + \
            ' lower' + str(parameters['lower']) + \
            ' allemb' + str(parameters['all_emb']) + \
            ' kernels' + str(parameters['kernels'])[1:-1] + \
            ' num_kernels' + str(parameters['num_kernels'])[1:-1] + \
            ' padding' + str(parameters['padding']) + \
            ' pts' + str(parameters['pts']) + \
            ' w_emb' + str(parameters['word_dim'])



train_sentences = load_sentences(config_para["train"], parameters['lower'], parameters['zeros'])
dev_sentences = load_sentences(config_para["dev"], parameters['lower'], parameters['zeros'])

avg_len = sum([len(i) for i in train_sentences])/float(len(train_sentences))
print("train average length: %d" % (avg_len))

if os.path.isfile(config_para["test"]):
    test_sentences = load_sentences(config_para["test"], parameters['lower'], parameters['zeros'])


update_tag_scheme(train_sentences, parameters['tag_scheme'])
update_tag_scheme(dev_sentences, parameters['tag_scheme'])
if os.path.isfile(config_para["test"]):
    update_tag_scheme(test_sentences, parameters['tag_scheme'])


dt_sentences = []
if os.path.isfile(config_para["test"]):
    dt_sentences = dev_sentences + test_sentences
else:
    dt_sentences = dev_sentences


if 'bin' in parameters['pre_emb']:
    wordmodel = gensim.models.KeyedVectors.load_word2vec_format(parameters['pre_emb'], binary=True)
else:
    wordmodel = gensim.models.KeyedVectors.load_word2vec_format(parameters['pre_emb'], binary=False)


# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
word_to_id = []
char_to_id = []
pt_to_id = []
tag_to_id = []
if not parameters['reload']:
    if parameters['pre_emb']:
        # mapping of words frenquency decreasing
        dico_words_train = word_mapping(train_sentences, parameters["lower"])[0]
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
            dico_words_train.copy(),
            wordmodel,
            list(itertools.chain.from_iterable(
                [[w[0] for w in s] for s in dt_sentences])
            ) if not parameters['all_emb'] else None
        )
    else:
        dico_words, word_to_id, id_to_word = word_mapping(train_sentences, parameters["lower"])
        dico_words_train = dico_words


    # Create a dictionary and a mapping for words / POS tags / tags
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
    dico_pts, pt_to_id, id_to_pt = pt_mapping(train_sentences + dev_sentences)
    if not os.path.exists(os.path.join(models_path, model_name)):
            os.makedirs(os.path.join(models_path,model_name))
    save_mappings(os.path.join(models_path, model_name, 'mappings.pkl'), word_to_id, char_to_id, tag_to_id, pt_to_id, dico_words, id_to_tag)
else:
    word_to_id, char_to_id, tag_to_id, pt_to_id, dico_words, id_to_tag = reload_mappings(os.path.join(models_path,model_name, 'mappings.pkl'))
    dico_words_train = dico_words
    id_to_word = {v: k for k, v in word_to_id.items()}


# Index sentences
m3 = 0
train_data,m1 = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, pt_to_id,parameters["lower"]
)
dev_data,m2 = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, pt_to_id,parameters["lower"]
)

if os.path.isfile(config_para["test"]):
    test_data,m3 = prepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id, pt_to_id, parameters["lower"]
    )
max_seq_len = max(m1,m2,m3)
print("max length is %i" % (max_seq_len))

print("%i / %i  sentences in train / dev." % (
   len(train_data), len(dev_data)))

#
# Train network
#
# 只出现一次的单词的下标
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])


n_epochs = parameters["epochs"]  # number of epochs over the training set
freq_eval = parameters["freq_eval"]  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf
count = 0

#initilaze the embedding matrix
word_emb_weight = np.zeros((len(dico_words), parameters['word_dim']))
c_found = 0
c_lower = 0
c_zeros = 0
n_words = len(dico_words)
for i in range(n_words):
        word = id_to_word[i]
        if word in wordmodel:
            word_emb_weight[i] = wordmodel[word]
            c_found += 1
        elif re.sub('\d', '0', word) in wordmodel:
            word_emb_weight[i] = wordmodel[
                re.sub('\d', '0', word)
            ]
            c_zeros += 1


print('Loaded %i pretrained embeddings.' % len(wordmodel.vocab))
print ('%i / %i (%.4f%%) words have been initialized with '
       'pretrained embeddings.' % (
            c_found + c_lower + c_zeros, n_words,
            100. * (c_found + c_lower + c_zeros) / n_words
      ))
print ('%i found directly, %i after lowercasing, '
       '%i after lowercasing + zero.' % (
          c_found, c_lower, c_zeros
      ))

gramcnn = GRAMCNN(n_words, len(char_to_id), len(pt_to_id),
                    use_word = parameters['use_word'],
                    use_char = parameters['use_char'],
                    use_pts = parameters['pts'],
                    num_classes = len(id_to_tag),
                    word_emb = parameters['word_dim'],
                    drop_out = parameters['dropout'],
                    word2vec = word_emb_weight,feature_maps=parameters['num_kernels'],#,200,200, 200,200],
                    kernels=parameters['kernels'], hidden_size = parameters['word_lstm_dim'], hidden_layers = parameters['hidden_layer'],
                    padding = parameters['padding'], max_seq_len = max_seq_len, train_size = len(train_data))

if parameters["reload"]:
    gramcnn.load(models_path, model_name)

for epoch in range(n_epochs):
    epoch_costs = []
    print("Starting epoch %i..." % epoch)

    for i, index in enumerate(np.random.permutation(len(train_data))):
        inputs, word_len = create_input(train_data[index], parameters, True, singletons,
                                        padding=parameters["padding"], max_seq_len=max_seq_len, use_pts=parameters['pts'])
        assert inputs['char_for']
        assert inputs['word']
        assert inputs['label']

        # break
        if len(inputs['label']) == 1:
            continue

        train_loss = []
        temp = []
        temp.append(word_len)
        batch_loss = gramcnn.train(inputs, temp)
        train_loss.append(batch_loss)

        if (i % 500 == 0 and i != 0):
            print("Epoch[%d], " % (epoch) + "Iter " + str(i) + \
                  ", Minibatch Loss= " + "{:.6f}".format(np.mean(train_loss[-500:])))
            train_loss = []


        if i % 2000 == 0 and i != 0:
            dev_score = evaluate(parameters, gramcnn, dev_sentences,
                                 dev_data, id_to_tag, padding = parameters['padding'],
                                 max_seq_len = max_seq_len, use_pts = parameters['pts'])
            print("dev_score_end")
            print("Score on dev: %.5f" % dev_score)
            if dev_score > best_dev:
                best_dev = dev_score
                print("New best score on dev.")
                print("Saving model to disk...")
                gramcnn.save(models_path ,model_name)
            if os.path.isfile(parameters["test"]):
                if i % 8000 == 0 and i != 0:
                    test_score = evaluate(parameters, gramcnn, test_sentences,
                                          test_data, id_to_tag, padding = parameters['padding'],
                                          max_seq_len = max_seq_len, use_pts = parameters['pts'])
                    print("Score on test: %.5f" % test_score)
                    if test_score > best_test:
                        best_test = test_score
                        print("New best score on test.")

        print()
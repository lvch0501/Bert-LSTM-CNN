import codecs
import re
import pickle
import numpy as np
import os

models_path = "../models"
eval_path = "../evaluation"
eval_temp = os.path.join(eval_path, "temp_result")
eval_script = os.path.join(eval_path, "conlleval")

def load_sentences(path, lower=True, zeros=True):

    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.strip()) if zeros else line.strip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


# replace all digit with 0
def zero_digits(s):
    return re.sub('\d', '0', s)


# ,
def update_tag_scheme(sentences, tag_scheme):
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    dico['<PAD>'] = 10000000 + 1
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def augment_with_pretrained(dictionary, word2vec, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from embeddings')
    #assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    # pretrained = set([
    #     line.rstrip().split()[0].strip()
    #     for line in codecs.open(ext_emb_path, 'r', 'utf-8')
    #     if len(ext_emb_path) > 0
    # ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in word2vec.vocab:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in word2vec.vocab for x in [
                word,
                #word.lower(),
                re.sub('\d', '0', word)
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word



def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<UNK>'] = 10000000
    dico['<PAD>'] = 9999999
    dico['{'] = 9999998
    dico['}'] = 9999997
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    #print dico
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    #print dico
    return dico, tag_to_id, id_to_tag


def pt_mapping(sentences):
    pts = [[word[-2] for word in s] for s in sentences]
    dico = create_dico(pts)
    dico[' '] = 100000000
    pt_to_id, id_to_pt = create_mapping(dico)
    print("Found %i unique pos tags" % len(dico))
    #print dico
    return dico, pt_to_id, id_to_pt


def save_mappings(mappings_path, word_to_id, char_to_id, tag_to_id, pt_to_id, dico_words, id_to_tag):
    """
    We need to save the mappings if we want to use the model later.
    """
    with open(mappings_path, 'wb') as f:
        mappings = {
            'word_to_id': word_to_id,
            'char_to_id': char_to_id,
            'tag_to_id': tag_to_id,
            'pt_to_id' : pt_to_id,
            'dico_words' : dico_words,
            'id_to_tag' : id_to_tag
        }
        pickle.dump(mappings, f)


def reload_mappings(mappings_path):
    """
    Load mappings from disk.
    """
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    id_to_word = mappings['word_to_id']
    id_to_char = mappings['char_to_id']
    id_to_tag = mappings['tag_to_id']
    id_to_pt = mappings['pt_to_id']
    dico_words = mappings['dico_words']
    tag_to_id = mappings['id_to_tag']
    return id_to_word, id_to_char, id_to_tag, id_to_pt, dico_words, tag_to_id


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, pt_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    #def f(x): return x.lower() if lower else x
    data = []
    maxlen = 0
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[w if w in word_to_id else '<UNK>']
                 for w in str_words]
        if (len(words) > maxlen):
            maxlen = len(words)
        # Skip characters that are not in the training set
        chars = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        pts = [pt_to_id[w[-2] if w[-2] in pt_to_id else ' '] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
            'pts'  : pts
        })
    return data, maxlen



def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 1
    elif s.upper() == s:
        return 2
    elif s[0].upper() == s[0]:
        return 3
    else:
        return 4


def create_input(data, parameters, add_label, singletons=None, padding = False, max_seq_len = 200, use_pts = False):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    input = {}
    words = data['words']
    word_len = len(words)
    chars = data['chars']
    tags = data['tags']
    char_for = []
    max_length = 0
    if not padding:
        if singletons is not None:
            words = insert_singletons(words, singletons)
        if parameters['cap_dim']:
            caps = data['caps']
        char_for, char_rev, char_pos, max_length = pad_word_chars(chars, singletons)
        pts = data['pts']
    else:
        if singletons is not None:
            words = insert_singletons(words, singletons)
        words = padding_word(words, max_seq_len)
        caps = padding_word(data['caps'], max_seq_len)
        pts = padding_word(data['pts'], max_seq_len)
        tags = padding_word(tags, max_seq_len)
        char_for, char_rev, char_pos, max_length = pad_word_chars(chars, singletons)
        char_for = padding_chars(char_for, max_seq_len, max_length)
    if parameters['word_dim']:
        input['word'] = words
    if parameters['char_dim']:
        input['char_for'] = char_for
    if parameters['cap_dim']:
        input['cap'] = caps
    if add_label:
        input['label'] = tags
    if use_pts:
        input['pts'] = pts
    return input, word_len



def insert_singletons(words, singletons, p=0.1):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words



def pad_word_chars(words, singletons):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    if singletons is not None:
        words = insert_unk(words)
    max_length = max([len(word) for word in words]) + 2
    if max_length < 7:
        max_length = 7
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        word = [2] + word + [3]
        padding = [1] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos, max_length


def insert_unk(words, p = 0.05):
    new_words = []
    flat = True
    for word in words:
        new_word = []
        for i in word:
            if np.random.uniform() < p and flat:
                new_word.append(0)
                flat = False
            else:
                new_word.append(i)
        new_words.append(new_word)
    return new_words


def padding_word(input, max_length):
    words_len = len(input)
    output = input + [0] * (max_length - words_len)
    return output


def padding_chars(chars, max_seq_len, max_char_len):
    word_len = len(chars)
    diff = max_seq_len - word_len
    output = chars
    for i in range(diff):
        output.append([1] * max_char_len)
    return output


def evaluate(parameters, sess, raw_sentences, parsed_sentences,
             id_to_tag, remove = True, padding = False, max_seq_len = 200, use_pts = False):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    print("Preparing Data")
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        inputs, s_len = create_input(data, parameters, add_label=False, singletons=None, padding=padding,
                                     max_seq_len=max_seq_len,
                                     use_pts=use_pts)

        # if parameters['crf']:
        #     y_preds = np.array(f_eval(*input))[1:-1]
        # else:
        #     y_preds = f_eval(*input).argmax(axis=1)
        # print inputs
        temp = []
        temp.append(s_len)
        y_preds = sess.test(inputs, temp)

        y_reals = np.array(data['tags']).astype(np.int32)

        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    print("run CONLL script")
    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(1000000, 2000000)
    output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)
    scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

    print("Result created")
    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    for line in eval_lines:
        print(line)

    # Remove temp files
    if remove:
        os.remove(output_path)
    os.remove(scores_path)

    # Confusion matrix with accuracy for each tag
    print("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(n_tags)] + ["Percent"])
    ))
    for i in range(n_tags):
        print("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in range(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        ))

    # Global accuracy
    print("%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    ))
    print(eval_lines[1].strip().split()[-1])
    # F1 on all entities
    if remove:
        return float(eval_lines[1].strip().split()[-1])
    else:
        return float(eval_lines[1].strip().split()[-1]), output_path


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags
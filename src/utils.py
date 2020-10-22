import codecs
import re
import pickle

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
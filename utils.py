import codecs
import re


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
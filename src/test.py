from utils import load_sentences, word_mapping
train_sentences = load_sentences("/Users/lch/PycharmProjects/GRAM-CNN/dataset/ncbi/train.eng", True, False)
words = [[x[0].lower() if True else x[0] for x in s] for s in train_sentences]
dico = {}
for items in words:
    for item in items:
        if item not in dico:
            dico[item] = 1
        else:
            dico[item] += 1
sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
for i, v in enumerate(sorted_items):
    print()
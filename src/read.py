from configparser import ConfigParser

def load_parameter():
    cp = ConfigParser()
    cp.read("/Users/lch/PycharmProjects/Bert-LSTM-CNN/Bert-LSTM-CNN/src/config.ini")
    section = cp.sections()[0]
    items = cp.items(section)
    para_dic = {}
    for i in items:
        para_dic[i[0]] = i[1]
    return para_dic


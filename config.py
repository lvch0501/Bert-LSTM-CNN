from configparser import ConfigParser

cp = ConfigParser()
cp.read("config.ini")
section = cp.sections()[0]
items = cp.items(section)
def load_parameter():
    para_dic = {}
    for i in items:
        para_dic[i[0]] = i[1]
    return para_dic



from collections import OrderedDict
from config import load_parameter

config_para = load_parameter()

parameters = OrderedDict()
#IOB OR IOEB
parameters['padding'] = config_para["padding"] == '1'
parameters['tag_scheme'] = config_para["tag_scheme"]
parameters['lower'] = config_para["lower"] == '1'
parameters['zeros'] = config_para["zeros"] == '1'
parameters['char_dim'] = config_para["char_dim"]
parameters['char_lstm_dim'] = config_para["char_lstm_dim"]
parameters['char_bidirect'] = config_para["char_bidirect"] == '1'
parameters['word_dim'] = config_para["word_dim"]
parameters['word_lstm_dim'] = config_para["word_lstm_dim"]
parameters['word_bidirect'] = config_para["word_bidirect"] == '1'
parameters['pre_emb'] = config_para["pre_emb"]
parameters['all_emb'] = config_para["all_emb"] == '1'
parameters['cap_dim'] = config_para["cap_dim"]
parameters['crf'] = config_para["crf"] == '1'
parameters['dropout'] = config_para["dropout"]
parameters['lr_method'] = config_para["lr_method"]
parameters['use_word'] = config_para["use_word"] == '1'
parameters['use_char'] = config_para["use_char"] == '1'
parameters['hidden_layer'] = config_para["hidden_layer"]
parameters['reload'] = config_para["reload"] == '1'
#parameters['kernels'] = [2,3,4,5] if type(config_para.kernel_size) == str else map(lambda x : int(x), opts.kernel_size)
parameters['kernels'] = config_para["kernels"]
#parameters['num_kernels'] = [100,100,100,100] if type(config_para.kernel_num) == str else map(lambda x : int(x), opts.kernel_num)
parameters['num_kernels'] = config_para["kernel_num"]
parameters['pts'] = config_para["pts"] == '1'



print()
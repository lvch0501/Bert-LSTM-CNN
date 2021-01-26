import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pickle
import numpy as np
file_path = "../models/use_wordFalse use_charTrue drop_out0.5 hidden_size100 hidden_layer1 lowerTrue allembTrue kernels2, 3, 4 num_kernels40, 40, 40 paddingFalse ptsFalse w_emb200/plot.pkl"

result = {}
with open(file_path, "rb+") as f:
    result = pickle.load(f)

x = list(range(len(result["F1"])))
y = [float(i) for i in result["F1"]]
y1_max=np.argmax(y)
show_max='['+str(y1_max)+' '+str(y[y1_max])+']'
plt.plot(y1_max,y[y1_max],'ko')
plt.annotate(show_max,xy=(y1_max,y[y1_max]),xytext=(y1_max,y[y1_max]))
max = np.max(y)
plt.plot(x, y)
y_major_locator = MultipleLocator(10)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim(-0.5, len(x))
plt.ylim(60, 110)
plt.savefig("../models/use_wordFalse use_charTrue drop_out0.5 hidden_size100 hidden_layer1 lowerTrue allembTrue kernels2, 3, 4 num_kernels40, 40, 40 paddingFalse ptsFalse w_emb200/11.png")
print()
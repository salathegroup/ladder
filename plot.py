#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import re
import glob


#Epoch :: 0 Accuracy :: 10.174



f, ax = plt.subplots()

#"Epoch :: 78 Accuracy :: 60.9889"


_file = "train_log"

for _file in glob.glob("results.log*") + glob.glob("train_log*"):
    print _file.split("/")[-1]
    e = []
    a = []
    for _line in open(_file, "r").readlines():
        s = re.findall("'Epoch : (.*)  Accuracy: ', (.*), '%'", _line)
        if len(s) == 0:
            s = re.findall("Epoch : (.*) Accuracy : (.*)%", _line)
        if len(s) == 0:
           s = re.findall("Epoch :: (.*) Accuracy :: (.*)", _line)
 
        if len(s) > 0:
            e.append(int(s[0][0]))
            a.append(float(s[0][1]))
    _name = _file.split("/")[-1].replace("results.log","").replace("train_log", "")
    _temp = zip(e,a)
    _temp = sorted(_temp)

    e = [x[0] for x in _temp]
    a = [x[1] for x in _temp]
    if len(e) > 0:    
        ax.plot(e, a, label=_name)


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right')
plt.savefig("plot.png")

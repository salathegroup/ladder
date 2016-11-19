#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import re
import glob


#Epoch :: 0 Accuracy :: 10.174



f, ax = plt.subplots()
for _file in glob.glob("results.log*"):
    print _file
    e = []
    a = []
    for _line in open(_file, "r").readlines():
        s = re.findall("Epoch : (.*)  Accuracy: ', (.*), '%'", _line)
        if len(s) == 0:
            s = re.findall("Epoch : (.*) Accuracy : (.*)%", _line)
        if len(s) > 0:
            e.append(int(s[0][0]))
            a.append(float(s[0][1]))

    _name = _file.split("/")[-1].replace("results.log","")
    if _name == "":
        _name += "CURRENT"
    ax.plot(e, a, label=_name)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right')
plt.savefig("plot.png")

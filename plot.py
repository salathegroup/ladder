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
e = []
a = []
for _line in open(_file, "r").readlines():
    s = re.findall("Epoch :: (.*) Accuracy :: (.*)", _line)
    if len(s) > 0:
        e.append(int(s[0][0]))
        a.append(float(s[0][1]))
ax.plot(e, a, label="CURRENT")

for _file in glob.glob("results.log*"):
    if _file.split("/")[-1] == "results.log":
        continue
    print _file.split("/")[-1]
    if _file.split("/")[-1] == "results.log":
        print "Ignoring : ", _file
        continue
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
    _name = _file.split("/")[-1].replace("results.log","")
    _temp = zip(e,a)
    _temp = sorted(_temp)

    e = [x[0] for x in _temp]
    a = [x[1] for x in _temp]
    ax.plot(e, a, label=_name)


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right')
plt.savefig("plot.png")

import input_data

print "===  Loading Data ==="
pv = input_data.read_data_sets("plantvillage_data", n_labeled=100, one_hot=True)
print pv

import profiling
from profiling import record
import model.alexnet as alexnet

import torch
import torch.nn as nn
from torch.autograd import Variable

# Iteration number
iter = 3

#
# Create model
#
model = alexnet.alexnet()

#
# Call PyTorch profiling tool (hook will be added automaticly).
#
profiling.profiling(model)

#
# Run model, we will see the profiling result.
#

for i in xrange(iter):
	# Forward:
	output = model.forward(Variable(torch.ones(2, 3, 224, 224), requires_grad=True))

	# Backward:
	grads = torch.ones(2, 1000)
	output.backward(grads);



# Print record

for i in xrange(iter):
	print("\n================================= Iteration {} =================================".format(i))

	
	layer_num = len(record['forward']) / iter 

	print("\nFORWARD:\n")
	for j in xrange(layer_num):
		record_item = record['forward'][i * layer_num + j]
		print("layer{:3d}:    {:.6f} ms 			({})".format(j, record_item[2] - record_item[1], record_item[0]))

	print("\nBACKWARD:\n")
	for j in (xrange(layer_num)):
		record_item = record['backward'][i * layer_num + layer_num - j - 1]
		print("layer{:3d}:    {:.6f} ms 			({})".format(j, record_item[2] - record_item[1], record_item[0]))


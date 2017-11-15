import profiling
from profiling import record
import model.alexnet as alexnet

import torch
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


#
# Print record
#
for i in xrange(iter):
	print("================================= Iteration {}: =================================".format(i))

	layer_num = len(record) / iter 
	for j in xrange(layer_num):
		print("layer{:3d}: {}".format(j, record[i * layer_num + j]))


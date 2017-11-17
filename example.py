from profiling import Profiling
# from profiling import record
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
# Use case 1: use it as context-manager
#

# profiler will measure the following 3 iterations.
with Profiling(model) as p:
	for i in xrange(iter):
		# Forward:
		output = model.forward(Variable(torch.ones(2, 3, 224, 224), requires_grad=True))

		# Backward:
		grads = torch.ones(2, 1000)
		output.backward(grads);

# profiler won't measure the following 2 extra iterations, since they are out of profiler's range.
for i in xrange(2):
	# Forward:
	output = model.forward(Variable(torch.ones(2, 3, 224, 224), requires_grad=True))

	# Backward:
	grads = torch.ones(2, 1000)
	output.backward(grads);

# profiler will print the 3 iterations result rather than 5.
print(p)



#
# Use case 2: use it directly
#

# p = Profiling(model).start()

# for i in xrange(iter):
# 	# Forward:
# 	output = model.forward(Variable(torch.ones(2, 3, 224, 224), requires_grad=True))

# 	# Backward:
# 	grads = torch.ones(2, 1000)
# 	output.backward(grads);

# p.stop()

# # profiler won't measure the following 2 extra iterations, since they are out of profiler's range.
# for i in xrange(2):
# 	# Forward:
# 	output = model.forward(Variable(torch.ones(2, 3, 224, 224), requires_grad=True))

# 	# Backward:
# 	grads = torch.ones(2, 1000)
# 	output.backward(grads);

# print(p)

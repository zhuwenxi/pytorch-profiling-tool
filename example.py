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

with Profiling(model) as p:
	for i in xrange(iter):
		# Forward:
		output = model.forward(Variable(torch.ones(2, 3, 224, 224), requires_grad=True))

		# Backward:
		grads = torch.ones(2, 1000)
		output.backward(grads);

p.print_result(iter)

#
# Use case 2: use it directly
#

# p = Profiling(model).run()

# for i in xrange(iter):
# 	# Forward:
# 	output = model.forward(Variable(torch.ones(2, 3, 224, 224), requires_grad=True))

# 	# Backward:
# 	grads = torch.ones(2, 1000)
# 	output.backward(grads);

# p.print_result(iter)
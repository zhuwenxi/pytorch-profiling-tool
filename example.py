import profiling
import model.alexnet as alexnet

import torch
from torch.autograd import Variable

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

# Forward:
output = model.forward(Variable(torch.ones(1, 3, 224, 224), requires_grad=True))

# Backward:
grads = torch.ones(1, 1000)
output.backward(grads);
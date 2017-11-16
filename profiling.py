import torch
from torch.autograd import Variable
from torch.autograd import Function

import time

# store the original "__call__" functions, which will be called by following "wrapper_call()"
origin_call = {}

# record layer time
layer_time_dict = {}

record = {'forward':[], 'backward': []}

# runtime hook all modules recursively, for forward data collecting.
def hook_modules(module):
	sub_modules = module.__dict__['_modules']

	for name, sub_module in sub_modules.items():
		# nn.Module is the only thing we care about.
		if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
			break

		if isinstance(sub_module, torch.nn.Container) or isinstance(sub_module, torch.nn.Sequential):
			#
			# nn.Container or nn.Sequential who have sub nn.Module. Recursively visit and hook their decendants.
			#
			hook_modules(sub_module)
		else:
			#
			# nn.Module who doesn't have sub nn.Module, hook it.
			#

			# Wrapper function to "__call__", with time counter in it.
			def wrapper_call(self, *input, **kwargs):
				start_time = time.time()
				result = origin_call[self.__class__](self, *input, **kwargs)
				stop_time = time.time()

				that = self
				def backward_pre_hook(*args):
					# print("pre {}: {}".format(that, id(that)))
					record['backward'].append((that, time.time()))

				# print("result.grad_fn: {}, addr: {}".format(result.grad_fn, id(result.grad_fn)))
				# print("result addr: {}".format(id(result)))
				# result.register_hook(backward_pre_hook)

				result.grad_fn.register_pre_hook(backward_pre_hook);

				# print("{:90s} forward: {:.4f} ms".format(self, stop_time - start_time))
				global record
				record['forward'].append((self, start_time, stop_time))

				return result

			# Replace "__call__" with "wrapper_call".
			if sub_module.__class__ not in origin_call:
				origin_call.update({sub_module.__class__: sub_module.__class__.__call__})
				sub_module.__class__.__call__ = wrapper_call

			def backward_post_hook(*args):
				# print("post {}: {}".format(args[0], id(args[0])))
				record['backward'][-1] = (record['backward'][-1][0], record['backward'][-1][1], time.time()) 
			
			sub_module.register_backward_hook(backward_post_hook)

def profiling(model):
	if isinstance(model, torch.nn.Module) is False:
		print("Not a valid model, please provide a 'nn.Module' instance.")

	hook_modules(model)
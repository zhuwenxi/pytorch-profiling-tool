import torch
from torch.autograd import Variable
from torch.autograd import Function

import time

class Profiling(object):
	def __init__(self, model):
		if isinstance(model, torch.nn.Module) is False:
			print("Not a valid model, please provide a 'nn.Module' instance.")

		self.model = model
		self.record = {'forward':[], 'backward': []}
		self.profiling_on = True
		self.origin_call = {}

	def __enter__(self):
		self.profiling_on = True
		self.run()
		return self

	def __exit__(self, *args):
		self.profiling_on = False

	def run(self):
		self.hook_modules(self.model)
		return self

	def print_result(self, iter):
		for i in xrange(iter):
			print("\n================================= Iteration {} =================================".format(i + 1))

	
			layer_num = len(self.record['forward']) / iter 

			print("\nFORWARD:\n")
			for j in xrange(layer_num):
				record_item = self.record['forward'][i * layer_num + j]
				print("layer{:3d}:    {:.6f} ms 			({})".format(j, record_item[2] - record_item[1], record_item[0]))

			print("\nBACKWARD:\n")
			for j in (xrange(layer_num)):
				record_item = self.record['backward'][i * layer_num + layer_num - j - 1]
				print("layer{:3d}:    {:.6f} ms 			({})".format(j, record_item[2] - record_item[1], record_item[0]))

	def hook_modules(self, module):
		this_profiler = self

		sub_modules = module.__dict__['_modules']

		for name, sub_module in sub_modules.items():
			# nn.Module is the only thing we care about.
			if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
				break

			if isinstance(sub_module, torch.nn.Container) or isinstance(sub_module, torch.nn.Sequential):
				#
				# nn.Container or nn.Sequential who have sub nn.Module. Recursively visit and hook their decendants.
				#
				self.hook_modules(sub_module)
			else:
				#
				# nn.Module who doesn't have sub nn.Module, hook it.
				#

				# Wrapper function to "__call__", with time counter in it.
				def wrapper_call(self, *input, **kwargs):
					start_time = time.time()
					result = this_profiler.origin_call[self.__class__](self, *input, **kwargs)
					stop_time = time.time()

					that = self
					def backward_pre_hook(*args):
						if (this_profiler.profiling_on):
							# print("pre {}: {}".format(that, id(that)))
							this_profiler.record['backward'].append((that, time.time()))

					result.grad_fn.register_pre_hook(backward_pre_hook);

					if (this_profiler.profiling_on):
						global record
						this_profiler.record['forward'].append((self, start_time, stop_time))
						# print("{:90s} forward: {:.4f} ms".format(self, stop_time - start_time))


					return result

				# Replace "__call__" with "wrapper_call".
				if sub_module.__class__ not in this_profiler.origin_call:
					this_profiler.origin_call.update({sub_module.__class__: sub_module.__class__.__call__})
					sub_module.__class__.__call__ = wrapper_call

				def backward_post_hook(*args):
					if (this_profiler.profiling_on):
						this_profiler.record['backward'][-1] = (this_profiler.record['backward'][-1][0], this_profiler.record['backward'][-1][1], time.time()) 
			
				sub_module.register_backward_hook(backward_post_hook)
import torch
from torch.autograd import Variable
from torch.autograd import Function

import time

class Profiling(object):
    def __init__(self, model, forward_only=False):
        if isinstance(model, torch.nn.Module) is False:
            print("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.record = {'forward':[], 'backward': []}
        self.profiling_on = True
        self.origin_call = {}
        self.hook_done = False
        self.layer_num = 0

        self.backward_record = []
        self.seen_backward_func = []
        self.prev_backward_func = []
        self.has_release_seen_list = False
        
        self.forward_only = True

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):

        ret = ""

        time_dict = {'forward': [], 'backward': []}

        global TD
        TD = time_dict

        iter = int(len(self.record['forward']) / self.layer_num)

        for i in range(iter):
            iter_forward = 0.0
            iter_backward = 0.0

            ret += "\n================================= Iteration {} =================================\n".format(i + 1)
    
            ret += "\nFORWARD TIME:\n\n"
            for j in range(self.layer_num):

                record_item = self.record['forward'][i * self.layer_num + j]
                ret += "layer{:3d}:          {:.6f} s          ({})\n".format(j + 1, record_item[2] - record_item[1], record_item[0])

                iter_forward += record_item[2] - record_item[1]

                if j >= len(time_dict['forward']):
                    assert j is len(time_dict['forward'])
                    time_dict['forward'].append((record_item[0], record_item[2] - record_item[1]))
                else:
                    time_dict['forward'][j] = (time_dict['forward'][j][0], time_dict['forward'][j][1] + record_item[2] - record_item[1])

            ret += "\nTOTAL(forward):          {:.6f} s          \n".format(iter_forward)

            if self.forward_only is False:
                ret += "\nBACKWARD TIME:\n\n"
                for j in (range(self.layer_num)):
                    record_item = self.record['backward'][i * self.layer_num + self.layer_num - j - 1]

                    iter_backward += record_item[2] - record_item[1]
                    try:
                        ret += "layer{:3d}:          {:.6f} s          ({})\n".format(j + 1, record_item[2] - record_item[1], record_item[0])

                        if j >= len(time_dict['backward']):
                            assert j is len(time_dict['backward'])
                            time_dict['backward'].append((record_item[0], record_item[2] - record_item[1]))
                        else:
                            time_dict['backward'][j] = (time_dict['backward'][j][0], time_dict['backward'][j][1] + record_item[2] - record_item[1])
                    except:
                        print("Oops, this layer doesn't execute backward post-hooks")
                        pass

                ret += "\nTOTAL(backward):          {:.6f} s          \n".format(iter_backward)


        ret += "\n================================= Average =================================\n"

        average_forward = 0.0
        average_backward = 0.0

        ret += "\nFORWARD TIME:\n\n"
        for i in range(self.layer_num):
            ret += "layer{:3d}:          {:.6f} s          ({})\n".format(i + 1, time_dict['forward'][i][1] / iter, time_dict['forward'][i][0]) 
            average_forward += time_dict['forward'][i][1] / iter

        ret += "\nTOTAL(forward):          {:.6f} s          \n".format(average_forward)

        if self.forward_only is False:
            ret += "\nBACKWARD TIME:\n\n"
            for i in range(self.layer_num):
                ret += "layer{:3d}:          {:.6f} s          ({})\n".format(i + 1, time_dict['backward'][i][1] / iter, time_dict['backward'][i][0]) 
                average_backward += time_dict['backward'][i][1] / iter

            ret += "\nTOTAL(backward):          {:.6f} s          \n".format(average_backward)

        return ret

    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.hook_modules(self.model)

        self.profiling_on = True

        return self

    def stop(self):
        self.profiling_on = False

        return self

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

                self.layer_num += 1
                #
                # nn.Module who doesn't have sub nn.Module, hook it.
                #

                # Wrapper function to "__call__", with time counter in it.
                def wrapper_call(self, *input, **kwargs):
                    if hasattr(self, '__hooked'):
                        # print("{} has been hooked".format(self))
                        start_time = time.time()
                    else:
                        pass
                        # print("what??? {}".format(self))
                    result = this_profiler.origin_call[self.__class__](self, *input, **kwargs)
                    if hasattr(self, '__hooked'):
                        stop_time = time.time()
                        
                    if hasattr(self, '__hooked'):
                        that = self
                        def backward_pre_hook(*args):
                            if (this_profiler.profiling_on):
                                # print('backward_pre_hook')
                                if that in [x[0] for x in this_profiler.record['backward']] and [x[0] for x in this_profiler.record['backward'][::-1]].index(that) < this_profiler.layer_num - 1:
                                    # print("that: {}".format(that, this_profiler.record['backward'], this_profiler.layer_num))
                                    pass
                                else:
                                    # print("append: {}({})".format(that, hex(id(that))))
                                    this_profiler.record['backward'].append((that, time.time()))

                            if len(this_profiler.record['backward']) % this_profiler.layer_num is 1 and len(this_profiler.record['backward']) > this_profiler.layer_num:
                                if this_profiler.has_release_seen_list is False:
                                    # here, we release memory
                                    this_profiler.seen_backward_func = []
                                    this_profiler.has_release_seen_list = True

                            if len(this_profiler.record['backward']) % this_profiler.layer_num is 2 and len(this_profiler.record['backward']) > this_profiler.layer_num:
                                this_profiler.has_release_seen_list = False
                        
                        def backward_post_hook(*args):
                            if (this_profiler.profiling_on):
                                # print('backward_post_hook')
                                if that in [x[0] for x in this_profiler.record['backward']]:
                                    index_of_that = len(this_profiler.record['backward']) - [x[0] for x in this_profiler.record['backward']][::-1].index(that) - 1
                                    record_of_that = this_profiler.record['backward'][index_of_that]
                                    assert record_of_that[0] is that

                                    this_profiler.record['backward'][index_of_that] = (record_of_that[0], record_of_that[1], time.time())
                                else:
                                    raise Error("Oops, this record is broken!")

                        
                        

                        def hook_result(result):
                            if isinstance(result, tuple) or isinstance(result, list):
                                for r in result:
                                    hook_result(r) 
                            elif isinstance(result, Variable):
                                result.grad_fn.register_pre_hook(backward_pre_hook);
                                # print("=========================== for {} ===========================".format(that))
                                this_profiler.register_backward_hook(result.grad_fn, backward_pre_hook, backward_post_hook)
                                # print("=========================== done ===========================")
                            else:
                                # raise AssertionError('result (%s) is neither a "tuple" nor a "Variable"!' % type(result))
                                # print('WARNING: result (%s) is neither a "tuple" nor a "Variable"!' % type(result))
                                pass

                        if this_profiler.forward_only is False:
                            hook_result(result)

                        if (this_profiler.profiling_on):
                            this_profiler.record['forward'].append((self, start_time, stop_time))

                    return result

                # Replace "__call__" with "wrapper_call".
                if sub_module.__class__ not in this_profiler.origin_call:
                    this_profiler.origin_call.update({sub_module.__class__: sub_module.__class__.__call__})
                    sub_module.__class__.__call__ = wrapper_call

                # This is a workaround,
                # since "sub_module.__hooked = True" surprisingly add a "_Profiling__hooked" attribute to sub_module instead of "__hooked".
                # I can not figure out why...
                object.__setattr__(sub_module, '__hooked', True)
                assert hasattr(sub_module,'__hooked')

                 
            
                # sub_module.register_backward_hook(backward_post_hook)

    def register_backward_hook(self, grad_fn, backward_pre_hook, backward_post_hook):
        # print("register pre/post hook for {} ({})".format(grad_fn, hex(id(grad_fn))))
        grad_fn.register_pre_hook(backward_pre_hook)
        grad_fn.register_hook(backward_post_hook);

        if grad_fn not in self.seen_backward_func:
            self.seen_backward_func.append(grad_fn)

        next_functions = grad_fn.next_functions
        if next_functions is not () and len(next_functions) is not 0:
            for k, v in next_functions:
                if k is not None:
                    if k not in self.seen_backward_func:
                        self.register_backward_hook(k, backward_pre_hook, backward_post_hook)
                    else:
                        pass
                        # print("!!!!!!!!!!!!!!!!!has seen: {}".format(k))


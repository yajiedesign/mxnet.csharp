import numpy as np


class UnitTestPythonWarp(object):
    def __init__(self):
        self.index = 0

    def eval_array(self, code, **arrs):
        self.index = 0
        dic = {}
        for key, value in arrs.items():  # styles is a regular dictionary
            shape = eval(value["shape"])
            dic[key] = np.array(value["value"])
            dic[key].shape = shape
        eval_ret = eval(code, globals(), dic)
        print(eval_ret.shape)
        print(eval_ret)
        return eval_ret.flat[:].tolist()
        pass

    def eval_scalar(self, code, **arrs):
        self.index = 0
        dic = {}
        for key, value in arrs.items():  # styles is a regular dictionary
            shape = eval(value["shape"])
            dic[key] = np.array(value["value"])
            dic[key].shape = shape
        eval_ret = eval(code, globals(), dic)
        print(eval_ret.shape)
        print(eval_ret)
        return int(eval_ret)
        pass

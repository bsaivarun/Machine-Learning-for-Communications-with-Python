# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(17)
np.set_printoptions(precision=5, linewidth=160, suppress=True)

class DataSet:
    __x_range = [0, 1]
    __sigma = 0.1
    __N_samples = 10
    _y_D = None
    _x_D = None
    
    @staticmethod
    def __gt_func(x):
        return np.sin(2*np.pi*x)

    @classmethod
    def get_data(cls):
        if cls._y_D is None and cls._x_D is None:
            cls._x_D = np.random.uniform(cls.__x_range[0], cls.__x_range[1], (1,cls.__N_samples))
            cls._y_D = cls.__gt_func(cls._x_D) + np.random.normal(0, cls.__sigma, cls._x_D.shape)
        return cls._y_D,cls._x_D
    
    @classmethod
    def get_ground_truth_data(cls):
        x = np.linspace(cls.__x_range[0], cls.__x_range[1]*1.0, 1000).reshape((1,-1))
        y = cls.__gt_func(x)
        return y,x

    @classmethod
    def plot_data(cls):
        y,x = cls.get_data()
        y_GT,x_GT = cls.get_ground_truth_data()
        plt.plot(x[0], y[0], 'o', x_GT[0], y_GT[0],'r--')
        plt.legend(['data','ground_truth (unknown)'])
                
    @classmethod
    def plot_model(cls, w, extend_data, norm_param=None):  
        w = np.array(w).reshape((-1,1))  
        cls.plot_data()
        x = np.linspace(cls.__x_range[0],cls.__x_range[1], 100).reshape((1,-1))
        xe = extend_data(x)
        if norm_param is not None:
            mean = norm_param['mean'].reshape((-1,1))
            stdd = np.sqrt(norm_param['var'].reshape((-1,1)))
            xe = (xe - mean)/stdd
        y = np.dot(w.T, xe)
        plt.plot(x[0], y[0], '-')
        plt.legend(['data','ground_truth (unknown)', 'model'])
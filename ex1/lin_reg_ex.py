
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
import mlcomm.nn.utils as nnu
from dataset1_linreg import DataSet  

#get and plot the data
y_D,x_D = DataSet.get_data()
DataSet.plot_data()
plt.show()


#extend x with ones:
x_D = nnu.poly_extend_data1D(x_D)


#random init of w:
### YOUR CODE HERE ###
w = np.random.normal(0,1,(x_D.shape[0],1))
#normalization:
x_D, norm_param = nnu.normalize_data(x_D)


#plot and compute cost
def extension_wrapper(x):
    return nnu.poly_extend_data1D(x)
DataSet.plot_model(w, extension_wrapper, norm_param)
plt.show()
print('Cost:%f' % nnu.lir_cost(w, y_D, x_D))


#compute gradient and do gradient descent
def gradient_wrapper(w):
    return nnu.lir_grad(w, y_D, x_D)
w = nnu.gradient_descent(1000, 0.05, w, gradient_wrapper)


#plot and compute cost
DataSet.plot_model(w, extension_wrapper, norm_param)
plt.show()
print('Cost:%f' % nnu.lir_cost(w, y_D, x_D))


import numpy as np

def lor_cost(w, y, x, lbd=0):
    """
    Computes cost for logistic regression with parameters w and data set x,y
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    cost = scalar
    """
    ## YOUR CODE HERE ###
    
    #####################
    return cost
    
def lor_grad(w, y, x, lbd=0): 
    """
    Returs gradient for logistic regression with the cross entropy loss function 
    for parameter w and data set y, x.
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    gradT = np array of size Mx1
    """
    ## YOUR CODE HERE ###
    
    #####################
    return gradT

def poly_extend_data2D(x, p=1):
    """
    Extend the provided input matrix x wtih all subsequent powers of terms of the input.
    x = np.array of size 2xN
    Output:
    x_e = np.array 
    Eg. for p=3 and x of dimensions 2xN. x_e should be a matrix such that 
    the 1st row is [1 1 .. 1], 2nd X[0,:], 3rd X[1,:], 4th X[0,:]**2,
    5th X[0,:]*X[1,:], 6th X[1,:]*2, 7th X[0,:]**3,  8th X[0,:]**2*X[1,:], 
    and so on... till 10th row equal X[1,:]**3 
    """      
    ### YOUR CODE HERE ###
     
    ### ######### ###    
    return x_e 

def normalize_data(x):
    """
    Normalizes data. Should not normalize the first row (we assume it is the row of ones).
    x = np.array of size MxN
    Output:
    x_norm     = normalized np.array of size MxN    
    norm_param = distionary with two keys "mean" and "var". Each key contains 
    a np.array of size Mx1 with the mean and variance of each row of data array. 
    For the first row,  set mean=0 and var=1
    """
    ### YOUR CODE HERE ###
    
	########################
    return x_norm, norm_param


def lir_grad(w, y, x): 
    """
    Returs gradient for linear regression with quadratic cost for parameter w and data set y, x.
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    gradT = np array of size Mx1
    """
    ### YOUR CODE HERE ###
    gradT = x.dot((w.T.dot(x)-y).T)
	######################
    return gradT



def gradient_descent(iter_num, l_rate, w_0, gradient_func):
    """
    Performs gradient descent for iter_num iterations with learning rate l_rate from initial
    position w_0. 
    w_0 = np array of size Mx1
    gradient_func(w) is a function which returns gradient for parameter w
    Output:
    w_opt = optimal parameters
    """
    ### YOUR CODE HERE ###
    w_opt = w_0
    for i in range(iter_num):
        w_opt = w_opt - l_rate*gradient_func(w_opt)
	######################
    return w_opt


def poly_extend_data1D(x, p=1):
    """
    Extend the provided input vector x, wtih subsequent powers of the input.
    x = np.array of size 1xN
    Output:
    x_e = np.array of size (p+1)xN such that 1st row = x^0, 2nd row = x^1, ...
    """      
    ### YOUR CODE HERE ###
    x_e = np.vstack(np.sin([2*np.pi*x*i for i in range(p)]/np.amax(x)))
    x_e[0,:] = np.ones(1,x.shape[1])    
	########################
    return x_e

def sin_extend_data1D(x, p):
    """
    Extend the provided input vector x, wtih P subsequent sin harmonics of the input.
    x = np.array of size 1xN
    Output:
    x_e = np.array of size (p+1)xN
    """      
    ### YOUR CODE HERE ###

    ######################    
    return x_e 

def lir_cost(w, y, x):
    """
    Computes cost for linear regression with parameters w and data set x,y
    y = np.array of size 1xN
    x = np.array of size MxN
    w = np array of size Mx1
    Output:
    cost = scalar
    """
    ## YOUR CODE HERE ###
    a = np.dot(w.T,x)
    cost = np.sum((y-a)**2)/2
    #####################
    return cost

def act_fct(x, type_fct):
    """
    Implements different activation functions to be used in Neural Networks. The
    variable x is the function parameter and type_func defines which functions should
    be chosen, i.e., y = f(x). Valid choices are
    
    'identity': y = f(x) = x
    'sigmoid': y = f(x) = 1/(1+exp(-x))
    'tanh': y = f(x) = tanh(x)
    'rect_lin_unit': y = f(x) = max(x,0)


    """
    if type_fct not in ['identity', 'sigmoid', 'tanh', 'rect_lin_unit']:
        raise ValueError('activation function type {} is not known'.format(type_fct))
    
    x = np.asarray(x, dtype=float)

    if type_fct == 'identity':
        y = x
    elif type_fct == 'sigmoid':
        y = 1/(1+np.exp(-x))
    elif type_fct == 'tanh':
        y = np.tanh(x)
    elif type_fct == 'rect_lin_unit':
        y = np.max(np.vstack((x, np.zeros(x.shape))), axis=0)

    return y
    
def dact_fct(x, type_fct):
    """
    Implements derivatives of activation functions to be used in Neural Networks. The
    Inputs:
        x = np.array of input values
        type_act = 
             'identity' : for activation  y = f(x) = x
             'sigmoid': for activation y = f(x) = 1/(1+exp(-x))
             'tanh': for activation y = f(x) = tanh(x)
             'rect_lin_unit': for activation y = f(x) = max(x,0)
    Output:         
       y = np.array containing f'(x)  	
    """
    y = None
    ###YOUR CODE HERE###
    
    ##################    
    return y
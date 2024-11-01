import numpy as np
def predict(X, theta):
    return X @ theta

def cost(X, y, theta):
    
    predicted = predict(X, theta)
    sqr_error = (predicted - y) ** 2
    sum_error = np.sum(sqr_error)
    m = np.size(y)
    return (1 / (2 * m)) * sum_error

def gradientDescent(X, y, alpha = 0.02, iter = 5000):
    
    theta = np.zeros(X.shape[1])
    m = np.size(y)
    X_T = np.transpose(X)
    pre_cost = cost(X, y, theta)
    for i in range(0, iter):
        error = predict(X, theta) - y
        theta = theta - (alpha / m) * (X_T @ error)
        cur_cost = cost(X, y, theta)
        if np.round(cur_cost, 15) == np.round(pre_cost, 15):
            break
        pre_cost = cur_cost
    return theta
    
    
    
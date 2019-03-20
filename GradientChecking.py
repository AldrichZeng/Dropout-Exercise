import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils  # 第一部分，初始化
import reg_utils  # 第二部分，正则化
import gc_utils  # 第三部分，梯度校验


def forward_propagation(x,theta):
    """
    一维线性模型
    """
    J=np.dot(theta,x)
    return J


def backward_propagation(x,theta):
    """
    反向传播
    """
    d_theta=x
    return d_theta


def gradient_check(x,theta,epsilon=1e-7):
    thetaplus=theta+epsilon
    thetaminus=theta-epsilon
    J_plus=forward_propagation(x,thetaplus)
    J_minus=forward_propagation(x,thetaminus)
    gradapprox=(J_plus-J_minus)/(2*epsilon)

    grad=backward_propagation(x,theta)
    numerator=np.linalg.norm(grad-gradapprox)  # 线性代数中的2范数（默认）
    denominator=np.linalg.norm(grad)+np.linalg.norm(gradapprox) # 线性代数中的2范数（默认）
    difference=numerator/denominator

    if difference<1e-7:
        print("Gradient Checking：梯度正常！")
    else:
        print("Gradient Checking: 梯度找出阈值！")

    return difference


def forward_propagation_n(X,Y,parameters):
    m=X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    #Linear->ReLU->Linear->ReLU->Linear->Sigmoid
    Z1 = np.dot(W1, X) + b1
    A1 = gc_utils.relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = gc_utils.relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = gc_utils.sigmoid(Z3)

    # 计算成本
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = (1 / m) * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backward_propagation_n(X,Y,cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1. / m) * np.dot(dZ3, A2.T)
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    # dW2 = 1. / m * np.dot(dZ2, A1.T) * 2  # Should not multiply by 2
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def gradient_check_n(parameters,gradients,X,Y,epsilon=1e-7):
    # 初始化参数
    parameters_values, keys = gc_utils.dictionary_to_vector(parameters)  # 将parameters字典转换为array
    grad=gc_utils.gradients_to_vector(gradients)
    num_parameters=parameters_values.shape[0]
    J_plus=np.zeros((num_parameters,1))
    J_minus=np.zeros((num_parameters,1))
    gradapprox=np.zeros((num_parameters,1))

    # 计算grad approx
    for i in range(num_parameters): # 遍历所有的参数
        # 计算 J_plus[i]
        thetaplus=np.copy(parameters_values)
        thetaplus[i][0]=thetaplus[i][0]+epsilon
        J_plus[i],cache=forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaplus))  # cache用不到

        # 计算 J_minus[i]
        thetaminus=np.copy(parameters_values)
        thetaminus[i][0]=thetaminus[i][0]-epsilon
        J_minus[i],cache=forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaminus))  # cache用不到

        # 计算 grad apporx[i]
        gradapprox[i]=(J_plus[i]-J_minus[i])/(2*epsilon)

    # 通过计算差异比较 gradapprox 和后向传播梯度
    numerator=np.linalg.norm(grad-gradapprox)
    denominator=np.linalg.norm(grad)+np.linalg.norm(gradapprox)
    difference=numerator/denominator

    if difference<1e-7:
        print("Gradient Checking: 梯度正常！")
    else:
        print("Gradient Checking:梯度超出阈值！")
    return difference












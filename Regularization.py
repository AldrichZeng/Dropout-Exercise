import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils  # 第一部分，初始化
import reg_utils  # 第二部分，正则化
import gc_utils  # 第三部分，梯度校验
plt.rcParams['figure.figsize']=(7.0,4.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'


# 实现一个3层网络 Linear->ReLU->linear->ReLU->Linear->Sigmoid
def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,is_plot=True,lambd=0,keep_prob=1):
    grads={}
    costs=[]
    m=X.shape[1]
    layers_dims=[X.shape[0],20,3,1]  # 每一层的维度

    parameters=reg_utils.initialize_parameters(layers_dims)
    for i in range(0,num_iterations):
        # 前向传播，计算A3
        if keep_prob==1:  # 执行正常的前向传播
            a3,cache=reg_utils.forward_propagation(X,parameters)
        elif keep_prob<1: # 执行dropout，随机失活结点
            a3,cache=forward_propagation_with_dropout(X,parameters,keep_prob)
        else :
            print("keep_prob参数错误，程序退出")
            exit

        # 计算Cost
        if lambd==0: # 不使用正则化
            cost=reg_utils.compute_cost(a3,Y)
        else: # 使用L2正则化
            cost=compute_cost_with_regularization(a3,Y,parameters,lambd)
        assert(lambd==0 or keep_prob==1)

        # 反向传播，计算梯度
        if (lambd==0 and keep_prob==1): # 不使用L2正则化和Dropout
            grads=reg_utils.backward_propagation(X,Y,cache)
        elif lambd!=0:# 使用L2正则化，不使用随机删除结点
            grads=backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob<1:  # 使用dropout随机删除结点，不使用L2正则化
            grads=backward_propagation_with_dropout(X,Y,cache,keep_prob)
        # 更新参数
        parameters=reg_utils.update_parameters(parameters,grads,learning_rate)

        if i % 1000==0:
            costs.append(cost)
            if print_cost:
                print("第"+str(i)+"次迭代，成本值为"+str(cost))
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration (per 100)')
        plt.title('learning_rate='+str(learning_rate))
        plt.show()

    return parameters


def compute_cost_with_regularization(A3,Y,parameters,lambd):
    m=Y.shape[1]
    W1=parameters["W1"]
    W2=parameters["W2"]
    W3=parameters["W3"]
    cross_entropy_cost=reg_utils.compute_cost(A3,Y)
    L2_regularization_cost=(lambd/2/m)*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))
    cost=cross_entropy_cost+L2_regularization_cost
    return cost


def backward_propagation_with_regularization(X,Y,cache,lambd):
    """
    L2正则化的反向传播
    :return: gradients 字典类型，保存梯度
    """
    m=X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)=cache
    dZ3=A3-Y
    dW3=(1/m)*np.dot(dZ3,A2.T)+lambd/m*W3
    db3=(1/m)*np.sum(dZ3,axis=1,keepdims=True)
    dA2=np.dot(W3.T,dZ3)
    dZ2=np.multiply(dA2,np.int64(A2>0))
    dW2=(1/m)*np.dot(dZ2,A1.T)+lambd/m*W2
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dA1=np.dot(W2.T,dZ2)
    dZ1=np.multiply(dA1,np.int64(A1>0))
    dW1=(1/m)*np.dot(dZ1,X.T)+lambd/m*W1
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
    gradients={"dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,
               "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
               "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients


def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    """
    在第1层和第2层采用Inverted Dropout
    """
    np.random.seed(1)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    W3=parameters["W3"]
    b3=parameters["b3"]

    # Linear->ReLU->Linear->ReLU->Linear->Sigmoid
    Z1=np.dot(W1,X)+b1
    A1=reg_utils.relu(Z1)

    # 采用Inverted Dropout
    D1=np.random.rand(A1.shape[0],A1.shape[1])  # 初始化矩阵,与A1具有相同维度。
    D1=D1<keep_prob # 将低于keep_prob的值设置为0，将高于keep_prob的值设置为1
    A1=A1*D1 # 舍弃A1的一些结点，将它的值变为0或False
    A1=A1/keep_prob # 缩放未舍弃的结点的值（这一步最关键，体现了“Inverted”）

    Z2=np.dot(W2,A1)+b2
    A2=reg_utils.relu(Z2)

    # 采用Inverted Dropout
    D2=np.random.rand(A2.shape[0],A2.shape[1])
    D2=D2<keep_prob
    A2=A2*D2
    A2=A2/keep_prob

    Z3=np.dot(W3,A2)+b3
    A3=reg_utils.sigmoid(Z3)

    cache=(Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return A3,cache


def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    """
    dropout的反向传播
    :return: gradients 字典类型，保存梯度
    """
    m=X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3=A3-Y
    dW3=(1/m)*np.dot(dZ3,A2.T)
    db3=(1/m)*np.sum(dZ3,axis=1,keepdims=True)
    dA2=np.dot(W3.T,dZ3)

    dA2=dA2*D2  # 使用前向传播期间相同的结点，舍弃那些关闭的结点（因为任何数乘以0都为0）
    dA2=dA2/keep_prob  # 缩放未舍弃的结点的值

    dZ2=np.multiply(dA2,np.int64(A2>0))
    dW2=(1/m)*np.dot(dZ2,A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dA1=np.dot(W2.T,dZ2)

    dA1=dA1*D1  # 使用前向传播期间相同的结点，舍弃那些关闭的结点（因为任何数乘以0都为0）
    dA1=dA1/keep_prob  # 缩放未舍弃的结点的值

    dZ1=np.multiply(dA1,np.int64(A1>0))
    dW1=(1/m)*np.dot(dZ1,X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

    gradients={"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

















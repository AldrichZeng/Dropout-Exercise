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


# 实现3层神经网络 Linear->ReLU->Linear->ReLU->Linear_Sigmoid
def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he",is_plot=True):
    grads={}
    costs=[]
    m=X.shape[1]
    layers_dims=[X.shape[0],10,5,1]

    # 初始化参数的类型
    if initialization=="zeros":
        parameters=initialize_parameters_zeros(layers_dims)
    elif initialization=="random":
        parameters=initialize_parameters_random(layers_dims)
    elif initialization=="he":
        parameters=initialize_parameters_he(layers_dims)
    else:
        print("初始化参数错误，程序退出")
        exit

    # 开始学习
    for i in range(0,num_iterations):
        # 前向传播
        a3,cache=init_utils.forward_propagation(X,parameters)
        # 计算损失
        cost=init_utils.compute_loss(a3,Y)
        # 反向传播
        grads=init_utils.backward_propagation(X,Y,cache)
        # 更新参数
        parameters=init_utils.update_parameters(parameters,grads,learning_rate)
        # 记录成本
        if i%100==0:
            costs.append(cost)
            if print_cost:
                print("第"+str(i)+"次迭代，成本值为"+str(cost))

    # 绘制成本曲线
    if is_plot:
        plt.plot(costs)
        plt.xlabel('iteration(per 100)')
        plt.ylabel('cost J')
        plt.title('learning_rate='+str(learning_rate))
        plt.show()

    return parameters


# 初始化参数为全0
# 参数 - layers_dims:列表，模型的层数和对应每一层的结点的数量
def initialize_parameters_zeros(layers_dims):
    parameters={}
    L=len(layers_dims)
    for l in range(1,L):
        parameters["W"+str(l)]=np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))
        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))
    return parameters


def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters={}
    L=len(layers_dims)
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*10  # 使用10倍缩放
        parameters["b"+str(l)]=np.random.randn(layers_dims[l],1)

        assert(parameters["W"+str(l)].shape==(layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape==(layers_dims[l],1))
    return parameters



# 抑梯度异常初始化
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters={}
    L=len(layers_dims)
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        # 使用断言确保我的数据格式是正确的
        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters





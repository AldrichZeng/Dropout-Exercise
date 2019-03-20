import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils  # 第一部分，初始化
import reg_utils  # 第二部分，正则化
import gc_utils  # 第三部分，梯度校验
import InitParameters
import Regularization
import GradientChecking

plt.rcParams['figure.figsize']=(7.0,4.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'


# 初始化参数
train_X,train_Y,test_X,test_Y=init_utils.load_dataset(is_plot=False)
#
# print("------------ 测试初始化为全0 -------------")
# parameters = InitParameters.initialize_parameters_zeros([3,2,1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# parameters = InitParameters.model(train_X, train_Y, initialization = "zeros",is_plot=True)
# print ("训练集:")
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print ("测试集:")
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
#
# print("------------ 测试初始化为随机值 -------------")
# parameters = InitParameters.initialize_parameters_random([3, 2, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# parameters = InitParameters.model(train_X, train_Y, initialization = "random",is_plot=True)
# print("训练集：")
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print("测试集：")
# predictions_test = init_utils.predict(test_X, test_Y, parameters)
#
# print(predictions_train)
# print(predictions_test)
#
# plt.title("Model with large random initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)  # 绘制边界
#
# print("------------ 测试初始化为抑梯度随机值 -------------")
# parameters = InitParameters.initialize_parameters_he([2, 4, 1])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# parameters = InitParameters.model(train_X, train_Y, initialization = "he",is_plot=True)
# print("训练集:")
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print("测试集:")
# init_utils.predictions_test = init_utils.predict(test_X, test_Y, parameters)
# plt.title("Model with He initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)


# train_X,train_Y,test_X,test_Y=reg_utils.load_2D_dataset(is_plot=False)
# # train_X.shape=(2,211)  样本数为211
# # train_Y.shape=(1,211)
# print("------------ 不使用正则化 ------------")
# parameters = Regularization.model(train_X, train_Y,is_plot=False)
# print("训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])(train_X, train_Y,is_plot=False)
# print("训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# print("------------ 使用L2正则化 ------------")
# parameters = Regularization.model(train_X, train_Y, lambd=0.7,is_plot=True)
# print("使用正则化，训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("使用正则化，测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

print(train_X.shape)
print("------------ 使用Dropout正则化 ------------")
parameters = Regularization.model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3,is_plot=True)

print("使用随机删除节点，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)  # 准确度
print("使用随机删除节点，测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)  # 准确度

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)



#测试forward_propagation
# print("-----------------测试forward_propagation-----------------")
# x, theta = 2, 4
# J = GradientChecking.forward_propagation(x, theta)
# print ("J = " + str(J))

#测试backward_propagation
# print("-----------------测试backward_propagation-----------------")
# x, theta = 2, 4
# dtheta = GradientChecking.backward_propagation(x, theta)
# print ("dtheta = " + str(dtheta))


#测试gradient_check
# print("-----------------测试gradient_check-----------------")
# x, theta = 2, 4
# difference = GradientChecking.gradient_check(x, theta)
# print("difference = " + str(difference))

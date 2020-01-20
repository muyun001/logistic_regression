# 实现logistic regression 逻辑回归   网址：https://www.jianshu.com/p/eb5f63eaae2b
import numpy as np
from dataset.lr_utils import load_dataset  # 读取数据集

# （一）数据导入和预处理
# 导入数据，“_orig”代表这里是原始数据，我们还要进一步处理才能使用：
# 训练集x（原始），训练集y(真实值)，测试集x（原始），测试集y（预测值），
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# print(train_set_x_orig.shape)  # (209, 64, 64, 3)  209张图片，每张图片是64*64个元素，有“RGB”3个channel
# 由数据集获取一些基本参数，如训练样本数m，图片大小：
# m_train = train_set_x_orig.shape[0]  # 训练集大小209
# m_test = test_set_x_orig.shape[0]  # 测试集大小209
# num_size = train_set_x_orig.shape[1]  # 图片宽度64，大小是64×64
# 向图片数据向量化 （矢量化）
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# 对数据进行标准化（先减去均值，然后除以方差，也就是(x-μ)/σ2）
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# （二）构建辅助函数
# 1.激活函数/sigmoid函数
def sigmoid(z):
    """z是传递过来的函数"""
    return 1 / (1 + np.exp(-z))


# 2.参数初始化函数（给参数都初始化为0）
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))  # ｗ是列向量，dim是传入的维度
    b = 0
    return w, b


# 3. 传播函数（propagate函数）
def propagate(w, b, X, Y):
    """
    传参:
    w -- 权重, shape： (num_px * num_px * 3, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (num_px * num_px * 3, m),m为样本数
    Y -- 真实标签，shape： (1,m)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """
    # 获取样本数m：
    m = X.shape[1]

    # 前向传播：
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m  # 代价函数，求“Y和A的平均差距”

    # 反向传播
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m

    # 返回值
    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


# 4. 优化函数（optimize函数）
def optimize(w, b, X, Y, learning_rate, num_iterations, print_cost=False):
    # 定义一个costs数组，存放每若干次迭代后的cost，从而可以画图看看cost的变化趋势：
    costs = []
    # 进行迭代：
    for i in range(num_iterations):
        gradients, cost = propagate(w, b, X, Y)  # 梯度和cost（代价函数，求“Y和A的平均差距”）
        dw = gradients["dw"]
        db = gradients["db"]

        # 用上面得到的梯度更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        ###############不是必须： 每100次迭代，保存一个cost看看结果,从而随时掌握模型的进展：
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    # 迭代完毕，将最终参数放进字典：
    params = {
        "w": w,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }
    return params, grads, costs


# 5.预测函数（predict函数）
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(m):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction


# (三). 将上面结合起来，搭建模型
def logistic_model(X_train, Y_train, X_test, Y_test, learning_rate=0.1, num_iterations=2000, print_cost=True):
    # 获取特征维度，初始化参数
    dim = X_train.shape[0]
    W, b = initialize_with_zeros(dim)

    # 梯度下降，迭代求出模型参数
    params, grads, costs = optimize(W, b, X_train, Y_train, learning_rate, num_iterations, print_cost)
    w = params['w']
    b = params['b']

    # 用学到的参数进行预测：
    prediction_train = predict(W, b, X_train)
    prediction_test = predict(W, b, X_test)

    # 计算准确率，分别放在训练集和测试集
    """
    我们的predict函数得到的是一个行向量（1，m），这个跟我们的标签Y是一样的形状。我们首先可以让两者相减：
    prediction_test - Y_test，
    如果对应位置相同，则变成0，不同的话要么是1要么是-1，于是再取绝对值：
    np.abs(prediction_test - Y_test)，
    就相当于得到了“哪些位置预测错了”的一个向量，于是我们再求一个均值：
    np.mean(np.abs(prediction_test - Y_test))，
    就是“错误率”了，然后用1来减去它，就是正确率了！
    """
    accuracy_train = 1 - np.mean(np.abs(prediction_train - Y_train))
    accuracy_test = 1 - np.mean(np.abs(prediction_test - Y_test))
    print("Accuracy on train set:", accuracy_train)
    print("Accuracy on test set:", accuracy_test)

    # 为了便于分析和检查，我们把得到的所有参数、超参数都存进一个字典返回出来：
    d = {"costs": costs,
         "Y_prediction_test": prediction_test,
         "Y_prediction_train": prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "accuracy_train": accuracy_train,
         "accuracy_test": accuracy_test
         }
    return d


d = logistic_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
                   print_cost=True)

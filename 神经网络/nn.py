class Network(object):

    def __init__(self, sizes):  # sizes: 每层神经元的个数,如[2, 3, 1]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]   # 当层有几个神经元就有几个biases
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]   # self.weights后面用的时候再转置


    def feedforward(self, a):   
        """ 每层输入a,输出a """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)    # 需要定义一个sigmoid函数
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        """mini-batch 随机梯度下降(stochastic gradient descent)
        "training_data" 是(x数据, y标签)元组的列表
        "epochs"是迭代多少个轮回
        "eta" 为学习率

        If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [     # mini_batches列表,每次训练取一个
                training_data[k:k+mini_batch_size] \
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # 训练,更新w,b
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)   # 测试准确率方法
            else:
                print "Epoch {0} complete".format(j)


   def update_mini_batch(self, mini_batch, eta):   # 为了更新w,b
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]   # 初始化w,b矩阵
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)   # 新方法,传入每张图片,返回w,b的更新量(一条记录模拟一次非线性回归方程)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]   # 整个mini_batch的对于w,b的更新量
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw    # 根据梯度下降方程,得到模型的w,b
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):    # BP算法,输入x,y,返回w,b的变化量
        # Backpropagation核心解决的问题: ∂C/∂w 和 ∂C/∂b 的计算, 针对cost函数C
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward 正向传递
        activation = x
        activations = [x]  # 存储每一层的最终输出
        zs = []  # 存储每一层的中间变量z值,即未经过激活函数
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass  反向传播
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])    # 针对最后输出层的损失函数,error损失函数 = 预测的误差 * 误差方程的导数 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  # 计算每一层的损失函数
            nabla_b[-l] = delta   # 更新每一层的b
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  # 更新每一层的w
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):   # 测试准确率方法
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]   # (预测结果,真实结果)的列表
        return sum(int(x == y) for (x, y) in test_results)  # 返回预测正确的个数

    def cost_derivative(self, output_activations, y):    # 求误差偏导的方程
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)




#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



# 使用方法:
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30轮, 10, 3.0, test_data)
# net.
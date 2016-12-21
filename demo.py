def demo_mnist_simple():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    import network
    net = network.Network([784, 30, 10])

    epochs = 30
    mini_batch_size = 10
    learning_rate = 3.0
    net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)

def demo_mnist_improved():
    # training data is list of tuples. The tuples first entry is the input data, an np.array of shape (784,1),
    # the second entry an np.array of shape (10,1), the vectorized version of the number
    # test data is built the same, except that the individual tuples second entry is only a single integer, not the
    # vectorized version like for the training data
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    import network_improved as network
    net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy = True)

def load_crosses(grey_transform=None, use_cancelled=False):
    import crosses_loader
    training_data, test_data = crosses_loader.load_data(grey_transform=grey_transform, use_cancelled=use_cancelled)
    return training_data, test_data

def demo_crosses(training_data, test_data):

    import network_improved as network
    data_size = training_data[0][0].shape[0]
    label_size = training_data[0][1].shape[0]
    #working configs:
    #  network: [data_size, 50,30,2], CrossEntropyCost, 60,20,0.01 best result after 14 epochs on evaluation: 6827/6859
    #  network: [data_size, 30,2], CrossEntropyCost, 60,20,0.01 best result after 14 epochs on evaluation: 6824/6859
    #  network: [data_size,2], CrossEntropyCost, 60,20,0.01 best result after 17 epochs on evaluation: 6832/6859
    #  network: [data_size,2], CrossEntropyCost, 60,20,0.1 FAST! best result after 22 epochs 6847/6859


    #  network: [data_size, 30,2], QuadraticCost, 60,20,0.01 best result after 33 epochs on evaluation: 6802/6859
    #  network: [data_size, 10,2], QuadraticCost, 60,20,0.01 best result after 30 epochs on evaluation: 6811/6859
    #  network: [data_size,2], QuadraticCost, 60,20,0.01 best result after 56 epochs on evaluation: 6815/6859

    net = network.Network([data_size, label_size], cost=network.CrossEntropyCost)
    net.SGD(training_data, 60, 20, 0.1, lmbda=0.1, evaluation_data=test_data, monitor_training_cost=True, monitor_training_accuracy=True, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)

def demo_crosses_simple(training_data, test_data, eta, cost, neuron):
    import network_simple as network
    data_size = training_data[0][0].shape[0]
    label_size = training_data[0][1].shape[0]
    net = network.Network(data_size, label_size, cost, neuron)
    # Working configs (numbers for test size 6859 without grey transform):
    # Best cost/neuron pairs: CrossEntropyCost+SigmoidNeuron >>> QuadraticCost+LinearNeuron > QuadraticCost+Sigmoid

    #demo_crosses_simple(training, test, 0.001, network.QuadraticCost, network.IdentityNeuron) # UNRELIABLE; peak 6769, slow, not learning with higher rate

    # demo.demo_crosses_simple(training, test, 0.0003, quad, linear) # UNRELIABLE; peak 6785; rate=0.0005 peak 6815 (about 50 epochs, SLOW!); not learning with rate >= 0.001!
    # demo.demo_crosses_simple(training, test, 0.00009, quad, linear) # (or 0.0001) to watch actual (slow) learning

    # demo.demo_crosses_simple(training, test, 0.01, quad, sigmoid) # peak 6786; rate=0.05: peak 6816; rate=0.08: peak 6826; not learning with rate >= 0.09!
    # demo.demo_crosses_simple(training, test, 0.1, cross, sigmoid) # peak 6847; FAST!; cost=inf if rate >=0.5 but working
    # demo.demo_crosses_simple(training, test, 0.1, cross, linear) # no working learning rate found (zero activated value doesnt work with logarithm in cost)

    net.SGD(training_data, 50, 50, eta, evaluation_data=test_data, print_errors=False)

def transform(grey): # takes normalized greyness values in range [0,1]
    return grey ** 0.5

def run_simple():
    training, test = load_crosses(grey_transform=None)
    import network_simple as network
    #demo_crosses_simple(training, test, 0.001, network.QuadraticCost, network.IdentityNeuron) # Working but not best possible results
    #demo_crosses_simple(training, test, 0.0005, network.QuadraticCost, network.LinearNeuron) # Unreliable, hugely depends on starting weights being good as activation derivative is exactly zero often
    demo_crosses_simple(training, test, 0.1, network.CrossEntropyCost, network.SigmoidNeuron) # Reliable and fast

if __name__ == "__main__":
    run_simple()
import numpy as np 
import pandas as pd 
import src.functions as fun
import time


def process_data(route, testAmount, scale,results,rem_axis = None):
    """
    Load, preprocess, and split dataset into training and test sets.

    Parameters
    ----------
    route : str
        Path to the CSV file containing the dataset.
    testAmount : int
        Number of samples to allocate to the test set.
    scale : float
        Normalization factor for input features (e.g., 255 for image pixel values).
    results : int
        Number of output classes.
    rem_axis : str or list, optional
        Column(s) to remove from the dataset before processing 
        (e.g., metadata columns). Default is None.

    Returns
    -------
    test_y : numpy.ndarray
        Test labels of shape (testAmount,).
    test_x : numpy.ndarray
        Test features of shape (num_features, testAmount), normalized by `scale`.
    train_y : numpy.ndarray
        Training labels of shape (num_train_samples,).
    train_x : numpy.ndarray
        Training features of shape (num_features, num_train_samples), normalized by `scale`.
    pixels : int
        Input feature dimension (number of features).
    train_trials : int
        Number of training samples.
    results : int
        Number of output classes (passed through unchanged).

    Notes
    -----
    - Assumes the first column of the dataset contains labels.
    - The dataset is shuffled before splitting.
    - Features are transposed so that each column corresponds to one sample.
    """
    data = pd.read_csv(route)
    if rem_axis is not None:
        data = data.drop(rem_axis,axis = 1)
    data = np.array(data)
    np.random.shuffle(data)

    test_data = data[0:testAmount].T
    test_y = test_data[0]
    test_x = test_data[1:]/scale

    train_data = data[testAmount:].T
    train_y = train_data[0]
    train_x = train_data[1:]/scale

    pixels, train_trials = train_x.shape
    
    return test_y,test_x,train_y,train_x, pixels,train_trials,results


def init_parameters(H, N, pixels, results, act = None):
    """
    Initialize network parameters (weights and biases) for a multilayer perceptron.

    Parameters
    ----------
    H : int
        Number of hidden layers.
    N : int
        Number of neurons per hidden layer.
    pixels : int
        Input feature dimension (number of input nodes).
    results : int
        Output dimension (number of classes).
    act : str, optional
        Initialization method for weights. Options:
        - "xavier" : Xavier/Glorot initialization (good for tanh/sigmoid activations).
        - "he" : He initialization (good for ReLU activations).
        - "uniform" : Uniform initialization in [-0.5, 0.5].
        Default is None.

    Returns
    -------
    params : list of [numpy.ndarray, numpy.ndarray]
        A list of parameter pairs for each layer:
        - params[i][0] : weight matrix of shape (n_out, n_in).
        - params[i][1] : bias vector of shape (n_out, 1).

    Raises
    ------
    ValueError
        If `act` is not one of {"xavier", "he", "uniform"}.

    Notes
    -----
    - The first layer uses `pixels` as input dimension and `N` neurons as output.
    - Hidden layers (if any) use `N` inputs and `N` outputs.
    - The final layer uses `N` inputs and `results` outputs.
    - Biases are always initialized to zeros (except in "uniform" where they are 
      initialized in [-0.5, 0.5]).
    """
    params = []
    if act == "xavier":
        n_in = pixels
        n_out = N
        limit = np.sqrt(6 / (n_in + n_out)) 
        params.append([
            np.random.uniform(-limit, limit, (n_out, n_in)), 
            np.zeros((n_out, 1))                             
        ])
        for _ in range(H - 1):
            n_in = N
            n_out = N
            limit = np.sqrt(6 / (n_in + n_out))
            params.append([
                np.random.uniform(-limit, limit, (n_out, n_in)),
                np.zeros((n_out, 1))                             
            ])
        n_in = N
        n_out = results
        limit = np.sqrt(6 / (n_in + n_out))
        params.append([
            np.random.uniform(-limit, limit, (n_out, n_in)),
            np.zeros((n_out, 1))                             
        ])
        return params
    elif act == "he":
        n_in = pixels
        n_out = N
        std_dev = np.sqrt(2 / n_in) 
        params.append([
            np.random.randn(n_out, n_in) * std_dev, 
            np.zeros((n_out, 1))                   
        ])
        for _ in range(H - 1):
            n_in = N
            n_out = N
            std_dev = np.sqrt(2 / n_in)
            params.append([
                np.random.randn(n_out, n_in) * std_dev,
                np.zeros((n_out, 1))                  
            ])
        n_in = N
        n_out = results
        std_dev = np.sqrt(2 / n_in) 
        params.append([
            np.random.randn(n_out, n_in) * std_dev, 
            np.zeros((n_out, 1))                  
        ])
        return params
    elif act == "uniform":
        params.append([
            np.random.rand(N, pixels) - 0.5,  
            np.random.rand(N, 1) - 0.5       
        ])
        for _ in range(H-1):
            params.append([
                np.random.rand(N, N) - 0.5,   
                np.random.rand(N, 1) - 0.5    
            ])
        params.append([
            np.random.rand(results, N) - 0.5,  
            np.random.rand(results, 1) - 0.5   
        ])
        return params
    else: 
        raise ValueError(f"unknown initialization {act}")

def forward_propogation(IN, params):
    """
    Perform forward propagation through the network.

    Parameters
    ----------
    IN : numpy.ndarray
        Input data (shape [input_dim, batch_size]).
    params : list of lists
        Network parameters (weights and biases per layer).

    Returns
    -------
    tuple
        - OUT: Output probabilities (shape [output_dim, batch_size]).
        - Zs: List of pre-activations for each layer.
        - As: List of activations (including input layer).
    """
    Zs, As = [], [IN] 
    
    for W, b in params[:-1]: 
        Z = np.dot(W, As[-1]) + b
        A = fun.ReLu(Z) 
        Zs.append(Z)
        As.append(A)
    
    W_out, b_out = params[-1]
    Z_out = np.dot(W_out, As[-1]) + b_out
    OUT = fun.softmax(Z_out)
    
    return OUT, Zs, As
        
def back_propogation(OUT, train_y, Z, A, params):
    """
    Compute gradients for all parameters using backpropagation.

    Parameters
    ----------
    OUT : numpy.ndarray
        Output probabilities from forward pass.
    train_y : numpy.ndarray
        Ground truth labels (shape [batch_size,]).
    Z : list of numpy.ndarray
        Pre-activations from forward pass.
    A : list of numpy.ndarray
        Activations from forward pass (including input).
    params : list of lists
        Current network parameters.

    Returns
    -------
    list of lists
        Gradients for each layer [dW, db], ordered from output to input.
    """
    Y = fun.one_hot(train_y)
    m = train_y.size

    gradient = []

    dz = OUT - Y
    dw = (1/m) * np.dot(dz, A[-2].T) 
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    gradient.append([dw, db])

    for i in range(len(params)-2, -1, -1):
        W_next = params[i+1][0]             
        dz = np.dot(W_next.T, dz) * fun.derv_ReLu(Z[i])
        dw = (1/m) * np.dot(dz, A[i].T)      
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        gradient.insert(0, [dw, db])        

    return gradient 

def make_lr_schedule(schedule, base_lr, iters, step_size=100, decay_rate=0.99):
    """
    Create a learning rate schedule function.

    Parameters
    ----------
    schedule : str
        The type of learning rate schedule to use. Options:
        - "fixed" : constant learning rate.
        - "step" : halve the learning rate every `step_size` epochs.
        - "exp" : exponential decay with factor `decay_rate`.
        - "cosine" : cosine annealing from base_lr → 0 over `iters` epochs.
    base_lr : float
        Base learning rate (initial value).
    iters : int
        Total number of training epochs (used for cosine schedule).
    step_size : int, optional
        Number of epochs between learning rate reductions in "step" schedule.
        Default is 100.
    decay_rate : float, optional
        Exponential decay factor for "exp" schedule. Default is 0.99.

    Returns
    -------
    function
        A function `f(epoch)` that returns the learning rate for a given epoch.

    Raises
    ------
    ValueError
        If `schedule` is not one of {"fixed", "step", "exp", "cosine"}.
    """
    if schedule == "fixed":
        return lambda epoch: base_lr
    elif schedule == "step":
        return lambda epoch: base_lr * (0.5 ** (epoch // step_size))
    elif schedule == "exp":
        return lambda epoch: base_lr * (decay_rate ** epoch)
    elif schedule == "cosine":
        return lambda epoch: base_lr * (0.5 * (1 + np.cos(np.pi * epoch / iters)))
    else:
        raise ValueError(f"Unknown schedule {schedule}")

def gradient_descent(H, N, X, Y, pixels, results, iterations, base_lr = 0.1, act = "uniform", schedule="fixed", step_size=100, decay_rate=0.99, batch_size = None):
    """
    Train a multi-layer perceptron (MLP) using gradient descent with optional
    mini-batching and learning rate scheduling.

    Parameters
    ----------
    H : int
        Number of hidden layers in the network.
    N : int
        Number of neurons per hidden layer.
    X : numpy.ndarray
        Training data of shape (input_dim, num_samples).
    Y : numpy.ndarray
        Training labels of shape (num_samples,).
    pixels : int
        Input dimension (number of features).
    results : int
        Output dimension (number of classes).
    iterations : int
        Number of training epochs.
    base_lr : float, optional
        Base learning rate. Default is 0.1.
    act : str, optional
        Initialization method for parameters. Options include
        {"uniform", "xavier", "he"}. Default is "uniform".
    schedule : str, optional
        Learning rate schedule to use. Options include
        {"fixed", "step", "exp", "cosine"}. Default is "fixed".
    step_size : int, optional
        Step size for "step" schedule (number of epochs before halving LR).
        Default is 100.
    decay_rate : float, optional
        Decay factor for "exp" schedule. Default is 0.99.
    batch_size : int or None, optional
        Number of samples per mini-batch. If None, use full-batch gradient descent.

    Returns
    -------
    params : list
        Trained network parameters for each layer.
    accuracy : list of float
        Training accuracy sampled every 100 epochs.
    hundred_duration : list of float
        Duration (in seconds) of epochs where accuracy was logged (every 100 epochs).
    epoch_time : list of float
        Duration (in seconds) of every epoch.

    Notes
    -----
    - The learning rate schedule is applied per epoch, not per batch.
    - Accuracy is computed on the full training set every 100 epochs.
    - For smaller batch sizes, more updates are performed per epoch,
      which can affect both convergence and runtime.
    """

    params = init_parameters(H, N, pixels, results, act)
    accuracy = []
    epoch_time = []
    hundred_duration = []

    lr_schedule = make_lr_schedule(schedule, base_lr, iterations, step_size, decay_rate)

    num_samples = X.shape[1]
    if batch_size is None:
        batch_size = num_samples

    for epoch in range(iterations):
        start_time = time.time()

        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[indices]

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[start:end]

            OUT, Z, A = forward_propogation(X_batch, params)
            gradient = back_propogation(OUT, Y_batch, Z, A, params)

            alpha = lr_schedule(epoch)
            for j in range(len(params)):
                params[j][0] -= alpha * gradient[j][0]
                params[j][1] -= alpha * gradient[j][1]

        epoch_duration = time.time() - start_time
        epoch_time.append(epoch_duration)

        if epoch % 100 == 0:
            OUT, _, _ = forward_propogation(X, params)
            acc = fun.get_accuracy(OUT, Y)
            print(f"Iteration {epoch}, Accuracy={acc:.4f}")
            accuracy.append(acc)
            hundred_duration.append(epoch_duration)

    return params, accuracy, hundred_duration, epoch_time

def train_and_evaluate(H, N, train_x, train_y, test_x, test_y, pixels, results, iters, alpha, act = "uniform", schedule = "fixed", batch_size = None):
    start = time.time()
    params, training_accuracy, hundred_duration, epoch_time = gradient_descent(H, N, train_x, train_y, pixels, results, iters, alpha, act, schedule, batch_size=batch_size)
    total_duration = time.time() - start
    OUT, _, _ = forward_propogation(test_x, params)
    preds = np.argmax(OUT, axis=0)
    test_acc = np.mean(preds == test_y)
    
    return test_acc, total_duration, training_accuracy, hundred_duration, epoch_time

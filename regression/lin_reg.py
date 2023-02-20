import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import ceil
from scipy import linalg


def load_data(data_path):
    data = pd.read_csv(data_path)
    features = np.array(data.iloc[:, :-1])
    labels = np.array(data.iloc[:, -1])

    return features, labels


def random_test_train_split(features, labels, test_percentage=20):
    p = np.random.permutation(len(features))
    features = features[p]
    labels = labels[p]

    test_data_size = int(test_percentage / 100 * len(features))

    train_features = features[:-test_data_size,:]

    train_labels = labels[:-test_data_size]

    test_features = features[-test_data_size:, :]
    test_labels = labels[-test_data_size:]

    return train_features, train_labels, test_features, test_labels


def standardize_features(train_features, test_features):

    train_mean = np.mean(train_features, axis=0)
    train_std = np.std(train_features, axis=0)

    train_features = (train_features - train_mean) / train_std
    test_features = (test_features - train_mean) / train_std

    return train_features, test_features


def get_2nd_degree_features(features):
    second_degree_features = features.copy()

    number_of_examples = features.shape[0]
    number_of_features = features.shape[1]

    number_of_columns_to_add = int((number_of_features + 1) * number_of_features / 2)
    
    columns_to_add = np.zeros((number_of_examples, number_of_columns_to_add))
    
    cnt = 0
    for i in range(features.shape[1]):
        for j in range(features.shape[1]):
            if i <= j:
                columns_to_add[:, cnt] = features[:, i] * features[:, j]
                cnt += 1
    
    second_degree_features = np.append(second_degree_features, columns_to_add, axis=1)

    return second_degree_features


def mse(predictions, labels):
    '''
    MSE cost function
    
    Input:
        - predictions (np.array of floats): predictions from model
        - labels (np.array of floats): true values

    Returns:
        - cost: MSE(predictions, labels): 1/m * sum((prediction-label)**2)
    '''
    cost = np.average(np.square(predictions - labels))
    return cost

def get_mse_gradients(features, predictions, labels):
    '''
    Returns gradient for MSE cost function

    Input:
        - features (np.array)
        - predictions (np.array)
        - labels (np.array)
    
    Returns:
        - gradients: 2 / m * sum ((prediction - label) * feature)
    '''
    m = features.shape[0]
    gradients = 2/m * (np.dot((predictions - labels), features))
    return gradients

def mse_gradient_descent(features, labels, iterations=10000, learning_rate=0.1):
    '''
    Gradient descent for linear regression
    Calculates optimal weights and bias using iterative batch grad desc

    Inputs:
        - features (np.array)
        - labels (np.array)
        - iterations (int): num of iterations for grad desc
        - learning_rate (float): learning rate used in grad desc
    
    Returns:
        - weights (np.array): optimal weights calculated by grad desc
    '''
    m = features.shape[0]
    n = features.shape[1]

    features = np.append(np.ones((m, 1)), features, axis=1)
    weights = np.zeros(n+1)

    for i in range(iterations):
        predictions = np.dot(weights, np.transpose(features))
        gradients = get_mse_gradients(features, predictions, labels)
        weights -= learning_rate * gradients
    
    return weights


def mse_analytic_solution(features, labels):
    '''
    Analytic solution for linear regression with mse

    Input:
        - features (np.array)
        - labels (np.array)

    Returns:
        - weights (np.array): optimal weights calculated by analytic solution
    '''

    m = features.shape[0]
    features = np.append(np.ones((m, 1)), features, axis=1)
    weights = np.dot(np.linalg.inv(np.dot(features.T, features)), np.dot(labels, features))

    return weights


def predict(features, weights):
    '''
    Get predictions for features using weights

    Input:
        - features (np.array)
        - weights (np.array)

    Returns:
        - predictions: weights.T @ features
    '''
    m = features.shape[0]
    features = np.append(np.ones((m, 1)), features, axis=1)
    predictions = np.dot(weights, np.transpose(features))

    return predictions


def mse_ridge_cost_function(prediction, labels, weights, lmbd):
    '''
    Calculating cost for mse with ridge regularization

    Inputs:
        - prediction (np.array)
        - labels (np.array)
        - weights (np.array)
        - lmbd (float): multiplier for regularization parameter
    
    Returns:
        - cost (float)

    '''
    m = len(prediction)

    # Bias should not be reguralized
    bias = weights[0]
    weights[0] = 0

    cost = (1/m * (np.sum(np.square(prediction - labels))))**0.5 + lmbd * np.sum(np.square(weights))

    weights[0] = bias
    return cost


def get_mse_ridge_gradients(features, predictions, labels, weights, lmbd):
    '''
    Calculating gradients for mse with ridge regularization

    Inputs:
        - prediction (np.array)
        - labels (np.array)
        - weights (np.array)
        - lmbd (float): multiplier for regularization parameter
    
    Returns:
        - gradients (np.array)

    '''
    m = features.shape[0]

    # Bias should not be reguralized
    bias = weights[0]
    weights[0] = 0

    gradients = 2/m * (np.dot((predictions - labels), features)) + 2/m * lmbd * weights

    weights[0] = bias
    return gradients


def mse_ridge_gradient_descent(features, labels, iterations=10000, learning_rate=0.1, lmbd=0.1):
    '''
    Gradient descent for linear regression with mse and ridge regularization
    Calculates optimal weights and bias using iterative batch grad desc

    Inputs:
        - features (np.array)
        - labels (np.array)
        - iterations (int): num of iterations for grad desc
        - learning_rate (float): learning rate used in grad desc
        - lmbd (float): multiplier for ridge regularization
    
    Returns:
        - weights (np.array): optimal weights calculated by grad desc
    '''
    
    m = features.shape[0]
    
    features = np.append(np.ones((m, 1)), features, axis=1)
    n = features.shape[1]
    weights = np.zeros(n)

    for i in range(iterations):
        predictions = np.dot(weights, np.transpose(features))
        gradients = get_mse_ridge_gradients(features, predictions, labels, weights, lmbd)
        weights -= learning_rate * gradients
    
    return weights


def mse_ridge_analytic_solution(features, labels, lmbd):
    '''
    Analytic solution for linear regression with mse and ridge regularization

    Input:
        - features (np.array)
        - labels (np.array)

    Returns:
        - weights (np.array): optimal weights calculated by analytic solution
    '''

    # Analytic solution of ridge regression cannot fit bias
    # Bias is calculated as mean of labels
    # Labels are then centralized so other weights can fit on data with mean=0
    # because bias already fit the mean
    bias = np.average(labels)
    labels = labels - labels.mean()
    n = features.shape[1]

    identity_mat = lmbd * np.eye(n,n)
    A = np.dot(features.T, features) + identity_mat
    Xy = np.dot(labels, features)

    # weights = np.dot(np.linalg.inv(A), Xy)
    weights = linalg.solve(A, Xy, assume_a='pos', overwrite_a=True)

    weights = np.append(bias, weights)

    return weights


def ridge_cross_validation(features, labels, search_space, algorithm='mse_ridge_lin_reg', k=5):
    '''
    Cross validation implementation ofr choosing lambda multiplier for ridge regularization

    Inputs:
        - features (np.array)
        - labels (np.array)
        - search_space (tuple) tuple with values (start, end, step)
            this is the space which will be searched for best lambda parameter
        - algorithm (string): algorithm which will be used for cross validating
            possible values: 'mse_lin_reg', 'ridge_lin_reg' 
    
    Returns:
        - lambdas (np.array): all values of lambda tried
        - means (np.array): mean cv error for all tried lambdas
        - stds (np.array): std values for all tried lambdas
    '''

    if algorithm == 'mse_lin_reg':
        get_weights = mse_analytic_solution
    elif algorithm == 'mse_ridge_lin_reg':
        get_weights = mse_ridge_analytic_solution

    else:
        raise Exception("Only valid values for algorithm are 'mse_lin_reg' and 'mse_ridge_lin_reg'")

    lambdas = np.linspace(*search_space)
    fold_size = ceil(len(features) / k)
    costs = [[] for _ in range(len(lambdas))]

    for index in range(k):

        test_mask = np.full(features.shape[0], False)

        if index == k-1:
            test_mask[range(index*fold_size, features.shape[0])] = True
        else:
            test_mask[range(index*fold_size, (index+1)*fold_size)] = True

        train_mask = np.logical_not(test_mask)

        train_features = features[train_mask]
        train_labels = labels[train_mask]
        test_features = features[test_mask]
        test_labels = labels[test_mask]
        
        for i, lmbd in enumerate(lambdas):

            
            weights = get_weights(train_features, train_labels, lmbd=lmbd)
            predictions = predict(test_features, weights)
            cost = mse(predictions, test_labels)
            costs[i].append(cost)
    
    means = []
    stds = []

    for i, cost in enumerate(costs):
        average_cost = np.mean(cost)
        std = np.std(cost)
        means.append(average_cost)
        stds.append(std)
    
    return np.array(lambdas), np.array(means), np.array(stds)


def main():
    np.random.seed(1302)
    data_path = "regression/data.csv"
    features, labels = load_data(data_path)
    
    # Splitting dataset to train and test set
    train_features, train_labels, test_features, test_labels = random_test_train_split(features, labels)
    # Making second degree features
    train_features, test_features = get_2nd_degree_features(train_features), get_2nd_degree_features(test_features)
    # Standardizing train and test features
    train_features, test_features = standardize_features(train_features, test_features)

    #weights = mse_ridge_analytic_solution(train_features, train_labels, lmbd = 53)
    
    lambdas, means, stds = ridge_cross_validation(train_features, train_labels, (40, 60, 5))
    
    plt.plot(lambdas, means)
    plt.fill_between(lambdas, means - stds,
                 means + stds, alpha=0.2,
                 color="darkorange")
    plt.xlabel("lambdas")
    plt.ylabel("Mean square error")
    plt.show()


if __name__ == '__main__':
    main()

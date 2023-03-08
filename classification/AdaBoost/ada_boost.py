import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from stamp import Stamp
import os

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


def get_error(predictions, labels):

    m = predictions.shape[0]
    differences = np.sum(predictions != labels)
    ratio = differences / m
    
    return ratio


def predict_single(trees, alphas, example):
    '''
    This function predicts class for single sample using all trained decision trees
    Classification is done through voting
    Since class can be either 1 or -1 we take the average of all classifications
    If this average is greater than 0 it means there were more 1s than -1s
    '''
    positive_predictions = 0
    for tree, alpha in zip(trees, alphas):
        prediction = tree.predict(example.reshape(1, -1))
        positive_predictions += alpha * prediction

    positive_predictions = positive_predictions[0]
    
    if positive_predictions > 0:
        return 1
    else:
        return -1


def predict_batch(trees, alphas, features):
    '''
    This function predicts classes for all examples in features parameter
    It serves as a wrapper around predict_single function
    '''

    predictions = []

    for example in features:
        prediction = predict_single(trees, alphas, example)
        predictions.append(prediction)
    
    predictions = np.array(predictions)
    return predictions


def plot_examples(trees, alphas, train_features, train_labels, weights, num_of_dots=50):
    '''
    Plots examples on a 2D grid where areas are colored based by their class
    Examples with bigger weight appear bigger on the plot
    '''

    epsilon_x = (train_features[:,0].max() - train_features[:,0].min()) / 20
    epsilon_y = (train_features[:,1].max() - train_features[:,1].min()) / 20
    xmin, xmax, ymin, ymax = train_features[:,0].max() + epsilon_x, train_features[:,0].min() - epsilon_x, train_features[:,1].max() + epsilon_y, train_features[:,1].min() - epsilon_y
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, num=num_of_dots, endpoint=True), 
        np.linspace(ymin, ymax, num=num_of_dots, endpoint=True))
    X = np.c_[xx.ravel(), yy.ravel()]
    
    Z = predict_batch(trees, alphas, X)

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')

    m = train_features.shape[0]
    features_positive = train_features[train_labels == 1]
    weights_positive = weights[train_labels == 1] * m * 20
    features_negative = train_features[train_labels == -1]
    weights_negative = weights[train_labels == -1] * m * 20

    plt.scatter(features_positive[:,0], features_positive[:,1], c='r', s=weights_positive)
    plt.scatter(features_negative[:,0], features_negative[:,1], c='b', s=weights_negative)

    plt.show()


def ada_boost(train_features, train_labels, plot_progress=True, iterations=5):

    m = train_features.shape[0]
    weights_of_examples = 1/m * np.ones(m)
    trees = []
    alphas = []

    for iteration in range(iterations):

        tree = Stamp()
        tree.fit(train_features, train_labels, weights_of_examples)
        

        train_predictions = tree.predict(train_features)

        errors = train_predictions != train_labels
        weighted_errors = weights_of_examples * errors
        weighted_error = weighted_errors.sum()

           
        alpha = np.log((1 - weighted_error) / weighted_error)
        weights_of_examples *= np.exp(alpha * errors)
        weights_of_examples /= sum(weights_of_examples)

        trees.append(tree)
        alphas.append(alpha)


        if plot_progress:
            plot_examples(trees, alphas, train_features, train_labels, weights_of_examples)
    
    plot_examples(trees, alphas, train_features, train_labels, weights_of_examples)
    
    return trees, alphas




def main():

    #path = os.path.join('classification', 'AdaBoost', '2d_spiral_dataset_-1_1.csv')
    #path = os.path.join('classification', 'AdaBoost', '2d_straight_dataset_-1_1.csv')
    path = os.path.join('classification', 'AdaBoost', '2d_square_dataset_-1_1.csv')
    features, labels = load_data(path)
    train_features, train_labels, test_features, test_labels = random_test_train_split(features, labels, test_percentage=10)
    train_features, test_features = standardize_features(train_features, test_features)

    trees, alphas = ada_boost(train_features, train_labels, iterations=7)
    test_predictions = predict_batch(trees, alphas, test_features)
    test_error = get_error(test_predictions, test_labels)
    print(f'test error = {test_error}')


if __name__ == '__main__':
    main()
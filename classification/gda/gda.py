import pandas as pd
import numpy as np

def load_data(data_path):
    data = pd.read_csv(data_path)
    features = np.array(data.iloc[:, :-1])
    labels = np.array(data.iloc[:, -1])

    return features, labels


def get_predictions_GDA(features, means_var_apriori_list):

    probabilities = []
    for mean, var, a_priori_by_class in means_var_apriori_list:
        diff = features - mean
        p =  1 / (np.sqrt(np.linalg.det(var))) * np.exp(-1/2 * diff.dot(np.linalg.inv(var)).dot(diff.T)) * a_priori_by_class
        probabilities.append(p.diagonal())
    
    probabilities = np.array(probabilities)
    predictions = np.argmax(probabilities, axis=0)
    
    return predictions


def get_means_var_apriori_list(features_by_class):

    m = sum([len(class_features) for class_features in features_by_class])
    means_var_apriori_list = []

    for i, feature_class in enumerate(features_by_class):
        feature = np.array(feature_class)
        mean = np.mean(feature, axis=0)
        diff = feature - mean
        var = (diff.T).dot(diff) / len(diff)
        a_priori_by_class = len(feature) / m
        means_var_apriori_list.append((mean, var, a_priori_by_class))
    
    return means_var_apriori_list


def get_features_by_class(features, labels):

    k = len(set(labels))
    features_by_class = [[] for _ in range(k)]

    for feature, label in zip(features, labels):
      features_by_class[label].append(feature)
    
    return features_by_class


def get_error(predictions, labels):

    m = predictions.shape[0]
    differences = np.sum(predictions == labels)
    ratio = differences / m
    
    return ratio


def main_GDA():
    data_path = 'classification/gda/multiclass_data.csv'

    features, labels = load_data(data_path)

    # divide features in separate lists by class
    # example [(a, 0), (b,0), (c,1), (d,2), (e,2)] ----->  [ [a, b], [c], [d, e] ]
    features_by_class = get_features_by_class(features, labels)
    
    # get mean, variance and a priori probability for each class
    # example [ [a, b], [c], [d, e] ] ----->  [ [mean(a, b), variance(a, b), a_priori(a, b)], [mean(c), variance(c), a_priori(c)], [mean(d, e), variance(d, e), a_priori(d, e)]]
    means_var_apriori_list = get_means_var_apriori_list(features_by_class)

    predictions = get_predictions_GDA(features, means_var_apriori_list)
    print(get_error(predictions, labels))

main_GDA()

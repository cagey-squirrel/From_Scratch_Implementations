import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

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


def get_predictions_softmax(data, weights):
    m = data.shape[0]
    data = np.append(np.ones((m, 1)), data, axis=1)

    pred_values = np.dot(data, weights.T)
    predictions = np.argmax(pred_values, axis=1)

    return predictions



def get_error(predictions, labels):

    m = predictions.shape[0]
    differences = np.sum(predictions == labels)
    ratio = differences / m
    
    return ratio


def calculate_likelihood_vectorized(data, weights, labels):
    m = data.shape[0]
    data = np.append(np.ones((m, 1)), data, axis=1)

    all_theta_dot_x = np.dot(weights, data.T)
    
    exp_all_theta_dot_x = np.exp(all_theta_dot_x)
    sum_all_theta_dot_x = np.sum(exp_all_theta_dot_x, axis=0)
    log_all_theta_dot_x = np.log(sum_all_theta_dot_x)

    likelihood = 0

    for i in range(m):
      label = labels[i]
      data_point = data[i]

      thetaL_dot_x = all_theta_dot_x[label][i]
      likelihood += thetaL_dot_x - log_all_theta_dot_x[i]

    return likelihood / m
    

def softmax_regression_single_update_vectorized(data, weights, labels, alpha):
    m = data.shape[0]
    data = np.append(np.ones((m, 1)), data, axis=1)

    
    n = data.shape[1]
    k = weights.shape[0]

    gradient_sum = np.zeros(weights.shape)

    all_theta_dot_x = np.dot(weights, data.T)

    exp_all_theta_dot_x = np.exp(all_theta_dot_x)
    sum_all_theta_dot_x = np.sum(exp_all_theta_dot_x, axis=0)

    for i in range(m):
      label = labels[i]
      data_point = data[i]

      truth_vector = np.array(range(k)) == label
      truth_vector = np.array(truth_vector)

      sum_all_theta_dot_xi = sum_all_theta_dot_x[i]
      e_to_thetaL_dot_x = exp_all_theta_dot_x[:,i]
      exp_ratio = e_to_thetaL_dot_x / sum_all_theta_dot_xi

      single_gradient = (truth_vector - exp_ratio)
      single_gradient = [ [sg_val]*n for sg_val in single_gradient ]
      single_gradient = np.array(single_gradient)
      single_gradient *= data_point

      gradient_sum += single_gradient
    
    weights += alpha * gradient_sum
    # reseting final weight row to zeros
    weights[k-1] = np.zeros(n)
      

def main_softmax():
  np.random.seed(1302)
  data_path = 'classification/softmax//multiclass_data.csv'
  examples_num = 3500

  features, labels = load_data(data_path)

  train_features, train_labels, test_features, test_labels = random_test_train_split(features, labels)
  train_features, test_features = standardize_features(train_features, test_features)
  
  m = train_features.shape[0]
  n = train_features.shape[1]
  k = len(set(labels))

  batch_sizes = [32, 64]
  alphas = [0.2, 0.1]

  fig, axis = plt.subplots(len(alphas), len(batch_sizes), figsize=(40, len(alphas)*10))

  for index_alpha, alfa in enumerate(alphas):
      for index_bs, batch_size in enumerate(batch_sizes):
          weights = np.zeros((k, n + 1))
          likelihoods = []
          vlikelihoods = []
          errors = []
          valid_error = []
          number_of_batches = int(math.ceil(m / batch_size))
          
          iter_num = int(np.ceil(examples_num / batch_size))
          for i in range(iter_num):

              start_index = (i % number_of_batches) * batch_size
              end_index = min(m, start_index + batch_size)
              batch_features = train_features[start_index:end_index]
              batch_labels = train_labels[start_index:end_index]

              likelihood = calculate_likelihood_vectorized(batch_features, weights, batch_labels)
              softmax_regression_single_update_vectorized(batch_features, weights, batch_labels, alfa)
              likelihoods.append(likelihood)

              likelihood = calculate_likelihood_vectorized(test_features, weights, test_labels)
              vlikelihoods.append(likelihood)
              predictions = get_predictions_softmax(batch_features, weights)
              error = get_error(predictions, batch_labels)
              errors.append(error)
              predictions = get_predictions_softmax(test_features, weights)
              error = get_error(predictions, test_labels)
              valid_error.append(error)

              #shuffle
              if start_index + batch_size > m:
                  p = np.random.permutation(len(train_features))
                  train_features = train_features[p]
                  train_labels = train_labels[p]

          axis[index_alpha][index_bs].plot([i*batch_size for i in range(iter_num)], likelihoods)
          axis[index_alpha][index_bs].title.set_text(f"Train set a={alfa}, batch_size={batch_size}")
          axis[index_alpha][index_bs].set_xlabel("Number of examples trained on")
          axis[index_alpha][index_bs].set_ylabel("Likelihood") 

  plt.show()

if __name__ == '__main__':
    main_softmax()


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import math
import sklearn
from cvxopt.solvers import qp
from cvxopt import matrix
import cvxopt
from time import time
import matplotlib.patches as mpatches
from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel

np.set_printoptions(suppress=True)

def load_data(data_path):
    data = pd.read_csv(data_path)
    features = np.array(data.iloc[:, :-1])
    labels = np.array(data.iloc[:, -1])

    return features, labels
  

def standardize_training(features):

    train_mean = np.mean(features, axis=0)
    train_std = np.std(features, axis=0)

    features = (features - train_mean) / train_std

    return features, train_mean, train_std


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


def hinge_loss(features, labels, w, b, C=1):
    regularization = np.dot(w, w.T)
    functional_margins = labels * (np.dot(features, w.T) + b)
    
    wrong_margin_indexes = functional_margins < 1
    loss = np.sum(1 - functional_margins[wrong_margin_indexes])

    return regularization[0][0] + loss
  

def predict_linear_SVM(X, svm_params):
    support_vectors, w, b, sigma = svm_params

    predictions = np.sign(np.dot(X, w.T) + b)
    return predictions


def predict_kernel_SVM(X, svm_params, kernel='gaus'):

    sv_features, alphas_mul_labels, b, sigma = svm_params
    #values = np.array([alphas_mul_labels[i] * gaus_kernel(X, sv, sigma) for i, sv in enumerate(sv_features)])
    m = X.shape[0]
    svm_m = sv_features.shape[0]

    if kernel == 'rbf':
      values = vectorized_gaus_kernel(X, sv_features, sigma)
    elif kernel == 'poly':
      values = vectorized_poly_kernel(X, sv_features)
    elif kernel == 'linear':
      values = vectorized_linear_kernel(X, sv_features)
    else:
      raise Exception("Kernel must be 'rbf' or poly' or 'linear'")

    # prediction = sum(alpha_i * label_y * value_i)
    alphas_mul_labels_mul_values = alphas_mul_labels * values

    sum_values = np.sum(alphas_mul_labels_mul_values, axis=0)
    predictions = np.sign(sum_values+b)

    return predictions


def get_error(predictions, labels):
    labels = labels.flatten().astype('int')
    predictions = predictions.flatten().astype('int')

    return np.sum(predictions != labels) / predictions.shape[0]


def plot_decision_boundary(svm_params, features, labels, hinge_losses, kernel='linear', num_of_dots=100, title=''):

    xmin, xmax, ymin, ymax = features[:,0].max(), features[:,0].min(), features[:,1].max(), features[:,1].min()
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, num=num_of_dots, endpoint=True), 
        np.linspace(ymin, ymax, num=num_of_dots, endpoint=True))
    X = np.c_[xx.ravel(), yy.ravel()]
    
    if kernel == 'truelinear':
      Z = predict_linear_SVM(X, svm_params)
    else:
      Z = predict_kernel_SVM(X, svm_params, kernel=kernel)

    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')

    labels = labels.flatten()
    features_positive = features[labels == 1]
    features_negative = features[labels == -1]

    support_vectors = svm_params[0]
    coords = support_vectors
    for i, txt in enumerate(hinge_losses):
      plt.annotate(str(txt)[:5], (coords[i,0]+0.05, coords[i,1]+0.05))
          
    plt.scatter(features_positive[:,0], features_positive[:,1], c='r')
    plt.scatter(features_negative[:,0], features_negative[:,1], c='b')
    plt.scatter(support_vectors[:,0], support_vectors[:,1], s=100, marker='p', facecolors='none', edgecolor='green', linewidth=2)

    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()


def train_primal_soft_SVM(features, labels, C, alpha=0.001, num_iter=1000):
    m, n = features.shape

    w = np.zeros((1, n))
    b = 0
    functional_margins = None

    for iter in range(num_iter):

        dw = 0
        db = 0

        functional_margins = (labels * (np.dot(features, w.T) + b)).flatten()
        bad_margin_indexes = functional_margins < 1
        dw = -C * np.sum(labels[bad_margin_indexes] * features[bad_margin_indexes,:], axis=0)
        db = np.sum(-labels[bad_margin_indexes])
 
        w -= alpha * (dw + w)
        b -= alpha * db


    support_vectors_indexes = (functional_margins <= 1).flatten()
    support_vectors = features[support_vectors_indexes]
    support_vectors_margins = functional_margins[support_vectors_indexes].flatten()
    hinge_losses = 1 - support_vectors_margins
            
    return w, b, support_vectors, hinge_losses


def vectorized_gaus_kernel(X, Y, sigma):
    n = X.shape[0]
    m = Y.shape[0]

    # Repeats X matrix m times, making m x m x n  shaped matrix
    repeated_X = np.tile(X, (m, 1, 1))

    if len(Y.shape) < 2:
      Y = Y[None, :]
    Y = Y[:, None , :]

    diff = repeated_X - Y
    norm = np.sum(diff**2, axis=2)
    gaus = np.exp(- sigma * norm)

    return gaus


def vectorized_poly_kernel(a, b, degree=3, constant=0):
    dot_prod = np.dot(b, a.T)
    dot_prod += constant
    dot_prod = dot_prod**degree
    return dot_prod


def vectorized_linear_kernel(a, b):
    dot_prod = np.dot(b, a.T)
    return dot_prod


def calculate_bias(sv_alphas, sv_labels, sv_features, C, kernel, sigma):
  '''
  Calculates bias for svm
  b is calculated by finding one support vector which has a margin of one
  (that support vector has an alpha value between 0 and C)
  Margin is calculated as y_i * (sum(alpha_k * y_k * K(x_k, x_i)) + b) = 1
  hence b = y_i * 1 - sum, where y_i * 1 is of course equal to just y_i
  '''
  margins_equal_one_indexes = np.logical_and(C/100000 < sv_alphas, (sv_alphas < (C-C/100000))).flatten()
  first_support_vector = margins_equal_one_indexes.argmax()

  # Finding one support vector with margin = 1
  fsv_label = sv_labels[first_support_vector]
  fsv_features = sv_features[first_support_vector]

  if kernel == 'rbf':
    fsv_features = fsv_features[None,:]
    K = vectorized_gaus_kernel(sv_features, fsv_features, sigma)
    K = K.T
  elif kernel == 'poly':
    K = vectorized_poly_kernel(sv_features, fsv_features)[:, None]
  elif kernel == 'linear':
    K = vectorized_linear_kernel(sv_features, fsv_features)[:, None]
    
  prod = sv_labels * sv_alphas * K
  suma = np.sum(prod)
  b = fsv_label - suma

  return b


def calculate_hinge_losses(sv_alphas, sv_labels, sv_features, kernel, sigma, b):
  '''
  Calculates hinge losses for support vectors
  '''
  if kernel == 'rbf':
    K = vectorized_gaus_kernel(sv_features, sv_features, sigma)
    K = K.T
  elif kernel == 'poly':
    K = vectorized_poly_kernel(sv_features, sv_features)[:, None]
  elif kernel == 'linear':
    K = vectorized_linear_kernel(sv_features, sv_features)[:, None]

  prod = sv_labels * sv_alphas * K
  suma = np.sum(prod, axis = 1) + b
  suma = suma[:, None]
  margins = sv_labels * suma
  psiz = 1 - margins
  psiz[psiz < 0] = 0

  hinge_losses = psiz.flatten()

  return hinge_losses

def train_dual_soft_SVM(features, labels, C, sigma, kernel='linear'):

      m = features.shape[0]

      if kernel == 'linear':
        P = np.dot(features, features.T)
      elif kernel == 'rbf':
        P = vectorized_gaus_kernel(features, features, sigma)
      elif kernel == 'poly':
        P = vectorized_poly_kernel(features, features)

      y_row = np.tile(labels.flatten(), (m, 1))
      y_col = np.tile(labels, (m))

      P *= y_row * y_col
      q = -1* np.ones((m,1))
      g1 = -1 * np.eye(m)
      g2 = np.eye(m)
      G = np.concatenate((g1, g2), axis=0)
      h1 = np.zeros((m, 1))
      h2 = C * np.ones((m, 1))
      h = np.concatenate((h1, h2), axis=0)
      A = labels.reshape((1, -1)).astype('float')
      b = np.zeros(1)

      P = matrix(P)
      q = matrix(q)
      G = matrix(G)
      h = matrix(h)
      A = matrix(A)
      b = matrix(b)

      cvxopt.solvers.options['show_progress'] = False
      results = qp(P, q, G, h, A, b)
      alphas = np.array(results['x'])

      w = np.sum(labels * alphas * features, axis=0)
      support_vector_indexes = (alphas > (C/100000)).flatten()
      sv_alphas = alphas[support_vector_indexes]
      sv_labels = labels[support_vector_indexes]
      alphas_mul_labels = sv_alphas * sv_labels
      sv_features = features[support_vector_indexes]

      b = calculate_bias(sv_alphas, sv_labels, sv_features, C, kernel, sigma)
      hinge_losses = calculate_hinge_losses(sv_alphas, sv_labels, sv_features, kernel, sigma, b)
      
      return alphas_mul_labels, sv_features, b, w, hinge_losses

      

    
def main_linear_SVM():
    np.random.seed(3270)
    
    features, labels = load_data("classification/svm/svmData.csv")
    labels = labels[:, None]
    train_features, train_labels, test_features, test_labels = random_test_train_split(features, labels, 50)
    
    train_features, mean_f, std_f = standardize_training(train_features)
    test_features = (test_features - mean_f) / std_f

    C = 100
    w, b, support_vectors, indexes_and_hinge = train_primal_soft_SVM(train_features, train_labels, C)

    svm_params = support_vectors, w, b, None
    plot_decision_boundary(svm_params, train_features, train_labels, indexes_and_hinge, kernel='truelinear')


    predictions = predict_linear_SVM(test_features, svm_params)
    error = get_error(predictions, test_labels)
    print(f'Test error = {error}')

    



def main_kernelized_SVM():

    np.random.seed(3270)
    
    features, labels = load_data("classification/svm/svmData.csv")
    labels = labels[:, None]
    train_features, train_labels, test_features, test_labels = random_test_train_split(features, labels, 50)
    
    train_features, mean_f, std_f = standardize_training(train_features)
    test_features = (test_features - mean_f) / std_f


    sigma = 0.1
    C = 1
    c_values = [0.01, 0.1, 1, 10, 100, 200, 300, 400]
    sigmas = [0.01, 1, 2, 3, 4, 5, 10, 20]
    c_values = [10, 100, 1000]
    test_errors = []

    kernel = 'rbf'

    for C in c_values:
      errors_c = []
      for sigma in sigmas:
        start = time()
        alphas_mul_labels, sv_features, b, w, hinge_losses = train_dual_soft_SVM(train_features, train_labels, C, sigma, kernel=kernel)
        #print(f'time elapsed = {(time()-start)}')
        svm_params = sv_features, alphas_mul_labels, b, sigma

        predictions = predict_kernel_SVM(test_features, svm_params, kernel=kernel)
        error = get_error(predictions, test_labels)
        errors_c.append(error)

        print(f'w={w}, b={b}')

        title = f'C = {C} sigma = {sigma} error = {error}'
        plot_decision_boundary(svm_params, train_features, train_labels, kernel=kernel, hinge_losses=hinge_losses, title=title)

      test_errors.append(errors_c)
    
    fig, ax = plt.subplots(figsize=(20, 5))

    test_errors = np.array(test_errors)
    im = plt.imshow(test_errors, cmap='cool', interpolation='none')
    
    values = np.unique(test_errors.ravel())
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label="error = {l}".format(l=values[i]) ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


    labels_c = ['0'] + [str(C) for C in c_values]
    labels_sigma = ['0'] + [str(sigma) for sigma in sigmas]
    print(labels_sigma)
    ax.set_xticklabels(labels_sigma)
    ax.set_yticklabels(labels_c)

    plt.title('Validation losses')
    plt.xlabel('sigmas')
    plt.ylabel("Cs")
    
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
  main_linear_SVM()
  main_kernelized_SVM()

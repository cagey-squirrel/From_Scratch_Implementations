import numpy as np


class Stamp(object):

    def __init__(self, split_search_strategy='intervals'):
        self.feature = 0   # Which feature is used for splitting: for example split people by height
        self.treshold = 0  # What treshold is used for splitting: for example which people are over 180cm
        self.result = -1   # What is the predicted class for that split: for example for people over 180cm return -1
        

        if split_search_strategy not in ['intervals', 'all_values']:
            raise Exception(f"Only possible values for spit_search are 'intervals' and 'all_values', got: {split_search_strategy}")
        self.split_search_strategy = split_search_strategy


    def fit(self, data, labels, weights):
        '''
        Fit function just finds the best split treshold and sets feature, treshold and result attributes
        '''
        self._find_best_split(data, labels, weights)


    def _find_best_split(self, data, labels, weights, num_intervals=100):
        '''
        Tries to find the best split by searching values between minimum and maximum values in data, dividing it in 'num_intervals' intervals
        '''
        m, n = data.shape
        min_error = 1
        for feature in range(n):
            data_by_feature = data[:, feature] 
            

            if self.split_search_strategy == 'intervals':
                min_val = data_by_feature.min()
                max_val = data_by_feature.max()
                values = np.linspace(min_val, max_val, num_intervals)
            
            else:
                values = data_by_feature = data[:, feature] 


            for value in values:
                result = -1
                predictions = np.ones(m)
                predictions[data_by_feature >= value] = -1

                errors = weights * (predictions != labels)
                error = errors.sum()

                # If error 
                if error > 0.5:
                    error = 1 - error
                    result = 1
                
                if error < min_error:
                    min_error = error
                    self.feature = feature
                    self.treshold = value
                    self.result = result


    def _single_predict(self, example):
        '''
        Predicts the result for a single example
        '''
        if example[self.feature] >= self.treshold:
            return self.result
        else:
            return -self.result


    def predict(self, data):
        '''
        Predicts the result for all data
        This function is just a vector wrapper around _single_predict function
        '''
        predictions = []

        for example in data:
            pred = self._single_predict(example)
            predictions.append(pred)
        
        return np.array(predictions)
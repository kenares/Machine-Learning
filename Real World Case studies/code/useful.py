
# coding: utf-8

# In[1]:
import math
import numpy 

class useful():
    
    def compute_entropy(column):
        """
        Convenience function:
            Calculate entropy given a pandas Series, list, or numpy array
        """
        counts = numpy.bincount(column)
        probabilities = counts / len(column)
    
        entropy = 0
    
        for prob in probabilities:
            if prob > 0 : 
                entropy += prob * math.log(prob, 2)
        
        return -entropy
    

    def compute_information_gain(data, split_name, target_name):
        """
        Calculate information gain given a dataset, column to split on, and target
        """
        total_entropy = useful.compute_entropy(data[target_name])
    
        column = data[split_name]
        median = column.median()
    
        left_split = data[column <= median]
        right_split = data[column > median]
    
        to_subtract = 0
    
        for subset in [left_split, right_split]:
            prob = (subset.shape[0] / data.shape[0])
            to_subtract += prob * useful.compute_entropy(subset[target_name])
        
        return total_entropy - to_subtract


    def find_best_column(data, target_name, columns):
        """
        Convenience function:
            find the optimale column for ID3 to split on
        """
        information_gain = {}
        for column in columns : 
            information_gain[column] = useful.compute_information_gain(data, column, target_name)
        
        return information_gain, sorted(information_gain, key = information_gain.get)


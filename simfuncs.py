"""
General helper functions for simulations

Preston Huft
"""
import csv
import numpy as np


## reading to and writing from files

def soln_to_csv(fname, data, labels):
    """
    fname: myfile.csv
    data: a list or array of equal length lists or arrays of data
    labels: a label describing each list in data
    
    e.g. 
        data = [array([1,2,3]), array([2,4,6], array([1,4,9]]
        labels = ['x', '2x', 'x^2']
    """
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')

        for d,l in zip(data, labels):
            writer.writerow([l] + list(d))

def soln_from_csv(fname):
    """
    fname: myfile.csv
    
    returns:
        data: an array of equal length arrays of data
        labels: a label describing each array in data
    
    e.g. 
        data = [array([1,2,3]), array([2,4,6], array([1,4,9]]
        labels = ['x', '2x', 'x^2']
    """
    labels = []
    data = []
    
    with open(fname, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')

        for row in reader:
            labels.append(row[0])
            data.append(np.array(row[1:], float))
            
    return data, labels
    
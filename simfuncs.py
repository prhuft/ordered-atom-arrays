"""
General helper functions for simulations

Preston Huft
"""
import csv
import numpy as np


## reading to and writing from files

def soln_to_csv(fname, data, labels):
    """
    Args:
        fname: myfile.csv
        data: a list or array of equal length lists or arrays of data
        labels: a label describing each list in data
    
        e.g., 
        data = [array([1,2,3]), array([2,4,6], array([1,4,9]]
        labels = ['x', '2x', 'x^2']
    """
    
    if type(data) == np.ndarray:
    # listify to avoid malforming strings
        data = [d for d in data]
    
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')

        for d,l in zip(data, labels):
            writer.writerow([l] + list(d))
            
    print(f"wrote data to {fname}")

def soln_from_csv(fname):
    """
    Args:
        fname: myfile.csv
    Returns:
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
            try:
                data.append(np.array([complex(x) for x in row[1:]]))
            except (ValueError,TypeError) as e:
                print(e)
                print(f"problematic row: {row}")
                break
            
    return data, labels
    
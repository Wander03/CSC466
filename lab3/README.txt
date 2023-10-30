Course: CSC 466
Quarter: Fall 2023
Assignment: Lab 3

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Output Files:
    C45: results_DT
    Random Forest: results_RF
    KNN: results_KNN

Final Model Parameters:

    Credit Approval Dataset

        C4.5
        Threshold = 0.3

        Random Forest:
        Attributes = 7
        Datapoints = 150
        Trees = 11
        Threshold = 0.1

        K Nearest Neighbors:
        K = 7

    Nursery Dataset

        C4.5
        Threshold = 0.1

        Random Forest:
        Attributes = 7
        Datapoints = 200
        Trees = 11
        Threshold = 0.1

        K Nearest Neighbors:
        K = 3


    Heart Disease Dataset

        C4.5
        Threshold = 0.1

        Random Forest:
        Attributes = 7
        Datapoints = 200
        Trees = 7
        Threshold = 0.1

        K Nearest Neighbors:
        K = 4

Description:

How to run IncduceC45.py:
    python3 InduceC45 <input file> 
                      <threshold value> 
                      <1 if gain ratio, 0 if gain> 
                      <OPTIONAL: restrictions file>

How to run classify.py:
    python3 classify.py <input file> 
                        <decision tree file> 
                        <OPTIONAL: 1 if silent run, 0 if not>

How to run validation.py: 
    python3 validation.py <training data> 
                          <number of folds>
                          <threshold value>
                          <1 if gain ratio, 0 if gain> 
                          <OPTIONAL: restictions file>

        number of folds: 0 = no cross-validation, -1 = all-but-one cross-validation

How to run randomForest.py: 
    python3 randomForest.py 
                    <input file> 
                    <NumAttributes> 
                    <NumDataPoints> 
                    <NumTrees>
                    <number of folds; 0 = no cross-validation, -1 = all-but-one cross-validation> 
                    <threshold value> 
                    <1 if gain ratio, 0 if gain> 

How to run knn.py:
    python3 knn.py <input file: consits of data to build classifier> 
                   <classify file: consists of data to be classified>
                   <K: number of neighbors to consider> 
                   <distance metric: 1 - eucledian, 2 - manhattan, 3 - cosine sim>
                   <OPTIONAL: restrictions file>

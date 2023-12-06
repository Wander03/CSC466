Course: CSC 466
Quarter: Fall 2023
Assignment: Lab 6

Name(s):
    Andrew Kerr // adkerr@calpoly.edu
    Brendan Callender // bscallen@calpoly.edu

Output Files:
   1. results_KNN   - has predicted authorships from KnnAuthorship.py
   2. results_RF    - has predicted authorships from RFAuthorship.py
   3. confusion     - has confusion matrixes
   4. out           - has distance matrix using cosine similarity


How to run: KnnAuthorship.py
	
    If Cosine Similarity:
                python3 KnnAuthorship.py
                        <ground truth file: consists of >
                        <tf-idf file: contains tf-idf vectorization of text docs >
                        <distance metric: 0 - cosine>
                        <K: number of neighbors to consider>

        If Okapi:
                python3 KnnAuthorship.py
                        <ground truth file: contains authorship info for each text doc>
                        <tf-idf file: contains tf-idf vectorization of text docs>
                        <distance metric: 0 - cosine>
                        <K: number of neighbors to consider>
                        <document frequency file: contains doc frequency for each word in vocab>
                        <term frequency file: contains tf vectorization of text docs>
                        <N: number of randomly selected docs to evaluate and predict using KNN>


How to run: RFAuthorship.py
	
	python3 RFAuthorship.py
                    <ground truth file>
                    <input file: consits of data to build classifier> 
                    <Num Attributes (per tree)> 
                    <Num Data Points (per tree)>
                    <Num Trees (in forest)>
                    <threshold value> 
                    <1 if gain ratio, 0 if gain> 
                    <outfile name>


How to run: classifierEvaluation.py
	
	python3 classifierEvaluation.py
                    <ground truth file>
                    <predictions file>
                    <0 if RF, 1 if KNN>


Output:

Some output from classifierEvaluation.py is printed to the terminal.
To write output to a specific out file do:

	python3 program.py arg_1 ... arg_n >> output_file.txt

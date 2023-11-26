Course: CSC 466
Quarter: Fall 2023
Assignment: Lab 5

Name(s):
    Andrew Kerr // adkerr@calpoly.edu
    Brendan Callender // bscallen@calpoly.edu


Output:

The output of each program is printed to the terminal.
To write output to a specific out file do:

	python3 program.py arg_1 ... arg_n >> output_file.txt


How to run: EvaluateCFRandom.py
	
	python3 EvaluateCFRandom.py
            <jester-data-1.csv: Path to jester-data-1.csv file>
            <Method: Memory-based method to use>
                Valid Methods:
                    1: Weighted Sum (User Based)
                    2: Weighted Sum (Item Based)
                    3: Adj Weighted Sum (User Based)
                    4: Adj Weighted Sum (Item Based)
                    5: Weighted Sum w/ KNN (User Based)
                    6: Adj Weighted Sum w/ KNN (User Based)
            <Size: Number of test cases to generate>
            <Repeats: Number of times to repeat test>
            <OPTIONAL: KNN: Number of Nearest Neighbors to use>


How to run: EvaluateCFList.py
	
	python3 EvaluateCFList.py
            <jester-data-1.csv: Path to jester-data-1.csv file>
            <Method: Memory-based method to use>
                Valid Methods:
                    1: Weighted Sum (User Based)
                    2: Weighted Sum (Item Based)
                    3: Adj Weighted Sum (User Based)
                    4: Adj Weighted Sum (Item Based)
                    5: Weighted Sum w/ KNN (User Based)
                    6: Adj Weighted Sum w/ KNN (User Based)
            <Filename: File containing list of test cases>
            <OPTIONAL: KNN: Number of Nearest Neighbors to use>


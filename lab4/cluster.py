"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 4

Name(s):
    Sophia Chung // spchung@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    Creates a Cluster class for use in kmeans
"""
from __future__ import annotations
import numpy as np


class Cluster:
    def __init__(self, centroid, size):
        self.centroid = centroid
        self.s = [0] * size
        self.num = 0
        self.points = []

    def __lt__(self, other: Cluster) -> bool:
        return self.num < other.num

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cluster):
            return False
        else:
             return self.centroid == other.centroid

    def __repr__(self) -> str:
        return f'(Centroid: {self.centroid})'
    
    def __str__(self) -> str:
        return f'(Centroid: {self.centroid}\nSum: {self.s}\nPoints: {self.points})'

    def get_centroid(self) -> list:
        return self.centroid

    def get_points(self) -> list:
        return self.points
    
    def get_num(self) -> int:
        return self.num

    def add(self, i: int, x: list) -> None:
        self.num += 1
        self.s = [val + self.s[index] for index, val in enumerate(x)]
        self.points.append(i)
    
    def remove(self, i: int, x: list) -> None:
        self.num -= 1
        self.s = [self.s[index] - val for index, val in enumerate(x)]
        self.points.remove(i)

    def mean(self) -> list:
        return np.divide(self.s, self.num)

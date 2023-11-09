"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 4

Name(s):
    Sophia Chung // spchung@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    Creates a Cluster class for use in hclustering
"""
from __future__ import annotations
import numpy as np


class Cluster:
    def __init__(self, points, height=None, children=None, data=None):
        self.points = points
        self.height = height
        self.children = children
        self.data = data

    def __lt__(self, other: Cluster) -> bool:
        return self.height < other.height

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cluster):
            return False
        else:
             return self.points == other.points

    def __repr__(self) -> str:
        return f'(Points: {self.points})'
    
    def __str__(self) -> str:
        return f'(Points: {self.points}\nHeight: {self.height}\nChildren: {self.children}\nData: {self.data})'

    def get_points(self) -> list:
        return self.points
    
    def get_height(self) -> float:
        return self.height

    def get_children(self) -> list:
        return self.children

    def get_data(self) -> list:
        return self.data

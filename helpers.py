from dataclasses import dataclass
import math
'''
TODO implement the hardest stuff which is the width decider and the backtrack decider
'''
@dataclass
# 2d point 
# My Dumbass should have done a 2d-pose class, will change that if this ends up working well
class Point2D:
    x: float
    y: float
    
    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)
    
    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)
    
    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_tuple(self):
        return (self.x, self.y)
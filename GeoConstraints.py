# Code by Robert Ren - r^2
# robert71@uw.edu or 425-553-5361
import numpy as np
import math
from typing import List, Tuple, Set, Dict, Optional
from width_calculations import calculate_widths
from helpers import Point2D

# Geometric constraint checker, makes sure that the current lane boundaries are working okay
class GeometricConstraints:
    def __init__(self, w_min=2.5, w_max=6.5, d_max=5.5, phi_max=90.0):
        self.w_min = w_min  # minimum lane width
        self.w_max = w_max  # maximum lane width  
        self.d_max = d_max  # maximum distance between consecutive points
        self.phi_max = math.radians(phi_max)  # maximum angle between segments
    
    
    def check_segment_constraint(self, path: List[int], points: List[Point2D]) -> bool:
        """Check Cseg constraint - angles between consecutive segments"""
        if len(path) < 3:
            return True
        
        # getting last 3 points
        i = len(path) - 3
        p1, p2, p3 = points[path[i]], points[path[i+1]], points[path[i+2]]
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle between vectors
        angle = self._angle_between_vectors(v1, v2)
        
        if abs(angle) > self.phi_max:
            return False
                
        return True

    def check_length_constraint(self, path : List[int], points : List[Point2D]) -> bool:
        if(len(path) < 2):
            return True
        
        #get last 2 points
        i = len(path) - 2
        p1, p2 = points[path[i]], points[path[i + 1]]

        return p1.distance(p2) < self.d_max
    

    #TODO Change this to the more comprehensive width finding algorithm
    def check_width_constraint(self, left_path: List[int], right_path: List[int], 
                             points: List[Point2D], M_fixed_prev) -> bool:
        return calculate_widths(left_path, right_path, points, M_fixed_prev)
    
    # Function that determines whether a boundary pair intersects with itself
    # left_path: the left path of the boundary, all elements are integers representing nodes in the graph
    # right_path: the right path of the boundary, all elements are integers representing nodes in the graph
    # points: points is the mapping of nodes (integers/index position in list) to actual physical points
    def check_polygon_constraint(self,
                                left_path: List[int],
                                right_path: List[int],
                                points: List[Point2D]) -> bool:
        should_backtrack = True # whether or not the constraint passed or not, shoudl we continue to backtrack
        constraint_passed = True # whether the geometric constraint itself has passed

        # too small to run the test
        if len(left_path) < 2 or len(right_path) < 2:
            return True, True

        # Build polygon indices: L followed by reversed R
        L = len(left_path)
        polygon_indices = left_path + right_path[::-1]
        n = len(polygon_indices)

        # Not possible to intersect if there are only 3 points, only possible with 4
        if n < 4:
            return True, True

        # Helper to fetch segment endpoints for segment k: [k, k+1] mod n
        def seg_endpoints(k: int):
            a = polygon_indices[k]
            b = polygon_indices[(k + 1) % n]
            return points[a], points[b]

        # Identify the dynamic 'bridge' edge: connects left_path[-1] to right_path[-1]
        # In the concatenated polygon, that's exactly segment with index (L-1).
        bridge_idx = L - 1 if L > 0 else -1

        # Determines whether segment i shares an endpoint index with segment j
        def share_vertex(i: int, j: int) -> bool:
            i0, i1 = seg_endpoints(i)
            j0, j1 = seg_endpoints(j)
            return (i0 == j0 or i0 == j1 or i1 == j0 or i1 == j1)

        # Check all unordered pairs of segments
        for i in range(n):
            for j in range(i + 1, n):
                # Skip identical segment or immediate neighbors (adjacent in cycle)
                # This condition should theortically never be run but my pussy ass is too scared to delete it LMAO
                if j == i:
                    continue
                
                # Shouldn't share end point with query
                # again, I could change the loop starting positions to deal with this but this is here as a failsafe
                # and to better show my logic and the conditions going into the code
                if j == (i + 1) % n or i == (j + 1) % n:
                    continue
                
                # Skip pairs that share a vertex for any reason (e.g., duplicate indices)
                if share_vertex(i, j):
                    continue

                # whether or not we are on the bridge that connects the left path to the right
                on_bridge = (i == bridge_idx or j == bridge_idx)

                # Determining whether or not there is an intersection
                # See documentation in the google drive or wherever the fuck the team decides to move it for the mathematics
                p1, q1 = seg_endpoints(i)
                p2, q2 = seg_endpoints(j)

                if self._segments_intersect(p1, q1, p2, q2):
                    # if we are on the bridge, we have failed the geometric constraints
                    # however, we should continue to backtrack because this is just the bridge
                    if(on_bridge):
                        constraint_passed = False
                    else:
                        return False, False

        return constraint_passed, should_backtrack
    
    # Determines whether the left path and the right path violates constraints
    def constraints_satisfied(self, left_path: List[int], right_path: List[int], 
                            points: List[Point2D]) -> bool:
        
        
        width_package = self.check_width_constraint(left_path, right_path, points)
        polygon_package = self.check_polygon_constraint(left_path, right_path, points)

        left_angle = self.check_segment_constraint(left_path, points)
        right_angle = self.check_segment_constraint(right_path, points)

        left_length = self.check_length_constraint(left_path, points)
        right_length = self.check_length_constraint(right_path, points)
        """Check if all constraints are currently satisfied"""
        constraint_satisfied = (left_angle and right_angle and left_length and 
                                right_length and width_package[2] and polygon_package[0])
        
        """Determine whether we should backtrack"""
        should_backtrack = (left_angle and right_angle and left_length and 
                                right_length and width_package[3] and polygon_package[1])
        
        return constraint_satisfied, should_backtrack, width_package[0]
    # Calculates unsigned angle between 2 vectors
    def _angle_between_vectors(self, v1: Point2D, v2: Point2D) -> float:
        dot_product = v1.x * v2.x + v1.y * v2.y
        norm1 = v1.norm()
        norm2 = v2.norm()
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.acos(cos_angle)
    
    # Calculates if two segments intersects
    # See documentation if curious about the mathematics
    def _segments_intersect(self, p1: Point2D, q1: Point2D, p2: Point2D, q2: Point2D, eps: float = 1e-9) -> bool:
        def orient(a: Point2D, b: Point2D, c: Point2D) -> float:
            return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)

        def on_segment(a: Point2D, b: Point2D, c: Point2D) -> bool:
            return (min(a.x, b.x) - eps <= c.x <= max(a.x, b.x) + eps and
                    min(a.y, b.y) - eps <= c.y <= max(a.y, b.y) + eps)

        o1 = orient(p1, q1, p2)
        o2 = orient(p1, q1, q2)
        o3 = orient(p2, q2, p1)
        o4 = orient(p2, q2, q1)

        # Proper intersection
        if (o1*o2 < 0) and (o3*o4 < 0):
            return True

        # Colinear / touching cases
        if abs(o1) <= eps and on_segment(p1, q1, p2): return True
        if abs(o2) <= eps and on_segment(p1, q1, q2): return True
        if abs(o3) <= eps and on_segment(p2, q2, p1): return True
        if abs(o4) <= eps and on_segment(p2, q2, q1): return True

        return False
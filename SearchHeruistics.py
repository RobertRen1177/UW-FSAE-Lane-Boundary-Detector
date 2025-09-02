import numpy as np
import math
from typing import List, Tuple, Set, Dict, Optional
from helpers import Point2D
# Search Heuristics for the DFS search
class SearchHeuristics:
    def __init__(self, car_heading: Optional[Point2D] = None):
        self.car_heading = car_heading or Point2D(1.0, 0.0)
    
    # Choose the next vertice
    def next_vertex_decider(self, path: List[int], unvisited: Set[int], 
                           points: List[Point2D], graph: Dict[int, Set[int]]) -> Optional[int]:
        if not unvisited:
            return None
            
        if len(path) < 2:
            # Use car heading direction for initial direction
            current_point = points[path[-1]]
            best_vertex = None
            min_angle = float('inf')
            
            for vertex in unvisited:
                if vertex in graph[path[-1]]:
                    next_point = points[vertex]
                    direction = next_point - current_point
                    angle = abs(self._angle_between_vectors(self.car_heading, direction))
                    
                    if angle < min_angle:
                        min_angle = angle
                        best_vertex = vertex
                        
            return best_vertex
        
        # Use last two points to determine direction
        p_prev = points[path[-2]]
        p_current = points[path[-1]]
        current_direction = p_current - p_prev
        
        best_vertex = None
        min_angle = float('inf')
        
        for vertex in unvisited:
            if vertex in graph[path[-1]]:
                p_next = points[vertex]
                next_direction = p_next - p_current
                
                angle = abs(self._angle_between_vectors(current_direction, next_direction))
                
                if angle < min_angle:
                    min_angle = angle
                    best_vertex = vertex
                    
        return best_vertex
    
    def left_right_decider(self, left_path: List[int], right_path: List[int],
                          left_next: Optional[int], right_next: Optional[int], points: List[Point2D]) -> int:
        if left_next is None and right_next is None:
            return 0  # arbitrary
        if left_next is None:
            return 1
        if right_next is None:
            return 0
        
        if len(left_path) < 2 or len(right_path) < 2:
            return 0  # Choose left arbitrarily
        
        # Calculate angles for both potential path pairs
        left_angles_1 = self._calculate_path_angles(left_path + [left_next], right_path, points)
        left_angles_2 = self._calculate_path_angles(left_path, right_path + [right_next], points)
        
        # Choose the option that minimizes angle difference
        diff_1 = abs(left_angles_1[0] - left_angles_1[1]) if left_angles_1[0] is not None and left_angles_1[1] is not None else float('inf')
        diff_2 = abs(left_angles_2[0] - left_angles_2[1]) if left_angles_2[0] is not None and left_angles_2[1] is not None else float('inf')
        
        return 0 if diff_1 < diff_2 else 1
    
    def _calculate_path_angles(self, left_path: List[int], right_path: List[int], 
                             points: List[Point2D]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate theta_l and theta_r angles for path pair"""
        if len(left_path) < 2 or len(right_path) < 2:
            return None, None
            
        # Get last points of each path
        l_prev = points[left_path[-2]]
        l_curr = points[left_path[-1]]
        r_prev = points[right_path[-2]]
        r_curr = points[right_path[-1]]
        
        # Calculate angles
        left_direction = l_curr - l_prev
        right_direction = r_curr - r_prev
        connection = r_curr - l_curr

        if connection.x == 0.0 and connection.y == 0.0:
            return math.pi, math.pi
        
        theta_l = self._angle_between_vectors(left_direction, connection)
        theta_r = self._angle_between_vectors(right_direction, Point2D(-connection.x, -connection.y))
        
        return theta_l, theta_r
    
    def _angle_between_vectors(self, v1: Point2D, v2: Point2D) -> float:
        """Calculate angle between two vectors"""
        dot_product = v1.x * v2.x + v1.y * v2.y
        norm1 = v1.norm()
        norm2 = v2.norm()
        
        if norm1 == 0 or norm2 == 0:
            return math.pi
            
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        
        return math.acos(cos_angle)

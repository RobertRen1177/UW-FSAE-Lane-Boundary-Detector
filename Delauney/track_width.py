'''
Given a path, and the left and right paths of the path, calculate the track width of every single midpoint
To do this, simply determine the closest points from each midpoint to the left and right boundaries represented
as a polyline
'''
#code by Robert Ren - r^2
import numpy as np

def closest_point_to_polyline(external_point, poly_line):
    external_point = np.array(external_point)
    poly_line = np.array(poly_line)
    
    min_distance = float('inf')
    closest_point = poly_line[0]
    
    for i in range(len(poly_line) - 1):
        p1 = poly_line[i]
        p2 = poly_line[i + 1]
        
        line_vec = p2 - p1
        point_vec = external_point - p1
        
        line_len_squared = np.dot(line_vec, line_vec)
        
        if line_len_squared == 0:
            candidate_point = p1
        else:
            t = np.dot(point_vec, line_vec) / line_len_squared
            t = max(0, min(1, t))
            candidate_point = p1 + t * line_vec
        
        distance = np.linalg.norm(external_point - candidate_point)
        
        if distance < min_distance:
            min_distance = distance
            closest_point = candidate_point
    
    return closest_point

def calculate_track_width(path, left_path, right_path, node_positions):
    track_widths = []
    for node in path:
        midpoint = node_positions[node]
        closet_left = closest_point_to_polyline(midpoint, left_path)
        closet_right = closest_point_to_polyline(midpoint, right_path)
        track_widths.append(np.linalg.norm(closet_left - closet_right))
    return track_widths

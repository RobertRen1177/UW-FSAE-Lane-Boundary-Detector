from scipy.spatial import Delaunay
import numpy as np
import math
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.spatial import Delaunay
from collections import deque
from track_width import calculate_track_width

'''
This file is super confusing so im going to a little TLDR
First this file builds a delauney triangulation through the cone poins and finds the midpoint of each edge
The a graph is built (bidirectional) such that each midpoint connects to every other midpoint in each triangle it is in
Car is then connected to the graph

Starting from the car position, "BFS" algorithm that can loop is ran to find all paths (paths that are blatantly not right are pruned)
Each valid path is run through a cost function and the lowest cost one is picked
-RR
'''
#pruning constants
PRUNING_ANGLE = 45
PRUNING_PATH_SEG_LENGTH = 10
PRUNING_EDGE_LENGTH = 10

#Cost function weights
ANGLE_CHANGE_WEIGHT = 4.0
TRACK_WIDTH_WEIGHT = 2.5
PATH_LENGTH_WEIGHT = 1.5
SEGMENT_LENGTH_WEIGHT = 1.5


#NORMALIZATION
ANGLE_NORM = 45
TRACK_WIDTH_NORM = 3.0
PATH_LENGTH_NORM = 25
CONE_LENGTH_NORM = 4.0
SEGMENT_LENGTH_NORM = 3.0

def build_midpoint_graph(cones, car_position = None, car_heading = None):
    '''
    cones: the cone map given from perception
    car_position: the position of the car

    Creates a graph through the delauney triangulation of the cone map

    returns the graph, the start node, a edge->midpoint dictionary, and midpoint->triangle dictionary
    '''
    #triangulate the grid to create order within disorder
    triangulation = Delaunay(cones)

    #get all triangles
    triangles = triangulation.simplices

    midpoints = {} #dictionary where key is the edge and the key is the idpoint NODE (not the actual midpoint)

    midpoint_to_edge = {}

    G = nx.Graph()

    for simplex in triangles:
        triangle_midpoints = []  #midpoints for this triangle
        triangle = [simplex[0], simplex[1], simplex[2]]  #Triangle points
        for i in range(3):
            p1, p2 = sorted((simplex[i], simplex[(i + 1) % 3])) #Mod and sorting to avoid duplicates (formatting)
            edge = (p1, p2) #these are the indices of cones btw, not the actual cone points

            if edge not in midpoints: #dont wanna duplicate
                midpoint = (cones_for_triangulation[p1] + cones_for_triangulation[p2]) / 2
                midpoints[edge] = len(midpoints)
                G.add_node(midpoints[edge], pos=midpoint)

            
            midpoint_idx = midpoints[edge]
            triangle_midpoints.append(midpoint_idx) #store for connectivity (will need to be used later in the graph computation)

            if midpoint_idx not in midpoint_to_edge:
                midpoint_to_edge[midpoint_idx] = edge

        #connect midpoints of the same triangle
        #Note, the graph is undirected so you can loop around the triangle both counter and clockwise
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(triangle_midpoints[i], triangle_midpoints[j])
    

    #connecting the car to the graph, simply using the basic solution of connecting car to the closest triangle
    car_node = None
    if car_position is not None:
        #find the closest midpoint to the car
        closest_midpoint_idx = None
        closest_midpoint = None
        min_distance = float('inf')

        for edge, idx in midpoints.items():
            midpoint = (cones_for_triangulation[edge[0]] + cones_for_triangulation[edge[1]]) / 2
            distance = np.linalg.norm(car_position - midpoint)

            if distance < min_distance:
                min_distance = distance
                closest_midpoint_idx = edge

        if closest_midpoint_idx is not None:
            #Find all triangles containing the closest edge
            connected_midpoints = set()

            for simplex in triangles:
                edges = [
                    tuple(sorted((simplex[0], simplex[1]))),
                    tuple(sorted((simplex[1], simplex[2]))),
                    tuple(sorted((simplex[0], simplex[2])))
                ]

                if closest_midpoint_idx in edges:  #if the closest edge is part of a triangle
                    for edge in edges:
                        if edge in midpoints:
                            connected_midpoints.add(midpoints[edge])
            
            car_node = len(G.nodes) 
            G.add_node(car_node, pos=car_position)

            for mp_idx in connected_midpoints:
                G.add_edge(car_node, mp_idx)
    
    return G, midpoints, car_node, midpoint_to_edge

def find_all_paths_bfs(G, car_node, midpoints, cones):
    '''
    G: Graph through the midpoints of delauney triangulation
    car_node: the node of the car or the starting node
    midpoints: edge->midpoint map
    cones: the cone positions given from perception

    finds all valid paths through the graph, while pruning out blatantly wrong paths

    returns all valid paths in a list
    '''
    valid_paths = []
    queue = deque([[car_node]]) #starting bfs queue
    while queue:
        path = queue.popleft()
        last_node = path[-1] 
        
        #BFS time, but we can explore multiple neighbors even if they've been explored by other paths
        for neighbor in G.neighbors(last_node):
            if neighbor in path: #preventing cycles from happening
                continue

            new_path = path + [neighbor]

            #getting midpoint positions (like actual positions)
            new_pos = np.array(G.nodes[neighbor]['pos'])
            last_pos = np.array(G.nodes[last_node]['pos'])

            #Compute segment length
            #if the distance between the new position and the last position is really big, then skip the path
            segment_length = np.linalg.norm(new_pos - last_pos)
            if segment_length > PRUNING_PATH_SEG_LENGTH and last_node is not car_node:
                continue
            
            valid_angle = True
            if(len(new_path) > 2 and new_path[-3] is not car_node):
                prev_node = new_path[-3]
                mid_node = new_path[-2]
                curr_node = new_path[-1]

                prev_pos = np.array(G.nodes[prev_node]['pos'])
                mid_pos = np.array(G.nodes[mid_node]['pos'])
                curr_pos = np.array(G.nodes[curr_node]['pos'])

                vec1 = mid_pos - prev_pos
                vec1 /= np.linalg.norm(vec1)

                vec2 = curr_pos - mid_pos
                vec2 /= np.linalg.norm(vec2)

                dot_product = np.dot(vec1, vec2)
                angle_change = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angle_change_degrees = np.degrees(angle_change)
                if abs(angle_change_degrees) > PRUNING_ANGLE and last_node is not car_node:
                    valid_angle = False
            
            if not valid_angle:
                continue
            
            #if we are not at the first node, also check that the edge length is reasonable
            if(last_node != car_node):
                edge = next((e for e, idx in midpoints.items() if idx == last_node), None)
                if edge is None:
                    continue
                if(np.linalg.norm(cones[edge[0]] - cones[edge[1]]) > PRUNING_EDGE_LENGTH):
                    continue
            
            queue.append(new_path)
            valid_paths.append(new_path)
    
    #deleting first node from every path (car connection node)
    for i in range(len(valid_paths)):
        if(len(valid_paths[i]) == 0):
            continue
        valid_paths[i] = valid_paths[i][1:] 

    return valid_paths


def find_track_width(path, midpoints_to_edges, node_positions, cones):
    '''first construct proper boundaries (all points left of the path) (all points to the right of the path)'''
    if(len(path) == 1):
        #return length of edge as no one gives a fuck what the boundary is
        p1, p2 = midpoints_to_edges[path[-1]]
        return np.linalg.norm(cones[p1] - cones[p2])
    
    '''
    Determining what side of the path the boundary is
    all points to the left of the path (aka cross product is positive)
    all points to the right of the path (aka cross product is negative)
    '''
    left_side = []
    right_side = []
    visited = set()
    for i in range(len(path)):
        if(i == len(path) - 1):
            p1 = node_positions[path[i]]
            p2 = node_positions[path[i - 1]]
            path_dir = p1 - p2
        else:
            p1 = node_positions[path[i]]
            p2 = node_positions[path[i + 1]]
            path_dir = p2 - p1

        edge = midpoints_to_edges[path[i]]
        cone1 = cones[edge[0]]
        cone1_idx = edge[0]
        cone2 = cones[edge[1]]
        cone2_idx = edge[1]
    
        cone1_to_p1 = cone1 - p1
        cross_product_1 = cone1_to_p1[0] * path_dir[1] - cone1_to_p1[1] * path_dir[0]
        
        # Sort cones based on cross product
        if cross_product_1 > 0:
            if cone1_idx not in visited:
                left_side.append(cone1)
            if cone2_idx not in visited:
                right_side.append(cone2)
        else:
            if cone1_idx not in visited:
                left_side.append(cone2)
            if cone2_idx not in visited:
                right_side.append(cone1)
        
        visited.add(cone1_idx)
        visited.add(cone2_idx)
    
    return calculate_track_width(path, left_side, right_side, node_positions)


def compute_path_cost(path, node_positions, cones, midpoints_to_edge, sensor_range=10):
    '''
    path: the path that the function is computing its cost for
    node_positions: midpoint_index -> midpoint position dictionary
    midpoints: edge->midpoint map
    cones: the cone map given to us by perception
    triangle_dict: midpoint_index->triangle index
    sensor_range: how far the car can look ahead

    Computes a cost for a valid path

    Return: returns the cost of the path
    '''
    if len(path) < 2:
        return float('inf'), None  #Discard paths that are too short to have any meaning
    angles = []
    track_widths = []
    segment_lengths = []
    path_length = 0
    

    #Compute angle costs only if there are at least 3 nodes
    if len(path) >= 3:
        for i in range(-1, len(path) - 2):
            pos1 = np.array([0, 0]) if i == -1 else np.array(node_positions[path[i]]) #adding deviation from starting heading
            pos2 = np.array(node_positions[path[i + 1]])
            pos3 = np.array(node_positions[path[i + 2]])

            vec1 = pos2 - pos1
            vec2 = pos3 - pos2
            vec1 /= np.linalg.norm(vec1)
            vec2 /= np.linalg.norm(vec2)

            dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
            angle_change = np.degrees(np.arccos(dot_product))
            angles.append(angle_change)

            if(i != -1):
                segment_length = np.linalg.norm(pos2 - pos1)
                segment_lengths.append(segment_length)
                path_length += segment_length

    #if there are no track widths something went terribly wrong
    track_widths = find_track_width(path, midpoints_to_edge, node_positions, cones)
    track_widths_bool = [track_width < 2.5 or track_width > 7.0 for track_width in track_widths]
    if(any(track_widths_bool)):
        return float('inf'), None
    #compute standard deviations
    if(len(track_widths) < 2):
        std_track_width = TRACK_WIDTH_NORM
    else:
        std_track_width = np.std(track_widths)

    #compute standard deviations
    if(len(segment_lengths) < 2):
        std_segment_lengths = SEGMENT_LENGTH_NORM
    else:
        std_segment_lengths = np.std(segment_lengths)

    #compute angle cost only if angles exist
    max_angle_change = max(angles) if angles else ANGLE_NORM
    
    #path length penalty (favoring ~25m paths)
    path_length_penalty = (path_length - sensor_range) ** 2
    
    #final cost function that is very scuffed
    angle_cost = ANGLE_CHANGE_WEIGHT * max_angle_change / ANGLE_NORM #dont need conditional because path finder does this already
    
    segment_length_cost = SEGMENT_LENGTH_WEIGHT * std_segment_lengths / SEGMENT_LENGTH_NORM

    track_width_cost = TRACK_WIDTH_WEIGHT * std_track_width / TRACK_WIDTH_NORM

    path_length_cost = PATH_LENGTH_WEIGHT * path_length_penalty / PATH_LENGTH_NORM


    total_cost = angle_cost + track_width_cost + path_length_cost + segment_length_cost
    return total_cost, track_widths

def select_best_path(valid_paths, node_positions, cones, midpoints_to_edge):
    '''
    finds the best path
    '''
    best_path = None
    best_path_track_width = None
    best_cost = float('inf')
    for path in valid_paths:
        cost, track_width = compute_path_cost(path, node_positions, cones, midpoints_to_edge)
        if cost < best_cost:
            best_cost = cost
            best_path = path
            best_path_track_width = track_width

    if(best_path is None):
        return None, None
    #when the best path has been selected, calculate the track
    return best_path, best_path_track_width

import numpy as np

cones1 = np.array([
    [13.09269529,  4.45135511, -0.3596712 ],
    [17.17727473,  4.91481665, -0.49335899],
    [11.22424827,  1.73029441, -0.3219821 ],
    [ 9.38312001, -0.34894386, -0.2672492 ],
    [12.22028656,  8.53685001, -0.42736825],
    [ 8.51601783,  4.57713532, -0.3058427 ],
    [ 6.81384983, -1.31817688, -0.25840061],
    [14.05711148, -9.09950316, -0.50752836],
    [ 9.77083306,  6.75457689, -0.49327014],
    [ 4.26952468, -1.54373011, -0.20366112],
    [ 1.9445746 , -1.36197695, -0.27167486]
])

cones2 = np.array([
    [11.24418924,  0.09918644, -0.32758854],
    [13.87223508, -4.16553117, -0.38951149],
    [10.33628209, -5.77247043, -0.24183792],
    [15.44283371, -8.74982627, -0.37658794],
    [ 4.12687191, -1.15983222, -0.14020456],
    [ 2.89216805,  2.55772771, -0.15923898],
    [ 1.10214047, -2.36081286, -0.10111462]
])

cones3 = np.array([
    [15.56918874,  8.66339061, -0.14345157],
    [ 5.73710443,  2.39267125, -0.131736  ],
    [10.93323896, -0.40667089, -0.31727742],
    [13.40990377, -4.70887964, -0.36741134],
    [ 9.81898131, -6.17526504, -0.21469049],
    [ 3.85344146, -1.40947354, -0.12591276],
    [ 2.71623709,  2.37858344, -0.19524106]
])

cones4 = np.array([
    [14.16862571,  6.25760103, -0.16439619],
    [ 8.36072658, -2.04194506, -0.22524068],
    [10.21942458, -6.64782736, -0.22042864],
    [ 6.48185035, -7.61451682, -0.15499285],
    [ 1.18107043, -2.02150675, -0.10049133]
])

cones5 = np.array([
    [ 9.93465628, -1.07922008, -0.39643656],
    [12.80902476, -2.68188222, -0.51619025],
    [14.57903133, -4.87974922, -0.57992206],
    [15.11335652, -7.62324619, -0.57655988],
    [11.04122341, -8.09664899, -0.37682902],
    [ 6.53603306,  0.32142461, -0.33615402],
    [ 6.91948458, -3.56708974, -0.34691376],
    [ 4.47863847, -2.63600179, -0.33686307]
])

# cones6 is actually a duplicate of cones3
cones6 = np.array([
    [15.56918874,  8.66339061, -0.14345157],
    [ 5.73710443,  2.39267125, -0.131736  ],
    [10.93323896, -0.40667089, -0.31727742],
    [13.40990377, -4.70887964, -0.36741134],
    [ 9.81898131, -6.17526504, -0.21469049],
    [ 3.85344146, -1.40947354, -0.12591276],
    [ 2.71623709,  2.37858344, -0.19524106]
])

cones7 = np.array([
    [ 6.63568726, -1.1364633 , -0.25140037],
    [ 9.92854105, -8.89961183, -0.3084694 ],
    [ 6.42730238, -7.09800871, -0.16230461],
    [ 2.7494667 ,  1.87884724, -0.19433153]
])

cones8 = np.array([
    [ 7.98906507, -1.69342013, -0.21579882],
    [ 9.35134436, -5.53740481, -0.33276538],
    [ 5.44432161, -4.81995713, -0.15402686],
    [ 5.99747076, -8.68873783, -0.24660405]
])

cones9 = np.array([[9.03206651, -3.6854952, -0.33961914], [9.66560302, -6.81808691, -0.32954925], [9.38951545, -9.39995367, -0.27203662], [5.81633507, -4.95950717, -0.27163835],
[6.07295484, -6.78853607, -0.2706291 ], [2.68753045, -2.1297958 , -0.14568515]])

cones10 = np.array([[9.93465628, -1.07922008, -0.39643656], [12.80902476, -2.68188222, -0.51619025], [14.57903133, -4.87974922, -0.57992206],
[15.11335652, -7.62324619, -0.57655988], [11.04122341, -8.09664899, -0.37682902], [6.53603306, 0.32142461, -0.33615402], [6.91948458, -3.56708974, -0.34691376],
[4.47863847, -2.63600179, -0.33686307]])


if __name__ == '__main__':
    cones = cones10

    cones_for_triangulation = []
    for cone in cones:
        cones_for_triangulation.append([cone[0], cone[1]])

    cones_for_triangulation = np.array(cones_for_triangulation)
    
    start = time.time()
    G, midpoints, start_node, midpoint_to_edge = build_midpoint_graph(cones_for_triangulation, np.array([0, 0]), 0)

    node_pos = nx.get_node_attributes(G, 'pos')
    valid_paths = find_all_paths_bfs(G, start_node, midpoints, cones_for_triangulation)
    best_path, best_cost = select_best_path(valid_paths, node_pos, cones_for_triangulation, midpoint_to_edge)
    print(time.time() - start)
    print(len(valid_paths))


    # for path in valid_paths:
    #     plt.triplot(cones[:, 0], cones[:, 1], Delaunay(cones_for_triangulation).simplices, alpha=0.3)
    #     plt.scatter(cones[:, 0], cones[:, 1], color='red', label="Cones")

    #     #Plot midpoints
    #     for node, pos in node_pos.items():
    #         plt.scatter(pos[0], pos[1], color='blue')

    #     path_coords = np.array([node_pos[node] for node in path])
    #     plt.plot(path_coords[:, 0], path_coords[:, 1], linestyle="--", alpha=0.7)
    #     plt.axis('equal')
    #     plt.scatter(0, 0, color="green")
        
    #     print(compute_path_cost(path, node_pos, cones_for_triangulation, midpoint_to_edge))
    #     print()
    #     plt.show()


    plt.triplot(cones[:, 0], cones[:, 1], Delaunay(cones_for_triangulation).simplices, alpha=0.3)
    plt.scatter(cones[:, 0], cones[:, 1], color='red', label="Cones")
    plt.scatter(0, 0, color="green")
    #Plot midpoints
    for node, pos in node_pos.items():
        plt.scatter(pos[0], pos[1], color='blue')
    
    print("BEST PATH")
    path_coords = np.array([node_pos[node] for node in best_path])
    plt.plot(path_coords[:, 0], path_coords[:, 1], linestyle="--", alpha=0.7)
    plt.show()
    print(compute_path_cost(best_path, node_pos, cones_for_triangulation, midpoint_to_edge))



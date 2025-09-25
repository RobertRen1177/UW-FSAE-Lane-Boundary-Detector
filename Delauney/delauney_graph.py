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
from cones import *
'''
This file is super confusing so im going to a little TLDR
First this file builds a delauney triangulation through the cone poins and finds the midpoint of each edge
The a graph is built (bidirectional) such that each midpoint connects to every other midpoint in each triangle it is in
Car is then connected to the graph

Starting from the car position, "BFS" algorithm that can loop is ran to find all paths (paths that are blatantly not right are pruned)
Each valid path is run through a cost function and the lowest cost one is picked
-RR
'''
# pruning constants
PRUNING_ANGLE = 45
PRUNING_PATH_SEG_LENGTH = 10
PRUNING_EDGE_LENGTH = 10

# Cost function weights
ANGLE_CHANGE_WEIGHT = 4.0
TRACK_WIDTH_WEIGHT = 3.0
CONE_DIST_WEIGHT = 1.0
SEGMENT_LENGTH_WEIGHT = 3.0
PATH_LENGTH_WEIGHT = 1.5



# NORMALIZATION
ANGLE_NORM = 45
TRACK_WIDTH_NORM = 3.0
CONE_DIST_NORM = 2.0
PATH_LENGTH_NORM = 25
CONE_LENGTH_NORM = 2.0
SEGMENT_LENGTH_NORM = 3.0

# HELPER FUNCTIONS FOR FINDING TRACKWIDTH####################


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
#################################################################################


def build_midpoint_graph(cones, car_position=None, car_heading=None):
    print(cones)
    print()
    '''
    cones: the cone map given from perception
    car_position: the position of the car

    Creates a graph through the delauney triangulation of the cone map

    returns the graph, the start node, a edge->midpoint dictionary, and midpoint->triangle dictionary
    '''
    # triangulate the grid to create order within disorder
    triangulation = Delaunay(cones)

    # get all triangles
    triangles = triangulation.simplices

    # dictionary where key is the edge and the key is the idpoint NODE (not the actual midpoint)
    midpoints = {}

    midpoint_to_edge = {}

    G = nx.Graph()

    for simplex in triangles:
        triangle_midpoints = []  # midpoints for this triangle
        triangle = [simplex[0], simplex[1], simplex[2]]  # Triangle points
        for i in range(3):
            # Mod and sorting to avoid duplicates (formatting)
            p1, p2 = sorted((simplex[i], simplex[(i + 1) % 3]))
            # these are the indices of cones btw, not the actual cone points
            edge = (p1, p2)

            if edge not in midpoints:  # dont wanna duplicate
                midpoint = (cones[p1] + cones[p2]) / 2
                midpoints[edge] = len(midpoints)
                G.add_node(midpoints[edge], pos=midpoint)

            midpoint_idx = midpoints[edge]
            # store for connectivity (will need to be used later in the graph computation)
            triangle_midpoints.append(midpoint_idx)

            if midpoint_idx not in midpoint_to_edge:
                midpoint_to_edge[midpoint_idx] = edge

        # connect midpoints of the same triangle
        # Note, the graph is undirected so you can loop around the triangle both counter and clockwise
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(triangle_midpoints[i], triangle_midpoints[j])

    # connecting the car to the graph, connecting to the three closest midpoints
    car_node = None
    if car_position is not None:
        # find the three closest midpoints to the car
        midpoint_distances = []

        for edge, idx in midpoints.items():
            midpoint = (cones[edge[0]] + cones[edge[1]]) / 2
            distance = np.linalg.norm(car_position - midpoint)
            midpoint_distances.append((distance, idx))

        # sort by distance and take the three closest
        midpoint_distances.sort(key=lambda x: x[0])
        three_closest = midpoint_distances[:3]

        if three_closest:
            car_node = len(G.nodes)
            G.add_node(car_node, pos=car_position)

            for _, mp_idx in three_closest:
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
    queue = deque([[car_node]])  # starting bfs queue
    while queue:
        path = queue.popleft()
        last_node = path[-1]

        # BFS time, but we can explore multiple neighbors even if they've been explored by other paths
        for neighbor in G.neighbors(last_node):
            if neighbor in path:  # preventing cycles from happening
                continue

            new_path = path + [neighbor]

            # getting midpoint positions (like actual positions)
            new_pos = np.array(G.nodes[neighbor]['pos'])
            last_pos = np.array(G.nodes[last_node]['pos'])

            # Compute segment length
            # if the distance between the new position and the last position is really big, then skip the path
            segment_length = np.linalg.norm(new_pos - last_pos)
            if segment_length > PRUNING_PATH_SEG_LENGTH and last_node is not car_node:
                continue

            valid_angle = True
            if (len(new_path) > 2 and new_path[-3] is not car_node):
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

            # if we are not at the first node, also check that the edge length is reasonable
            if (last_node != car_node):
                edge = next((e for e, idx in midpoints.items()
                            if idx == last_node), None)
                if edge is None:
                    continue
                if (np.linalg.norm(cones[edge[0]] - cones[edge[1]]) > PRUNING_EDGE_LENGTH):
                    continue

            queue.append(new_path)
            valid_paths.append(new_path)

    # deleting first node from every path (car connection node)
    for i in range(len(valid_paths)):
        if (len(valid_paths[i]) == 0):
            continue
        valid_paths[i] = valid_paths[i][1:]

    return valid_paths

def find_track_width(path, midpoints_to_edges, node_positions, cones):
    """
    Separate cones into left and right sides relative to the path.
    Returns: (track_width, left_distances, right_distances)
    """

    if len(path) == 1:
        # Edge case: single edge only
        p1, p2 = midpoints_to_edges[path[-1]]
        return np.linalg.norm(cones[p1] - cones[p2]), [], []

    left_side = []
    right_side = []
    visited = set()

    for i in range(len(path)):
        # Determine path direction
        if i == len(path) - 1:  # last node → use backward direction
            p1 = node_positions[path[i]]
            p2 = node_positions[path[i - 1]]
            path_dir = p1 - p2
        else:  # all other nodes → use forward direction
            p1 = node_positions[path[i]]
            p2 = node_positions[path[i + 1]]
            path_dir = p2 - p1

        edge = midpoints_to_edges[path[i]]
        for cone_idx in edge:
            if cone_idx in visited:
                continue
            cone = cones[cone_idx]
            vec = cone - p1
            cross = vec[0] * path_dir[1] - vec[1] * path_dir[0]

            if cross > 0:
                left_side.append((cone_idx, cone))
            else:
                right_side.append((cone_idx, cone))

            visited.add(cone_idx)

    # Deduplicate by index, keep only cone coordinates
    left_side_u = [cone for idx, cone in dict(left_side).items()]
    right_side_u = [cone for idx, cone in dict(right_side).items()]

    # Distance helper
    def distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # Consecutive distances along each side
    left_distance = [distance(left_side_u[i], left_side_u[i + 1])
                     for i in range(len(left_side_u) - 1)]
    right_distance = [distance(right_side_u[i], right_side_u[i + 1])
                      for i in range(len(right_side_u) - 1)]

    # Combine left and right distances
    combined_distances = left_distance + right_distance

    # Track width calculation (your custom function)
    return calculate_track_width(path, left_side_u, right_side_u, node_positions), combined_distances




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
        # Discard paths that are too short to have any meaning
        return float('inf'), None
    angles = []
    track_widths = []
    segment_lengths = []
    path_length = 0

    # Compute angle costs only if there are at least 3 nodes
    if len(path) >= 3:
        for i in range(-1, len(path) - 2):
            # adding deviation from starting heading
            pos1 = np.array([0, 0]) if i == - \
                1 else np.array(node_positions[path[i]])
            pos2 = np.array(node_positions[path[i + 1]])
            pos3 = np.array(node_positions[path[i + 2]])

            vec1 = pos2 - pos1
            vec2 = pos3 - pos2
            vec1 /= np.linalg.norm(vec1)
            vec2 /= np.linalg.norm(vec2)

            dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
            angle_change = np.degrees(np.arccos(dot_product))
            angles.append(angle_change)

            if (i != -1):
                segment_length = np.linalg.norm(pos2 - pos1)
                segment_lengths.append(segment_length)
                path_length += segment_length

    # if there are no track widths something went terribly wrong
    track_widths, combined_distances = find_track_width(
        path, midpoints_to_edge, node_positions, cones)
    
    track_widths_bool = [track_width < 2.5 or track_width >
                         7.0 for track_width in track_widths]
    
    if (any(track_widths_bool)):
        return float('inf'), None
    
    # compute standard deviations
    if (len(track_widths) < 2):
        std_track_width = TRACK_WIDTH_NORM
    else:
        std_track_width = np.std(track_widths)
        std_track_width = std_track_width if std_track_width < TRACK_WIDTH_NORM else TRACK_WIDTH_NORM
    
    if (len(combined_distances) < 2):
        std_combined_distances = CONE_DIST_NORM
    else:
        std_combined_distances = np.std(combined_distances)
        std_combined_distances = std_combined_distances if std_combined_distances < CONE_DIST_NORM else CONE_DIST_NORM
    
    if(len(segment_lengths) < 2):
        std_segment_length = SEGMENT_LENGTH_NORM
    else:
        std_segment_length = np.std(segment_lengths)
        std_segment_length = std_segment_length if std_segment_length < SEGMENT_LENGTH_NORM else SEGMENT_LENGTH_NORM

    # compute angle cost only if angles exist
    max_angle_change = max(angles) if angles else ANGLE_NORM

    # path length penalty (favoring ~25m paths)
    path_length_penalty = (path_length - sensor_range) ** 2
    path_length_penalty = path_length_penalty if path_length_penalty < PATH_LENGTH_NORM else PATH_LENGTH_NORM

    # final cost function that is very scuffed
    # dont need conditional because path finder does this already
    angle_cost = ANGLE_CHANGE_WEIGHT * max_angle_change / ANGLE_NORM

    combined_distances_cost = CONE_DIST_WEIGHT * std_combined_distances / CONE_DIST_NORM

    track_width_cost = TRACK_WIDTH_WEIGHT * std_track_width / TRACK_WIDTH_NORM

    segment_length_cost = SEGMENT_LENGTH_WEIGHT * std_segment_length / SEGMENT_LENGTH_NORM

    path_length_cost = PATH_LENGTH_WEIGHT * path_length_penalty / PATH_LENGTH_NORM

    total_cost = angle_cost + track_width_cost + \
        path_length_cost + combined_distances_cost + segment_length_cost
    print(angle_cost, combined_distances_cost, track_width_cost, segment_length_cost, path_length_cost)
    return total_cost, track_widths


def select_best_path(valid_paths, node_positions, cones, midpoints_to_edge):
    '''
    finds the best path
    '''
    best_path = None
    best_path_track_width = None
    best_cost = float('inf')
    for path in valid_paths:
        cost, track_width = compute_path_cost(
            path, node_positions, cones, midpoints_to_edge)
        if cost < best_cost:
            best_cost = cost
            best_path = path
            best_path_track_width = track_width

    if (best_path is None):
        return None, None
    # when the best path has been selected, calculate the track
    return best_path, best_path_track_width


if __name__ == '__main__':
    cones = cones

    cones_for_triangulation = []
    for cone in cones:
        cones_for_triangulation.append([cone[0], cone[1]])

    cones_for_triangulation = np.array(cones_for_triangulation)

    start = time.time()
    G, midpoints, start_node, midpoint_to_edge = build_midpoint_graph(
        cones_for_triangulation, np.array([0, 0]), 0)

    node_pos = nx.get_node_attributes(G, 'pos')
    valid_paths = find_all_paths_bfs(
        G, start_node, midpoints, cones_for_triangulation)
    best_path, best_cost = select_best_path(
        valid_paths, node_pos, cones_for_triangulation, midpoint_to_edge)
    print(time.time() - start)

    # for path in valid_paths:
    #     plt.triplot(cones[:, 0], cones[:, 1], Delaunay(
    #         cones_for_triangulation).simplices, alpha=0.3)
    #     plt.scatter(cones[:, 0], cones[:, 1], color='red', label="Cones")

    #     # Plot midpoints
    #     for node, pos in node_pos.items():
    #         plt.scatter(pos[0], pos[1], color='blue')

    #     path_coords = np.array([node_pos[node] for node in path])
    #     plt.plot(path_coords[:, 0], path_coords[:, 1],
    #              linestyle="--", alpha=0.7)
    #     plt.axis('equal')
    #     plt.scatter(0, 0, color="green")

    #     print(compute_path_cost(path, node_pos,
    #           cones_for_triangulation, midpoint_to_edge))
    #     print()
    #     plt.show()

    plt.triplot(cones[:, 0], cones[:, 1], Delaunay(
        cones_for_triangulation).simplices, alpha=0.3)
    plt.scatter(cones[:, 0], cones[:, 1], color='red', label="Cones")
    plt.scatter(0, 0, color="green")
    # Plot midpoints
    for node, pos in node_pos.items():
        plt.scatter(pos[0], pos[1], color='blue')

    print("BEST PATH")
    path_coords = np.array([node_pos[node] for node in best_path])
    plt.plot(path_coords[:, 0], path_coords[:, 1], linestyle="--", alpha=0.7)
    plt.show()
    print(compute_path_cost(best_path, node_pos,
          cones_for_triangulation, midpoint_to_edge))


# cone in the middle of the track thing

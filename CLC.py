#Back tracker - r^2

import numpy as np
import math
from typing import List, Tuple, Set, Dict, Optional
from typing import Deque
from collections import deque
from helpers import Point2D
from GeoConstraints import GeometricConstraints
from SearchHeruistics import SearchHeuristics
from width_calculations import calculate_widths

#finds all possible lane boundaries, the step prior to feeding it through a binary classifier to pick the best one
class CLCEnumerator:
    def __init__(self,
                 constraints: GeometricConstraints,
                 heuristics: SearchHeuristics,
                 max_steps: int = 2500):
        self.constraints = constraints
        self.heuristics = heuristics
        self.max_steps = max_steps

    def enumerate_candidates(self,
                             points: List[Point2D],
                             graph: Dict[int, Set[int]],
                             start_left: int,
                             start_right: int,
                             M_fixed_init: Optional[List[Tuple[float, float]]] = None
                             ) -> List[Dict]:
        
        # ---- Local helpers ---------------------------------------------------
        def path_ok_and_bt(left_path: List[int],
                           right_path: List[int],
                           M_fixed_prev: Optional[List[Tuple[float, float]]]):

            '''
            The first part of the code is determining whether or not constraints are violated
            Either constraints can be violated, but can still be backtracked (and fixed during the backtracking)
            Constraints can also be violated with no way of them getting fixed during backtracking
            Or both can pass
            '''
            # 1. segment turning-angle + spacing per side (fast, local)
            left_angle_ok  = self.constraints.check_segment_constraint(left_path, points)
            right_angle_ok = self.constraints.check_segment_constraint(right_path, points)
            if not (left_angle_ok and right_angle_ok):
                # hopeless: exceeding phi_max can't be fixed later
                return False, False, M_fixed_prev

            left_len_ok  = self.constraints.check_length_constraint(left_path, points)
            right_len_ok = self.constraints.check_length_constraint(right_path, points)
            if not (left_len_ok and right_len_ok):
                # hopeless: spacing > d_max is a no‑go
                return False, False, M_fixed_prev

            # 2. polygon simplicity (skip bridge edge inside the checker)
            poly_pass, poly_bt = self.constraints.check_polygon_constraint(left_path, right_path, points)
            if not poly_pass and not poly_bt:
                return False, False, M_fixed_prev

            # 3. lane width (online, with fixed vs mutable)
            M_fixed_new, M_mut, width_pass, width_bt = calculate_widths(
                left_path, right_path, points, M_fixed_prev,
                wmin=self.constraints.w_min, wmax=self.constraints.w_max
            )
            if not width_pass and not width_bt:
                # fixed violation or too‑short mutable → hopeless
                return False, False, M_fixed_new

            # Combine results:
            passed = (left_angle_ok and right_angle_ok and
                      left_len_ok and right_len_ok and
                      poly_pass and width_pass)
            
            #whether we should continue backtracking
            should_keep = (poly_bt and width_bt)

            # If constraints haven't passed yet but it's potentially recoverable,
            # signal "continue backtracking" by returning (False, True).
            if not passed and should_keep:
                return False, True, M_fixed_new

            return True, True, M_fixed_new

        def neighbors_of(idx: int, visited: Set[int]) -> Set[int]:
            """Graph neighbors that are not yet visited on this path."""
            return {n for n in graph.get(idx, set()) if n not in visited}

        # ---- DFS state -------------------------------------------------------
        candidates: List[Dict] = []


        # ---- DFS loop --------------------------------------------------------
        stack : Deque = [([start_left], [start_right], set([start_left]), set([start_right]), None, 0, None )]

        while stack:
            left_path, right_path, visL, visR, M_fixed_prev, steps, undo_info = stack.pop()

            # ---- UNDO visited when backtracking ----
            if undo_info:
                side, node = undo_info
                if side == "L" and node in visL:
                    visL.remove(node)
                elif side == "R" and node in visR:
                    visR.remove(node)

            if steps > self.max_steps:
                continue

            left_unvisited  = neighbors_of(left_path[-1], visL)
            right_unvisited = neighbors_of(right_path[-1], visR)

            left_next  = self.heuristics.next_vertex_decider(left_path,  left_unvisited,  points, graph)
            right_next = self.heuristics.next_vertex_decider(right_path, right_unvisited, points, graph)

            if left_next is None and right_next is None:
                passed, keep, M_fixed_new = path_ok_and_bt(left_path, right_path, M_fixed_prev)
                if passed:
                    candidates.append({"left_path": left_path,
                                    "right_path": right_path,
                                    "M_fixed": M_fixed_new})
                continue

            if left_next is None and right_next is not None:
                side_to_extend = 1
            elif right_next is None and left_next is not None:
                side_to_extend = 0
            else:
                side_to_extend = self.heuristics.left_right_decider(
                    left_path, right_path, left_next, right_next, points
                )

            for trial in ([side_to_extend], [1-side_to_extend]):
                s = trial[0]

                if s == 0 and left_next is not None:
                    new_left  = left_path + [left_next]
                    new_right = right_path
                    visL.add(left_next)
                    undo_info = ("L", left_next)
                    new_visL, new_visR = visL, visR
                elif s == 1 and right_next is not None:
                    new_left  = left_path
                    new_right = right_path + [right_next]
                    visR.add(right_next)  # mark visited
                    undo_info = ("R", right_next)
                    new_visL, new_visR = visL, visR
                else:
                    continue

                passed, keep, M_fixed_new = path_ok_and_bt(new_left, new_right, M_fixed_prev)
                if not passed and not keep:
                    # undo immediately if branch is hopeless
                    if undo_info[0] == "L":
                        visL.remove(undo_info[1])
                    else:
                        visR.remove(undo_info[1])
                    continue

                stack.append((new_left, new_right, new_visL, new_visR, M_fixed_new, steps+1, undo_info))

                if passed:
                    candidates.append({"left_path": new_left,
                                    "right_path": new_right,
                                    "M_fixed": M_fixed_new})



        return candidates


# Code that is used for constructing a graph of points. 
# It is undirected and all points within d_max of the query point are connected to the graph
# Parameterss:
#  points: the point map given to us by perceptions lidar scan
#  d_max: the search radius from a query opoint
# Returns: returns adjency list of our graph
def construct_search_graph(points: List[Point2D], d_max: float = 5.5) -> Dict[int, Set[int]]:
    graph = {i: set() for i in range(len(points))}
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = (points[i] - points[j]).norm()
            if distance <= d_max:
                graph[i].add(j)
                graph[j].add(i)
                
    return graph

# Finding starting vertices to start our DFS search
# Parameters:
#  points: points given to us by Perceptions Lidar Scan
#  car_pos: the position of our car
#  car_heading: the heading of our car
#  max_radius: the radius that the starting points have to be in
# Returns: tuple where the first element is the graph index of left point 
#          the second element is the graph index of the right point
'''
improvements to be made in the future: could use a more robust dot product approach to find 
symmettry but in practice what we have currently should be fine
'''
def find_starting_vertices(points: List[Point2D], car_pos: Point2D, 
                          car_heading: Point2D, max_radius: float = 2.0) -> Tuple[int, int]:
    candidates = []
    for i, point in enumerate(points):
        distance = (point - car_pos).norm()
        if distance <= max_radius:
            direction = point - car_pos
            angle = math.atan2(direction.y, direction.x) - math.atan2(car_heading.y, car_heading.x)
            angle = (angle + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]
            
            candidates.append((i, distance, angle))
    
    if not candidates:
        raise ValueError("No starting vertices found within radius")
    
    # Find left (positive angle) and right (negative angle) starting points
    left_candidates = [(i, d, a) for i, d, a in candidates if a > 0]
    right_candidates = [(i, d, a) for i, d, a in candidates if a < 0]
    
    if not left_candidates or not right_candidates:
        raise ValueError("No symmetric start points found within radius")
    
    # Choose most symmetric pair
    best_pair = None
    best_score = float("inf")

    # weights (tunable) (angle symmetry most important)
    w_ang, w_distbal, w_total = 1.0, 0.5, 0.1

    for iL, dL, aL in left_candidates:
        for iR, dR, aR in right_candidates:
            angle_sym   = abs(aL + aR)      # want ~0 (left + right angles cancel)
            dist_balance = abs(dL - dR)     # want similar radii
            total_dist   = dL + dR          # mild bias toward closer pair
            score = w_ang*angle_sym + w_distbal*dist_balance + w_total*total_dist
            if score < best_score:
                best_score = score
                best_pair = (iL, iR)

    if best_pair is None:
        # fallback (shouldn't happen if both lists non-empty)
        best_left  = min(left_candidates,  key=lambda x: x[1])
        best_right = min(right_candidates, key=lambda x: x[1])
        return best_left[0], best_right[0]

    return best_pair[0], best_pair[1]

points = ([[9.03206651, -3.6854952, -0.33961914], [9.66560302, -6.81808691, -0.32954925], [9.38951545, -9.39995367, -0.27203662], [5.81633507, -4.95950717, -0.27163835],
[6.07295484, -6.78853607, -0.2706291 ], [2.68753045, -2.1297958 , -0.14568515]])
car_pos = Point2D(6, 2)

car_heading = Point2D(math.cos(math.radians(-90)), math.sin(math.radians(-90)))

for i in range(len(points)):
    points[i] = Point2D(points[i][0], points[i][1])

# Example wiring:
constraints = GeometricConstraints(w_min=2.5, w_max=7.0, d_max=5, phi_max=70.0)
heuristics  = SearchHeuristics(car_heading=car_heading)

graph = construct_search_graph(points, d_max=5.5)
start_L, start_R = 5, 0

enumerator = CLCEnumerator(constraints, heuristics, max_steps=2500)
cands = enumerator.enumerate_candidates(points, graph, start_L, start_R, M_fixed_init=None)

import matplotlib.pyplot as plt

def show_candidates(candidates, points):
    """
    Step through candidates one by one.
    When you close a figure window, the next candidate will display.
    """
    for idx, cand in enumerate(candidates):
        left_path = [points[i] for i in cand["left_path"]]
        right_path = [points[i] for i in cand["right_path"]]

        plt.figure()
        plt.title(f"Candidate {idx+1}/{len(candidates)}")

        # Plot all points
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        plt.scatter(xs, ys, c='blue', s=20, label="All cones")

        # Plot left path
        lx = [p.x for p in left_path]
        ly = [p.y for p in left_path]
        plt.plot(lx, ly, 'r-o', label="Left path")

        # Plot right path
        rx = [p.x for p in right_path]
        ry = [p.y for p in right_path]
        plt.plot(rx, ry, 'b-o', label="Right path")

        # Optionally close the polygon
        if left_path and right_path:
            poly_x = lx + rx[::-1] + [lx[0]]
            poly_y = ly + ry[::-1] + [ly[0]]
            plt.plot(poly_x, poly_y, 'g--', alpha=0.5)

        plt.legend()
        plt.axis("equal")

        # Show this candidate — execution pauses until window is closed
        plt.show()

show_candidates(cands, points)
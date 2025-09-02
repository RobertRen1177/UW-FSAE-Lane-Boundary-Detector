# File by Robert Ren - r^2
# robert71@uw.edu or 425-553-5361 for questions
# All Math explained in documentation
import math
from typing import List, Tuple, Set, Dict, Optional
from helpers import Point2D

Param = float                # index-based param: i + lambda, with lambda in [0,1)
Match = Tuple[float, float]  # (u on left polyline, v on right polyline)

# Evaluates point on poly line by splitting t into integer and float between 0 and 1
def _poly_eval(points: List[Point2D], path: List[int], t: Param) -> Point2D:
    n = len(path)
    if n == 0:
        raise ValueError("Empty path")
    if t <= 0:
        return points[path[0]]
    if t >= n - 1:
        return points[path[-1]]
    i = int(math.floor(t))
    lam = t - i
    p0 = points[path[i]]
    p1 = points[path[i+1]]
    return Point2D(p0.x * (1.0 - lam) + p1.x * lam,
                   p0.y * (1.0 - lam) + p1.y * lam)

# Given line segment AB, find closest point on the segment (in terms of a float between 0 and 1)
# to point X
def _closest_point_on_segment(A: Point2D, B: Point2D, X: Point2D) -> Tuple[float, Point2D]:
    ux, uy = (B.x - A.x), (B.y - A.y)
    vx, vy = (X.x - A.x), (X.y - A.y)
    denom = ux*ux + uy*uy
    if denom == 0.0:
        return 0.0, A
    t = max(0.0, min(1.0, (vx*ux + vy*uy) / denom))
    return t, Point2D(A.x + ux*t, A.y + uy*t)

# Given Edge A0 <-> A1 and Edge B0 <-> B1
# Uses derivative optimization to get two points on the line that are the shortest distance between them
# returns (sa, sb, Pa, Pb) where sa and sb are the parameter along A and B respectively
# and Pa and Pb are the actual points
def _closest_points_between_segments(A0: Point2D, A1: Point2D,
                                     B0: Point2D, B1: Point2D) -> Tuple[float, float, Point2D, Point2D]:
    ux, uy = (A1.x - A0.x), (A1.y - A0.y)
    vx, vy = (B1.x - B0.x), (B1.y - B0.y)
    wx, wy = (A0.x - B0.x), (A0.y - B0.y)

    a = ux*ux + uy*uy
    b = ux*vx + uy*vy
    c = vx*vx + vy*vy
    d = ux*wx + uy*wy
    e = vx*wx + vy*wy
    D = a*c - b*b
    EPS = 1e-12

    if D < EPS:
        # Lines almost parallel: clamp sa = 0 and project B onto A
        sa = 0.0
        sb = max(0.0, min(1.0, e / c if c > EPS else 0.0))
    else:
        sa = (b*e - c*d) / D
        sb = (a*e - b*d) / D
        # clamp to [0,1] with corrections
        if sa < 0.0:
            sa = 0.0
            sb = max(0.0, min(1.0, e / c if c > EPS else 0.0))
        elif sa > 1.0:
            sa = 1.0
            sb = max(0.0, min(1.0, (e + b) / c if c > EPS else 0.0))

        if sb < 0.0:
            sb = 0.0
            sa = max(0.0, min(1.0, -d / a if a > EPS else 0.0))
        elif sb > 1.0:
            sb = 1.0
            sa = max(0.0, min(1.0, (b - d) / a if a > EPS else 0.0))

    Pa = Point2D(A0.x + ux*sa, A0.y + uy*sa)
    Pb = Point2D(B0.x + vx*sb, B0.y + vy*sb)
    return sa, sb, Pa, Pb


# Given point i on query path, finds parameter of the point on the target_path that is the shortest distance
def _match_vertex_to_polyline(query_path: List[int], i: int,
                              target_path: List[int],
                              points: List[Point2D]) -> float:
    X = points[query_path[i]]
    best_d2 = float('inf')
    best_param = 0.0
    for j in range(len(target_path) - 1):
        A = points[target_path[j]]
        B = points[target_path[j+1]]
        t, Q = _closest_point_on_segment(A, B, X)
        dx, dy = (X.x - Q.x), (X.y - Q.y)
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best_param = j + t
    return best_param

# Given point i on the query finds, finds the parameter on the target line that is closest to [i, i + 1]
# Returns a tuple where the first element is the parameter for the query path and second is the parameter for the target
def _match_segment_to_polyline(query_path: List[int], i: int,
                               target_path: List[int],
                               points: List[Point2D]) -> Tuple[float, float]:
    A0 = points[query_path[i]]
    A1 = points[query_path[i+1]]
    best_d2 = float('inf')
    best_s = 0.0
    best_t = 0.0
    best_j = 0
    for j in range(len(target_path) - 1):
        B0 = points[target_path[j]]
        B1 = points[target_path[j+1]]
        s, t, Pa, Pb = _closest_points_between_segments(A0, A1, B0, B1)
        dx, dy = (Pa.x - Pb.x), (Pa.y - Pb.y)
        d2 = dx*dx + dy*dy
        if d2 < best_d2:
            best_d2 = d2
            best_s = s
            best_t = t
            best_j = j
    return (i + best_s, best_j + best_t)

# Returns Matches 
# Returns a list of tuples, each tuples first element is the left paths parameter 
# the second element is the right paths parameter
def calculate_M(left: List[int], right: List[int],
                points: List[Point2D],
                k: int, s: int,
                u_start: float, v_start: float,
                eps: float = 1e-9) -> List[Tuple[float, float]]:
    matches: List[Tuple[float, float]] = []

    # Choose which side is the query and where to start indexing
    if s == 0:
        # Query = LEFT, Target = RIGHT
        query_path, target_path = left, right
        i0 = max(0, int(math.floor(u_start)))  # start from last-fixed u index on LEFT
        if k == 0:
            rng = range(i0, len(query_path))            # vertices i
        else:
            rng = range(i0, max(i0, len(query_path)-1)) # segments i in [i0, len-2]
        for i in rng:
            if k == 0:
                u = float(i)
                v = _match_vertex_to_polyline(query_path, i, target_path, points)
            else:
                u, v = _match_segment_to_polyline(query_path, i, target_path, points)
            # (u,v) already in (u_on_L, v_on_R) form here
            # Enforce lexicographic not-before (u_start,v_start)
            if (u < u_start - eps) or (abs(u - u_start) <= eps and v < v_start - eps):
                continue
            matches.append((u, v))

    else:
        # Query = RIGHT, Target = LEFT  → flip outputs to (u_on_L, v_on_R)
        query_path, target_path = right, left
        i0 = max(0, int(math.floor(v_start)))  # start from last-fixed v index on RIGHT
        if k == 0:
            rng = range(i0, len(query_path))
            for i in rng:
                v = float(i)
                u = _match_vertex_to_polyline(query_path, i, target_path, points)  # param on LEFT
                # flip to (u_on_L, v_on_R)
                if (u < u_start - eps) or (abs(u - u_start) <= eps and v < v_start - eps):
                    continue
                matches.append((u, v))
        else:
            rng = range(i0, max(i0, len(query_path)-1))
            for i in rng:
                # match segment i of RIGHT vs LEFT → returns (param_on_R, param_on_L) if called naively
                # but our helper returns (s_on_query, t_on_target) in local coords.
                v_local, u_local = _match_segment_to_polyline(query_path, i, target_path, points)
                u, v = u_local, v_local  # flip to (u_on_L, v_on_R)
                if (u < u_start - eps) or (abs(u - u_start) <= eps and v < v_start - eps):
                    continue
                matches.append((u, v))

    # Keep sorted (u,v) as required downstream
    matches.sort(key=lambda uv: (uv[0], uv[1]))
    return matches


# =========================
# Lane-width computation (online: fixed vs mutable)
# =========================

def _split_fixed_mutable(M_sorted: List[Match], lenL: int, lenR: int, eps: float = 1e-9):
    """
    Split M into (new_fixed, new_mutable) at the first match that touches an end:
    u == lenL-1  OR  v == lenR-1.
    All BEFORE that index become fixed; the rest remain mutable.
    """
    touch_idx = None
    for idx, (u, v) in enumerate(M_sorted):
        if abs(u - (lenL - 1)) <= eps or abs(v - (lenR - 1)) <= eps:
            touch_idx = idx
            break
    if touch_idx is None:
        return [], M_sorted
    return M_sorted[:touch_idx], M_sorted[touch_idx:]

def _widths_from_matches(M_list: List[Match],
                         left: List[int], right: List[int],
                         points: List[Point2D]) -> List[float]:
    w = []
    for (u, v) in M_list:
        Lu = _poly_eval(points, left, u)
        Rv = _poly_eval(points, right, v)
        dx, dy = (Lu.x - Rv.x), (Lu.y - Rv.y)
        w.append(math.hypot(dx, dy))
    return w

# Constraint checker 
def calculate_widths(left: List[int], right: List[int],
                     points: List[Point2D],
                     M_fixed: Optional[List[Tuple[float, float]]],
                     wmin: float = 2.5, wmax: float = 6.5,
                     eps: float = 1e-9
                     ) -> Tuple[List[Tuple[float, float]],
                                List[Tuple[float, float]],
                                List[float], List[float], bool]:
    """
    Returns:
      (M_fixed_new, M_mutable, widths_fixed, widths_mutable, violates_fixed)
    """
    # 1) resume point from existing fixed matches
    if not M_fixed:
        u_start, v_start = 0.0, 0.0
        M_fixed_curr: List[Tuple[float, float]] = []
    else:
        u_start, v_start = M_fixed[-1]
        M_fixed_curr = list(M_fixed)

    # 2) compute the four matching sets, starting at (u_start, v_start)
    M0 = calculate_M(left, right, points, k=0, s=0, u_start=u_start, v_start=v_start)  # L-vertex → R
    M1 = calculate_M(left, right, points, k=1, s=0, u_start=u_start, v_start=v_start)  # L-seg    → R
    M2 = calculate_M(left, right, points, k=0, s=1, u_start=u_start, v_start=v_start)  # R-vertex → L (flipped)
    M3 = calculate_M(left, right, points, k=1, s=1, u_start=u_start, v_start=v_start)  # R-seg    → L (flipped)
    Mall = sorted(M0 + M1 + M2 + M3, key=lambda uv: (uv[0], uv[1]))

    # 3) safety filter (lexicographic) in case any emission slipped before (u_start, v_start)
    def _not_before(m):
        u, v = m
        return (u > u_start + eps) or (abs(u - u_start) <= eps and v >= v_start - eps)

    M_new = [m for m in Mall if _not_before(m)] if M_fixed_curr else Mall

    # 4) split into fixed vs mutable at first end-touch
    new_fixed, M_mutable = _split_fixed_mutable(M_new, len(left), len(right), eps=eps)
    M_fixed_curr.extend(new_fixed)

    # 5) widths
    widths_fixed   = _widths_from_matches(M_fixed_curr, left, right, points)
    widths_mutable = _widths_from_matches(M_mutable,     left, right, points)

    # 6) Determine backtracking decider
    should_backtrack = True
    constraints_passed = True

    if (any((w <= wmin) or (w >= wmax) for w in widths_fixed)):
        return M_fixed_curr, M_mutable, False, False
    
    for width in widths_mutable:
        if(width > wmax):
            constraints_passed = False
        elif(width < wmin):
            return M_fixed_curr, M_mutable, False, False

    return M_fixed_curr, M_mutable, constraints_passed, should_backtrack



"""Analytic geometry primitives for a centred, object-local Cuboid."""

from __future__ import annotations

import numpy as np


def cuboid_feature_mask(point_object: np.ndarray, half_size: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    """Return the ``(+x,-x,+y,-y,+z,-z)`` surface-feature mask."""
    point, half = np.asarray(point_object, dtype=np.float64), np.asarray(half_size, dtype=np.float64)
    if point.shape[-1] != 3 or half.shape != (3,):
        raise ValueError("point must end in 3 and half_size must have shape (3,)")
    output = np.zeros(point.shape[:-1] + (6,), dtype=bool)
    for axis in range(3):
        output[..., 2 * axis] = np.abs(point[..., axis] - half[axis]) <= tolerance
        output[..., 2 * axis + 1] = np.abs(point[..., axis] + half[axis]) <= tolerance
    return output


def cuboid_signed_distance(points_object: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    """Positive outside / zero surface / negative inside signed box distance."""
    points, half = np.asarray(points_object, dtype=np.float64), np.asarray(half_size, dtype=np.float64)
    if points.shape[-1] != 3 or half.shape != (3,):
        raise ValueError("points must end in 3 and half_size must have shape (3,)")
    q = np.abs(points) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def project_to_cuboid_surface(points_object: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    """Project each point to the nearest Cuboid boundary, deterministically."""
    points, half = np.asarray(points_object, dtype=np.float64), np.asarray(half_size, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or half.shape != (3,) or np.any(half <= 0):
        raise ValueError("points must be [N,3] and half_size must contain three positive values")
    projected = np.clip(points, -half, half)
    inside = np.all(np.abs(points) < half, axis=1)
    if np.any(inside):
        rows = np.flatnonzero(inside)
        axes = np.argmin(half[None, :] - np.abs(points[rows]), axis=1)
        projected[rows, axes] = np.where(points[rows, axes] < 0.0, -half[axes], half[axes])
    return projected


def mesh_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Return deterministic, unique, undirected triangle-mesh edges."""
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (F,3)")
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def cuboid_crossing_contacts(vertices_object: np.ndarray, edges: np.ndarray, half_size: np.ndarray,
                             distances: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return exact segment/Cuboid boundary intersections for SDF sign crossings."""
    vertices, edge_array = np.asarray(vertices_object, dtype=np.float64), np.asarray(edges, dtype=np.int64)
    half = np.asarray(half_size, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("vertices must be [V,3] and edges must be [E,2]")
    values = cuboid_signed_distance(vertices, half) if distances is None else np.asarray(distances, dtype=np.float64)
    chosen = edge_array[values[edge_array[:, 0]] * values[edge_array[:, 1]] < 0.0]
    if not len(chosen):
        return chosen, np.empty((0, 3), dtype=np.float64)
    p0, p1 = vertices[chosen[:, 0]], vertices[chosen[:, 1]]
    delta = p1 - p0
    lower, upper = np.full(len(chosen), -np.inf), np.full(len(chosen), np.inf)
    for axis in range(3):
        moving = np.abs(delta[:, axis]) > 1e-15
        first = np.divide(-half[axis] - p0[:, axis], delta[:, axis], out=np.zeros(len(chosen)), where=moving)
        second = np.divide(half[axis] - p0[:, axis], delta[:, axis], out=np.zeros(len(chosen)), where=moving)
        lower = np.where(moving, np.maximum(lower, np.minimum(first, second)), lower)
        upper = np.where(moving, np.minimum(upper, np.maximum(first, second)), upper)
    t = np.where(values[chosen[:, 0]] < 0.0, upper, lower)
    return chosen, np.clip(p0 + t[:, None] * delta, -half, half)

"""Five-fingertip MANO-to-Cuboid contact extraction.

The public contact taxonomy deliberately excludes the MANO wrist, palm, MCP,
and PIP regions.  Each output frame has one optional target for thumb3,
index3, middle3, ring3, and pinky3 only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .geometry import (
    cuboid_crossing_contacts, cuboid_feature_mask, cuboid_signed_distance,
    mesh_edges_from_faces, project_to_cuboid_surface,
)

FINGERTIP_NAMES = ("thumb3", "index3", "middle3", "ring3", "pinky3")
_FINGERTIP_REGION_IDS = np.array((3, 6, 9, 12, 15), dtype=np.int16)
# Native MANO LBS order: wrist, index, middle, pinky, ring, thumb.
_LBS_JOINT_TO_REGION = np.array((0, 4, 5, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12, 1, 2, 3), dtype=np.int16)
_CONTACT_FIELDS = {
    "contact_points_object", "contact_points_hand", "contact_normals", "contact_valid",
    "contact_regions", "contact_links_hand", "contact_links_valid",
    "contact_points_object_local", "contact_region_points_object_local", "contact_region_valid",
    "contact_region_feature_mask", "contact_target_points_object_local", "contact_target_valid",
    "contact_target_feature_mask",
}


def mano_lbs_vertex_regions(lbs_weights: np.ndarray) -> np.ndarray:
    """Classify vertices by their dominant native MANO LBS joint."""
    weights = np.asarray(lbs_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != 16 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("MANO LBS weights must be finite, non-negative [V,16]")
    return _LBS_JOINT_TO_REGION[np.argmax(weights, axis=1)]


@dataclass(frozen=True)
class ContactFrame:
    points_object: np.ndarray
    valid: np.ndarray
    feature_mask: np.ndarray


def _aggregate_patch(points: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks = cuboid_feature_mask(points, half)
    counts = masks.sum(axis=0)
    face = int(np.argmax(counts))
    if counts[face] > len(points) / 2:
        axis, positive = divmod(face, 2)
        target = np.clip(np.mean(points, axis=0), -half, half)
        target[axis] = half[axis] if positive == 0 else -half[axis]
    else:
        target = points[int(np.argmin(np.linalg.norm(points[:, None] - points[None, :], axis=-1).sum(axis=1)))].copy()
    return target, cuboid_feature_mask(target, half)


def _primary_mesh_patch(points: np.ndarray, vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Keep the largest connected candidate patch; earliest point breaks ties."""
    if not len(points):
        return points
    parent = np.arange(len(points))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index
    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    owners: dict[int, list[int]] = {}
    for index, (left, right) in enumerate(vertices):
        owners.setdefault(int(left), []).append(index)
        owners.setdefault(int(right), []).append(index)
    for group in owners.values():
        for index in group[1:]:
            union(group[0], index)
    for left, right in edges:
        for a in owners.get(int(left), ()):
            for b in owners.get(int(right), ()):
                union(a, b)
    components: dict[int, list[int]] = {}
    for index in range(len(points)):
        components.setdefault(find(index), []).append(index)
    selected = min(components.values(), key=lambda indices: (-len(indices), indices[0]))
    return points[np.asarray(selected)]


class HumanContactExtractor:
    """Extract object-surface targets from the five distal MANO segments only."""

    def __init__(self, *, half_size: np.ndarray, vertex_regions: np.ndarray, mano_edges: np.ndarray,
                 threshold_m: float = 0.002):
        self.half_size = np.asarray(half_size, dtype=np.float64)
        self.vertex_regions = np.asarray(vertex_regions, dtype=np.int16)
        self.mano_edges = np.asarray(mano_edges, dtype=np.int64)
        self.threshold_m = float(threshold_m)
        if self.half_size.shape != (3,) or np.any(self.half_size <= 0) or self.vertex_regions.ndim != 1:
            raise ValueError("invalid Cuboid size or MANO regions")
        if self.mano_edges.ndim != 2 or self.mano_edges.shape[1] != 2 or self.threshold_m <= 0:
            raise ValueError("invalid MANO edges or contact threshold")
        self.tip_mask = np.isin(self.vertex_regions, _FINGERTIP_REGION_IDS)
        self.tip_edges = self.mano_edges[self.tip_mask[self.mano_edges].any(axis=1)]
        self.contact_vertex_ids = np.unique(np.concatenate((np.flatnonzero(self.tip_mask), self.tip_edges.ravel())))
        self.tip_edges_by_region = tuple(self.tip_edges[(self.vertex_regions[self.tip_edges] == region).any(axis=1)]
                                          for region in _FINGERTIP_REGION_IDS)

    @classmethod
    def from_mano_reference(cls, *, half_size: np.ndarray, reference_vertices: np.ndarray,
                            lbs_weights: np.ndarray, faces: np.ndarray, threshold_m: float = 0.002) -> "HumanContactExtractor":
        regions = mano_lbs_vertex_regions(lbs_weights)
        if np.asarray(reference_vertices).shape != (len(regions), 3):
            raise ValueError("reference MANO vertices and LBS weights disagree")
        return cls(half_size=half_size, vertex_regions=regions, mano_edges=mesh_edges_from_faces(faces), threshold_m=threshold_m)

    def extract_frame(self, *, vertices_world: np.ndarray, object_translation_world: np.ndarray,
                      rotation_world_object: np.ndarray) -> ContactFrame:
        vertices = np.asarray(vertices_world, dtype=np.float64)
        translation, rotation = np.asarray(object_translation_world, dtype=np.float64), np.asarray(rotation_world_object, dtype=np.float64)
        if vertices.shape != (len(self.vertex_regions), 3) or translation.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("MANO vertices or object pose have an invalid shape")
        object_vertices = np.empty_like(vertices)
        object_vertices[self.contact_vertex_ids] = (vertices[self.contact_vertex_ids] - translation) @ rotation
        distance = np.full(len(vertices), np.nan)
        distance[self.contact_vertex_ids] = cuboid_signed_distance(object_vertices[self.contact_vertex_ids], self.half_size)
        tip_ids = np.flatnonzero(self.tip_mask)
        tip_distances = distance[self.tip_mask]
        near_ids = tip_ids[(tip_distances >= 0) & (tip_distances < self.threshold_m) &
                           ~np.isclose(tip_distances, self.threshold_m, rtol=0, atol=1e-12)]
        crossing_edges, crossing_points = cuboid_crossing_contacts(object_vertices, self.tip_edges, self.half_size, distance)
        crossing_inside = (np.where(distance[crossing_edges[:, 0]] < 0, crossing_edges[:, 0], crossing_edges[:, 1])
                           if len(crossing_edges) else np.empty(0, dtype=np.int64))
        candidates = np.concatenate((project_to_cuboid_surface(object_vertices[near_ids], self.half_size), crossing_points))
        regions = np.concatenate((self.vertex_regions[near_ids], self.vertex_regions[crossing_inside])).astype(np.int16)
        generators = np.concatenate((np.column_stack((near_ids, near_ids)), crossing_edges)).astype(np.int64)
        targets, valid, features = (np.full((5, 3), np.nan), np.zeros(5, dtype=bool), np.zeros((5, 6), dtype=bool))
        for target_id, region in enumerate(_FINGERTIP_REGION_IDS):
            selected = regions == region
            patch = _primary_mesh_patch(candidates[selected], generators[selected], self.tip_edges_by_region[target_id])
            if len(patch):
                targets[target_id], features[target_id] = _aggregate_patch(patch, self.half_size)
                valid[target_id] = True
        return ContactFrame(targets, valid, features)


def make_enriched_archive(source_arrays: dict[str, np.ndarray], metadata: dict, *, targets_world: np.ndarray,
                          targets_object: np.ndarray, target_valid: np.ndarray, target_features: np.ndarray,
                          threshold_m: float, source_frame_start: int, source_frame_stop: int) -> dict[str, np.ndarray]:
    """Return a cropped DexterHand archive enriched with viewer-compatible contacts."""
    output = {key: value.copy() for key, value in source_arrays.items() if key not in _CONTACT_FIELDS and key != "metadata"}
    target_valid = np.asarray(target_valid, dtype=bool)
    output.update(
        contact_points_object=np.asarray(targets_world, dtype=np.float32),
        contact_points_hand=np.asarray(targets_world, dtype=np.float32),
        contact_valid=target_valid,
        contact_regions=np.broadcast_to(np.arange(5, dtype=np.int16), target_valid.shape).copy(),
        contact_links_hand=np.asarray(targets_world, dtype=np.float32),
        contact_links_valid=target_valid.copy(),
        contact_points_object_local=np.asarray(targets_object, dtype=np.float32),
        contact_region_points_object_local=np.asarray(targets_object, dtype=np.float32),
        contact_region_valid=target_valid.copy(),
        contact_region_feature_mask=np.asarray(target_features, dtype=bool),
        contact_target_points_object_local=np.asarray(targets_object, dtype=np.float32),
        contact_target_valid=target_valid.copy(),
        contact_target_feature_mask=np.asarray(target_features, dtype=bool),
    )
    output_metadata = copy.deepcopy(metadata)
    for key in tuple(output_metadata):
        if key.startswith("contact_"):
            output_metadata.pop(key)
    output_metadata.update(
        contact_region_names=FINGERTIP_NAMES,
        contact_threshold_m=float(threshold_m),
        contact_method="analytic_cuboid_sdf_fingertip_patch_target",
        contact_excluded_regions=("wrist", "palm", "mcp", "pip", "non_tip"),
        contact_points_frame="viewer-only world-frame mirror of explicit fingertip targets",
        source_frame_start=int(source_frame_start), source_frame_stop=int(source_frame_stop),
    )
    # DexterCap stores segment starts as source-frame offsets.  Rebase them so
    # its viewer does not skip or mis-segment the newly cropped trajectory.
    if output_metadata.get("index_init_frame") is not None:
        original_starts = np.asarray(output_metadata["index_init_frame"], dtype=np.int64)
        rebased = [0] + [int(frame - source_frame_start) for frame in original_starts
                         if source_frame_start < frame < source_frame_stop]
        output_metadata["index_init_frame"] = sorted(set(rebased))
    output["metadata"] = np.array(output_metadata, dtype=object)
    return output

from pathlib import Path

import numpy as np

from dexterhand_contact.cli import crop_source, default_output_path, frame_window
from dexterhand_contact.contact import FINGERTIP_NAMES, HumanContactExtractor, make_enriched_archive
from dexterhand_contact.geometry import cuboid_crossing_contacts, cuboid_signed_distance


def test_time_range_is_stop_exclusive_and_crops_every_frame_aligned_field():
    assert frame_window(60.0, 0.01, 0.05, 10) == (1, 3)
    arrays = {"poses": np.arange(40).reshape(10, 4), "static": np.ones((2, 3))}
    cropped = crop_source(arrays, 10, 1, 3)
    assert cropped["poses"].shape == (2, 4)
    assert cropped["static"].shape == (2, 3)


def test_default_output_is_under_the_project_output_directory():
    path = default_output_path(Path("/data/demo.npz"), 2.0, 6.5)
    assert path.name == "demo-contact-2-6.5s.npz"
    assert path.parent.name == "output"


def test_extractor_rejects_palm_and_keeps_only_a_fingertip_target():
    extractor = HumanContactExtractor(
        half_size=np.full(3, 0.05), vertex_regions=np.array([0, 3, 3], dtype=np.int16),
        mano_edges=np.array([[0, 1], [1, 2]], dtype=np.int64),
    )
    frame = extractor.extract_frame(
        vertices_world=np.array([[0.051, 0.0, 0.0], [0.0, 0.0, 0.0], [0.06, 0.0, 0.0]]),
        object_translation_world=np.zeros(3), rotation_world_object=np.eye(3),
    )
    np.testing.assert_array_equal(frame.valid, [True, False, False, False, False])
    np.testing.assert_allclose(frame.points_object[0], [0.05, 0.0, 0.0])


def test_crossing_contact_is_exact_and_deep_interior_vertex_is_not_fabricated():
    vertices = np.array([[-0.06, 0.0, 0.0], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    selected, points = cuboid_crossing_contacts(vertices, edges, np.full(3, 0.05), cuboid_signed_distance(vertices, np.full(3, 0.05)))
    np.testing.assert_array_equal(selected, [[0, 1]])
    np.testing.assert_allclose(points, [[-0.05, 0.0, 0.0]])


def test_enriched_archive_has_only_five_fingertip_semantics_and_crop_provenance():
    source = {"hand_poses": np.zeros((2, 45), dtype=np.float32)}
    valid = np.array([[True, False, False, False, False], [False] * 5])
    archive = make_enriched_archive(
        source, {"fps": 60.0}, targets_world=np.full((2, 5, 3), np.nan),
        targets_object=np.full((2, 5, 3), np.nan), target_valid=valid,
        target_features=np.zeros((2, 5, 6), dtype=bool), threshold_m=0.002,
        source_frame_start=30, source_frame_stop=32,
    )
    metadata = archive["metadata"].item()
    assert metadata["contact_region_names"] == FINGERTIP_NAMES
    assert metadata["contact_excluded_regions"] == ("wrist", "palm", "mcp", "pip", "non_tip")
    assert (metadata["source_frame_start"], metadata["source_frame_stop"]) == (30, 32)

"""Command line interface for cropping and extracting DexterHand contacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .compat import enable_legacy_mano_compatibility
from .contact import HumanContactExtractor, make_enriched_archive


REQUIRED_FIELDS = {"metadata", "hand_translations", "hand_orientations_axis_angle", "hand_poses", "hand_shapes", "object_translations", "object_orientations_quat_xyzw"}


def resolve_mano_model_path(path: Path) -> Path:
    supplied = path.expanduser().resolve()
    if supplied.is_file() and supplied.name == "MANO_RIGHT.pkl":
        return supplied
    for candidate in (supplied / "MANO_RIGHT.pkl", supplied / "mano" / "MANO_RIGHT.pkl"):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"MANO_RIGHT.pkl not found under {supplied}")


def _mano_model(path: Path, batch_size: int, device: str):
    enable_legacy_mano_compatibility()
    import smplx
    return smplx.create(str(path), model_type="mano", flat_hand_mean=True, is_rhand=True, use_pca=False, batch_size=batch_size).to(device)


def frame_window(fps: float, start_s: float, end_s: float | None, count: int) -> tuple[int, int]:
    if not np.isfinite(fps) or fps <= 0 or start_s < 0 or (end_s is not None and end_s < start_s):
        raise ValueError("invalid FPS or time interval")
    start = min(count, max(0, int(np.ceil(start_s * fps))))
    stop = count if end_s is None else min(count, max(start, int(np.ceil(end_s * fps))))
    if start == stop:
        raise ValueError("time interval selects no source frames")
    return start, stop


def crop_source(arrays: dict[str, np.ndarray], count: int, start: int, stop: int) -> dict[str, np.ndarray]:
    """Crop all frame-aligned arrays while preserving scalar/static source data."""
    output: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key == "metadata":
            continue
        output[key] = value[start:stop].copy() if value.ndim and len(value) == count else value.copy()
    return output


def extract_targets(arrays: dict[str, np.ndarray], metadata: dict, *, mano_path: Path, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    count = len(arrays["hand_translations"])
    half_size = np.asarray(metadata["object_size"], dtype=np.float64) / 2
    invalid = np.asarray(metadata["invalid_point_value"])
    reference_model = _mano_model(mano_path, 1, device)
    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=device)
    reference = reference_model(global_orient=zeros3, transl=zeros3,
                                hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
                                betas=torch.zeros((1, 10), dtype=torch.float32, device=device))
    extractor = HumanContactExtractor.from_mano_reference(
        half_size=half_size, reference_vertices=reference.vertices[0].detach().cpu().numpy(),
        lbs_weights=reference_model.lbs_weights.detach().cpu().numpy(), faces=np.asarray(reference_model.faces),
    )
    targets_object = np.full((count, 5, 3), np.nan, dtype=np.float32)
    targets_world = np.full_like(targets_object, np.nan)
    target_valid = np.zeros((count, 5), dtype=bool)
    target_features = np.zeros((count, 5, 6), dtype=bool)
    for start in tqdm(range(0, count, batch_size), desc="Extracting fingertip contacts"):
        stop = min(start + batch_size, count)
        size = stop - start
        model = _mano_model(mano_path, size, device)
        tensor = lambda key: torch.as_tensor(arrays[key][start:stop], dtype=torch.float32, device=device)
        result = model(global_orient=torch.zeros((size, 3), dtype=torch.float32, device=device),
                       transl=torch.zeros((size, 3), dtype=torch.float32, device=device),
                       hand_pose=tensor("hand_poses"), betas=tensor("hand_shapes"))
        vertices = result.vertices.detach().cpu().numpy()
        rotations = Rotation.from_rotvec(arrays["hand_orientations_axis_angle"][start:stop]).as_matrix()
        vertices = np.einsum("nij,nvj->nvi", rotations, vertices) + arrays["hand_translations"][start:stop, None]
        for local, frame in enumerate(range(start, stop)):
            translation = arrays["object_translations"][frame]
            if np.allclose(translation, invalid):
                continue
            rotation = Rotation.from_quat(arrays["object_orientations_quat_xyzw"][frame]).as_matrix()
            contact = extractor.extract_frame(vertices_world=vertices[local], object_translation_world=translation,
                                              rotation_world_object=rotation)
            targets_object[frame], target_valid[frame], target_features[frame] = contact.points_object, contact.valid, contact.feature_mask
            targets_world[frame, contact.valid] = contact.points_object[contact.valid] @ rotation.T + translation
    return targets_world, targets_object, target_valid, target_features


def extract_command(args: argparse.Namespace) -> int:
    source, destination = Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if source == destination:
        raise ValueError("--output must be a new file; input is never overwritten")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    with np.load(source, allow_pickle=True) as archive:
        missing = REQUIRED_FIELDS.difference(archive.files)
        if missing:
            raise ValueError(f"DexterHand archive misses required fields: {sorted(missing)}")
        raw = {key: archive[key].copy() for key in archive.files}
    metadata = raw["metadata"].item()
    if str(metadata.get("hand_side", "")).lower() != "right":
        raise ValueError("this release supports right-hand DexterHand data only; convert left data before extraction")
    count = len(raw["hand_translations"])
    if any(len(raw[field]) != count for field in REQUIRED_FIELDS - {"metadata"}):
        raise ValueError("required DexterHand arrays do not share a frame count")
    start, stop = frame_window(float(metadata["fps"]), args.start, args.end, count)
    cropped = crop_source(raw, count, start, stop)
    targets_world, targets_object, valid, features = extract_targets(
        cropped, metadata, mano_path=resolve_mano_model_path(Path(args.mano_model_path)), batch_size=args.batch_size, device=args.device)
    enriched = make_enriched_archive(cropped, metadata, targets_world=targets_world, targets_object=targets_object,
                                     target_valid=valid, target_features=features, threshold_m=0.002,
                                     source_frame_start=start, source_frame_stop=stop)
    enriched["metadata"].item().update(source_time_start_s=float(args.start), source_time_stop_s=float(args.end) if args.end is not None else count / float(metadata["fps"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary, **enriched)
    os.replace(temporary, destination)
    print(f"Cropped source frames: [{start}, {stop}) at {metadata['fps']} FPS")
    print(f"Fingertip targets: {int(valid.sum())} across {int(valid.any(axis=1).sum())} frames")
    print(f"Saved processed DexterHand trajectory: {destination}")
    return 0


def visualize_command(args: argparse.Namespace) -> int:
    root = Path(args.dextercap_root).expanduser().resolve()
    viewer = root / "Dataset" / "visualize.py"
    if not viewer.is_file():
        raise FileNotFoundError(f"DexterCap viewer not found: {viewer}")
    command = [sys.executable, str(viewer), "--data_path", str(Path(args.input).expanduser().resolve()), "--hand_side", "right"]
    if args.start is not None:
        command += ["--start", str(args.start)]
    if args.end is not None:
        command += ["--end", str(args.end)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(command, env=environment, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop a DexterHand trajectory and extract five fingertip/Cuboid contacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="crop a right-hand DexterHand NPZ and append contact fields")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--start", type=float, default=0.0, help="inclusive time in seconds")
    extract.add_argument("--end", type=float, help="exclusive time in seconds")
    extract.add_argument("--mano-model-path", default=os.environ.get("MANO_MODEL_PATH", ""), required="MANO_MODEL_PATH" not in os.environ)
    extract.add_argument("--batch-size", type=int, default=128)
    extract.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(handler=extract_command)
    viewer = subparsers.add_parser("visualize", help="open a processed NPZ with DexterCap's Rerun viewer")
    viewer.add_argument("--input", required=True)
    viewer.add_argument("--dextercap-root", default=os.environ.get("DEXTERCAP_ROOT", ""),
                        required="DEXTERCAP_ROOT" not in os.environ)
    viewer.add_argument("--start", type=float)
    viewer.add_argument("--end", type=float)
    viewer.set_defaults(handler=visualize_command)
    args = parser.parse_args()
    try:
        raise SystemExit(args.handler(args))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))

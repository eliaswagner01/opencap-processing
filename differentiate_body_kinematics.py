"""Derive accelerations from an OpenSim BodyKinematics velocity file.

BodyKinematics can report NaN accelerations when AnalyzeTool is driven only by
a coordinates file. The corresponding velocities are valid. Global velocities
can be differentiated directly; body-local linear velocities additionally need
the rotating-frame transport term omega cross velocity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parent.parent
    / "Data"
    / "OpenSimData"
    / "OpenCap"
    / "Accelarations"
    / "results"
)
VELOCITY_MARKER = "_BodyKinematics_vel_"
ACCELERATION_MARKER = "_BodyKinematics_acc_from_velocity_"


def read_storage(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        end_header = next(
            index for index, line in enumerate(lines) if line.strip() == "endheader"
        )
    except StopIteration as exc:
        raise ValueError(f"No endheader line found in {path}") from exc

    if end_header + 1 >= len(lines):
        raise ValueError(f"No column labels found in {path}")

    header = lines[:end_header]
    labels = lines[end_header + 1].split()
    rows = [line.split() for line in lines[end_header + 2 :] if line.strip()]
    data = np.asarray(rows, dtype=float)

    if data.ndim != 2 or data.shape[1] != len(labels):
        raise ValueError(
            f"Expected {len(labels)} columns in {path}, found shape {data.shape}"
        )
    if data.shape[0] < 3:
        raise ValueError("At least three samples are required for differentiation")
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("Time values must be strictly increasing")

    return header, labels, data


def values_are_in_degrees(header: list[str]) -> bool:
    metadata = next(
        (line for line in header if line.strip().startswith("inDegrees=")),
        "inDegrees=yes",
    )
    return metadata.split("=", maxsplit=1)[1].strip().lower() == "yes"


def derive_accelerations(
    labels: list[str], data: np.ndarray, body_local: bool, in_degrees: bool
) -> np.ndarray:
    time = data[:, 0]
    acceleration = np.empty_like(data)
    acceleration[:, 0] = time
    acceleration[:, 1:] = np.gradient(
        data[:, 1:], time, axis=0, edge_order=2
    )

    if not body_local:
        return acceleration

    label_indices = {label: index for index, label in enumerate(labels)}
    bodies = {
        label.removesuffix("_X")
        for label in labels
        if label.endswith("_X")
    }
    for body in bodies:
        linear_labels = [f"{body}_{axis}" for axis in "XYZ"]
        angular_labels = [f"{body}_O{axis.lower()}" for axis in "XYZ"]
        if not all(label in label_indices for label in linear_labels + angular_labels):
            continue

        linear_indices = [label_indices[label] for label in linear_labels]
        angular_indices = [label_indices[label] for label in angular_labels]
        linear_velocity = data[:, linear_indices]
        angular_velocity = data[:, angular_indices]
        if in_degrees:
            angular_velocity = np.deg2rad(angular_velocity)

        acceleration[:, linear_indices] += np.cross(
            angular_velocity, linear_velocity
        )

    return acceleration


def write_accelerations(
    path: Path,
    source: Path,
    source_header: list[str],
    labels: list[str],
    data: np.ndarray,
    body_local: bool,
) -> None:
    in_degrees = values_are_in_degrees(source_header)
    acceleration = derive_accelerations(
        labels, data, body_local, in_degrees
    )

    in_degrees_metadata = next(
        (line for line in source_header if line.strip().startswith("inDegrees=")),
        "inDegrees=yes",
    )
    frame_description = "body-local" if body_local else "global"
    header = [
        f"Accelerations derived from {frame_description} BodyKinematics velocities",
        "version=1",
        f"nRows={acceleration.shape[0]}",
        f"nColumns={acceleration.shape[1]}",
        in_degrees_metadata.strip(),
        "",
        f"Source velocity file: {source}",
        "Linear accelerations are in m/s^2.",
        "Angular accelerations are in deg/s^2 when inDegrees=yes.",
        "Second-order finite differences were applied to the velocity data.",
        (
            "Body-local linear accelerations include the omega cross velocity "
            "transport term."
            if body_local
            else "Global linear accelerations are direct velocity derivatives."
        ),
        "",
        "endheader",
        "\t".join(labels),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(header) + "\n")
        np.savetxt(output, acceleration, delimiter="\t", fmt="%.10f")


def process_file(velocity_file: Path, output_file: Path) -> None:
    source_header, labels, data = read_storage(velocity_file)
    body_local = "bodylocal" in velocity_file.name.lower()
    write_accelerations(
        output_file, velocity_file, source_header, labels, data, body_local
    )

    _, _, acceleration = read_storage(output_file)
    if not np.isfinite(acceleration).all():
        raise ValueError(f"Non-finite values found in {output_file}")


def process_results_root(results_root: Path) -> list[Path]:
    velocity_files = sorted(results_root.rglob(f"*{VELOCITY_MARKER}*.sto"))
    if not velocity_files:
        raise FileNotFoundError(
            f"No BodyKinematics velocity files found below {results_root}"
        )

    output_files = []
    for velocity_file in velocity_files:
        output_file = velocity_file.with_name(
            velocity_file.name.replace(
                VELOCITY_MARKER, ACCELERATION_MARKER, 1
            )
        )
        process_file(velocity_file, output_file)
        output_files.append(output_file)
        print(f"Created {output_file}")
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Differentiate one OpenSim BodyKinematics velocity STO file, or "
            "process every velocity file in the session results directory."
        )
    )
    parser.add_argument("velocity_file", type=Path, nargs="?")
    parser.add_argument("output_file", type=Path, nargs="?")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Results directory used in batch mode.",
    )
    args = parser.parse_args()

    if args.velocity_file is None and args.output_file is None:
        outputs = process_results_root(args.results_root)
        print(f"Processed {len(outputs)} BodyKinematics velocity files.")
        return

    if args.velocity_file is None or args.output_file is None:
        parser.error(
            "velocity_file and output_file must either both be provided or both omitted"
        )

    process_file(args.velocity_file, args.output_file)
    print(f"Created {args.output_file}")


if __name__ == "__main__":
    main()

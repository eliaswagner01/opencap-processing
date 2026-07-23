"""Rerun OpenCap JRA with loads applied to and expressed in parent bodies."""

from pathlib import Path
import sys

import numpy as np
import yaml


SESSION_ID = "3375ffbc-daeb-4a43-b4f7-ac9899cd4c71"


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError("Required input does not exist: {}".format(path))
    return path


def run_trial(compute_mcf, base_dir, model_path, output_dir, case,
              contact_spheres):
    trial_name = output_dir.name
    states_path = require_file(
        output_dir
        / "kinematics_activations_{}_{}.mot".format(trial_name, case)
    )
    grf_path = require_file(
        output_dir / "GRF_{}_{}.mot".format(trial_name, case)
    )
    forces_path = require_file(
        output_dir / "forces_{}_{}.mot".format(trial_name, case)
    )
    trajectories_path = require_file(output_dir / "optimaltrajectories.npy")

    trajectories = np.load(trajectories_path, allow_pickle=True).item()
    if case not in trajectories:
        raise KeyError("Case '{}' is not in {}".format(case, trajectories_path))
    coordinate_speeds = trajectories[case]["coordinate_speeds"].T

    compute_mcf(
        str(base_dir / "OpenSimPipeline"),
        str(output_dir),
        str(model_path),
        str(states_path),
        str(states_path),
        str(grf_path),
        grfType="sphere",
        contactSides=["right", "left"],
        contactSpheres=contact_spheres,
        muscleForceFilePath=str(forces_path),
        pathReserveGeneralizedForces=str(forces_path),
        Qds=coordinate_speeds,
        replaceMuscles=True,
        outputFileSuffix=case,
        outputParentFrameJRA=True,
    )

    parent_result = (
        output_dir
        / (
            "results_JRAforMCF_{}_parent_"
            "JointReactionAnalysisParent_ReactionLoads.sto"
        ).format(case)
    )
    require_file(parent_result)
    return parent_result


def main():
    base_dir = Path(__file__).resolve().parent
    joint_reaction_dir = base_dir / "OpenSimPipeline" / "JointReaction"
    sys.path.insert(0, str(joint_reaction_dir))

    from computeJointLoading import computeMCF

    open_sim_data = base_dir / "Data" / SESSION_ID / "OpenSimData"
    model_dir = open_sim_data / "Model"
    dynamics_dir = open_sim_data / "Dynamics"
    force_map_path = require_file(model_dir / "ExternalFunction" / "F_map.npy")
    force_map = np.load(force_map_path, allow_pickle=True).item()["GRFs"]
    contact_spheres = {
        "right": force_map["rightContactSpheres"],
        "left": force_map["leftContactSpheres"],
        "bodies": {
            "right": force_map["rightContactSphereBodies"],
            "left": force_map["leftContactSphereBodies"],
        },
    }

    completed = []
    skipped = []
    failed = []
    for output_dir in sorted(path for path in dynamics_dir.iterdir()
                             if path.is_dir()):
        trial_name = output_dir.name
        setup_paths = sorted(output_dir.glob("Setup_*.yaml"))
        if not setup_paths:
            skipped.append((trial_name, ["Setup_<case>.yaml"]))
            continue

        for setup_path in setup_paths:
            case = setup_path.stem[len("Setup_"):]
            with setup_path.open("r", encoding="utf-8") as stream:
                setup = yaml.safe_load(stream)
            model_name = setup.get("OpenSimModel", "LaiUhlrich2022")
            model_path = model_dir / "{}_scaled_adjusted.osim".format(
                model_name)
            required_trial_files = [
                output_dir
                / "kinematics_activations_{}_{}.mot".format(trial_name, case),
                output_dir / "GRF_{}_{}.mot".format(trial_name, case),
                output_dir / "forces_{}_{}.mot".format(trial_name, case),
                output_dir / "optimaltrajectories.npy",
                model_path,
            ]
            missing = [path.name for path in required_trial_files
                       if not path.is_file()]
            run_name = "{} [{}]".format(trial_name, case)
            if missing:
                skipped.append((run_name, missing))
                continue

            print("\nRerunning parent-frame JRA for {}...".format(run_name))
            try:
                parent_result = run_trial(
                    computeMCF, base_dir, model_path, output_dir, case,
                    contact_spheres)
                completed.append(parent_result)
                print("Written: {}".format(parent_result))
            except Exception as exc:
                failed.append((run_name, exc))
                print("FAILED {}: {}".format(run_name, exc))

    print("\nCompleted {} parent-frame JRA file(s).".format(len(completed)))
    for parent_result in completed:
        print("  {}".format(parent_result))
    for trial_name, missing in skipped:
        print("Skipped {} (missing: {}).".format(
            trial_name, ", ".join(missing)))
    if failed:
        details = "; ".join("{}: {}".format(name, error)
                            for name, error in failed)
        raise RuntimeError("Parent-frame JRA failed for {}".format(details))


if __name__ == "__main__":
    main()

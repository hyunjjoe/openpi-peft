import argparse
import collections
import json
import logging
import math
import pathlib
import pickle

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # same as training

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the simple OpenPI evaluation loop for a selection of LIBERO tasks."
    )
    parser.add_argument(
        "--eval",
        dest="eval_only",
        action="store_true",
        help="Run the evaluation loop without saving per-episode obs/trajectory pickles.",
    )
    parser.add_argument(
        "--task-summary-dest",
        type=pathlib.Path,
        default=None,
        help=(
            "If provided, write the task success rate summary to this JSON path. "
        ),
    )
    return parser.parse_args()

def _quat2axisangle(quat):
    # Copied from robosuite
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    host, port = "0.0.0.0", 8000
    task_suite_name = "libero_90"  # e.g., "libero_spatial", "libero_10", "libero_90"
    # task_ids = [12, 14, 28, 29, 38]                 # small subset of tasks
    # task_ids = list(range(39, 90))
    task_ids = [1, 2, 3, 12, 14, 28, 29, 38] # eval set
    num_trials_per_task = 50
    num_steps_wait = 10                 # wait for objects to settle

    # Max horizon per suite (copied from main.py)
    if task_suite_name == "libero_spatial":
        max_steps = 220
    elif task_suite_name == "libero_object":
        max_steps = 280
    elif task_suite_name == "libero_goal":
        max_steps = 300
    elif task_suite_name == "libero_10":
        max_steps = 520
    elif task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict[task_suite_name]()

    total_episodes, total_successes = 0, 0
    episode_summaries: list[dict] = []
    per_task_stats: dict[int, dict[str, int]] = {}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        num_init_states = len(init_states)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=7)

        logging.info(f"=== Task {task_id}: {task_description} ===")

        # Experiment name and base directories for this task
        exp_name = f"{task_suite_name}_tasks1-90_trials{num_trials_per_task}"
        obs_dir = pathlib.Path("data/eval/pi05_libero_topk1/libero_final_obs") / exp_name
        traj_dir = pathlib.Path("data/eval/pi05_libero_topk1/libero_simple_trajectories") / exp_name
        if not args.eval_only:
            obs_dir.mkdir(parents=True, exist_ok=True)
            traj_dir.mkdir(parents=True, exist_ok=True)

        task_episodes, task_successes = 0, 0
        base_seed = 7
        for episode_idx in range(num_trials_per_task):
            # -------- Resume logic: skip episodes that already have a saved trajectory --------
            existing_suffix = None
            existing_traj_path = None
            if not args.eval_only:
                for suffix_candidate in ("success", "failure"):
                    candidate = traj_dir / f"traj_task{task_id}_ep{episode_idx}_{suffix_candidate}.pkl"
                    if candidate.exists():
                        existing_suffix = suffix_candidate
                        existing_traj_path = candidate
                        break

            if existing_suffix is not None:
                # Episode already completed in a previous run: load minimal info and skip rollout.
                with existing_traj_path.open("rb") as f:
                    episode_data = pickle.load(f)

                obs_path = obs_dir / f"final_obs_task{task_id}_ep{episode_idx}_{existing_suffix}.pkl"

                success_bool = existing_suffix == "success"
                task_episodes += 1
                total_episodes += 1
                if success_bool:
                    task_successes += 1
                    total_successes += 1

                episode_summaries.append(
                    {
                        "task_suite_name": task_suite_name,
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "success": bool(success_bool),
                        "prompt": str(episode_data.get("prompt", task_description)),
                        "final_obs_path": str(obs_path),
                        "trajectory_path": str(existing_traj_path),
                    }
                )

                logging.info(
                    f"[resume] Skipping already completed task {task_id} episode {episode_idx}: "
                    f"success={success_bool}, "
                    f"task_success_rate={task_successes/task_episodes:.2f}, "
                    f"total_success_rate={total_successes/total_episodes:.2f}"
                )
                continue

            # -------- Fresh rollout for this episode --------
            # Reset environment at the start of each episode to clear any terminated state.
            if episode_idx < num_init_states:
                env.reset()
                obs = env.set_init_state(init_states[episode_idx])
            else:
                # After exhausting the official LIBERO init states, sample new ones via env.reset()
                env.seed(base_seed + episode_idx)
                obs = env.reset()

            action_plan = collections.deque()
            
            # Trajectory data: store obs, state, action at each step
            episode_data = None
            if not args.eval_only:
                episode_data = {
                    "prompt": str(task_description),
                    "observations": [],
                    "states": [],
                    "actions": [],
                }

            t = 0
            done = False
            final_obs = obs  # Track the last observation
            while t < max_steps + num_steps_wait:
                # Wait for objects to fall for first few steps
                if t < num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    final_obs = obs  # Update final obs even during warmup
                    # If the environment terminates during the waiting phase (e.g., early failure),
                    # stop the episode to avoid stepping a terminated env.
                    if done:
                        break
                    t += 1
                    continue

                # Preprocess images (rotate 180 deg + resize + pad)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, 224, 224)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, 224, 224)
                )

                # Construct state vector (what we send to the policy)
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                # Query policy when action chunk is exhausted
                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    # Replan every 5 steps
                    replan_steps = 5
                    assert len(action_chunk) >= replan_steps
                    action_plan.extend(action_chunk[:replan_steps])

                action = action_plan.popleft()
                
                # Store observation, state, action for this timestep
                if episode_data is not None:
                    episode_data["observations"].append(obs)
                    episode_data["states"].append(state)
                    episode_data["actions"].append(np.asarray(action, dtype=np.float32))
                
                obs, reward, done, info = env.step(action.tolist())
                final_obs = obs  # Update final observation after each step
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            if not args.eval_only:
                obs_path = obs_dir / f"final_obs_task{task_id}_ep{episode_idx}_{suffix}.pkl"
                with obs_path.open("wb") as f:
                    pickle.dump(final_obs, f)

                # Save trajectory (observations, states, actions, prompt) as pickle
                traj_path = traj_dir / f"traj_task{task_id}_ep{episode_idx}_{suffix}.pkl"
                with traj_path.open("wb") as f:
                    pickle.dump(episode_data, f)

                # Record summary for this episode
                episode_summaries.append(
                    {
                        "task_suite_name": task_suite_name,
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "success": bool(done),
                        "prompt": str(task_description),
                        "final_obs_path": str(obs_path),
                        "trajectory_path": str(traj_path),
                    }
                )

            logging.info(
                f"Task {task_id} episode {episode_idx}: success={done}, "
                f"task_success_rate={task_successes/task_episodes:.2f}, "
                f"total_success_rate={total_successes/total_episodes:.2f}"
            )

        per_task_stats[int(task_id)] = {
            "episodes": task_episodes,
            "successes": task_successes,
        }

    logging.info(f"Final total success rate: {total_successes/total_episodes:.2f} over {total_episodes} episodes.")
    
    task_success_rates = []
    for task_id in task_ids:
        stats = per_task_stats.get(int(task_id), {"episodes": 0, "successes": 0})
        episodes = stats["episodes"]
        successes = stats["successes"]
        success_rate = successes / episodes if episodes else 0.0
        task_success_rates.append(
            {
                "task_id": int(task_id),
                "episodes": episodes,
                "successes": successes,
                "success_rate": success_rate,
            }
        )
    # Save a JSON summary for all episodes in this run.
    summary = {
        "task_suite_name": task_suite_name,
        "task_ids": list(task_ids),
        "num_trials_per_task": num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "episodes": episode_summaries,
        "task_success_rates": task_success_rates,
    }
    summary_base_dir = pathlib.Path("data/libero_simple_trajectories")
    default_summary_path = summary_base_dir / f"{task_suite_name}_tasks{'-'.join(map(str, task_ids))}_trials{num_trials_per_task}_summary.json"
    if args.task_success_summary_path is None:
        summary_base_dir.mkdir(parents=True, exist_ok=True)
        summary_path = default_summary_path
    else:
        summary_path = args.task_success_summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Wrote task success summary to %s", summary_path)
    
if __name__ == "__main__":
    main()
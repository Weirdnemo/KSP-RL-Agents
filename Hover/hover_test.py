import argparse
import csv
import os
import time
import numpy as np

from stable_baselines3 import PPO

# Import your environment
from ksp_hover_env import HoverEnv  # <-- change if filename/class differs


# ------------ Termination classification helper ------------
def classify_termination(alt_history, reached_hover, crashed_flag, timed_out_flag, max_steps_flag):
    """
    Derive a human-readable termination reason using recorded trajectory
    and env flags.
    """
    if crashed_flag:
        return "crash"
    if not reached_hover:
        # never got to hover band; check what happened
        if timed_out_flag or max_steps_flag:
            return "timeout_before_500"
        # overshoot? (shouldn't if never hit 500, but just in case)
        if np.max(alt_history) > 600:
            return "overshoot_before_500"
        return "ended_before_500"
    # reached hover band at least once
    if np.max(alt_history) > 600:
        return "overshoot_>600"
    if np.min(alt_history[alt_history >= 500]) < 350:
        return "fell_below_350_after_500"
    if timed_out_flag:
        return "timeout"
    if max_steps_flag:
        return "step_limit"
    return "unknown"


# ------------ Single episode runner ------------
def run_hover_episode(model, env, target_alt=500.0, hover_band_narrow=10.0, hover_band_wide=50.0,
                      render=False, csv_writer=None):
    """
    Run ONE evaluation episode with the trained model.
    Collect detailed telemetry and return metrics dict.
    """
    obs, _ = env.reset()
    alt, vspd, fuel = obs

    start_wall = time.time()
    start_fuel = fuel

    alt_hist = []
    vspd_hist = []
    fuel_hist = []
    thr_hist = []

    within_narrow = 0
    within_wide = 0
    reached_hover = False

    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated):
        # model action
        action, _ = model.predict(obs, deterministic=True)
        print(f"[DEBUG] Agent action: {action}")
        throttle = float(action[0])

        # env step
        obs, reward, done, truncated, info = env.step(action)

        alt, vspd, fuel = obs
        total_reward += reward
        steps += 1

        # track hover
        if alt >= target_alt:
            reached_hover = True
        if abs(alt - target_alt) <= hover_band_narrow:
            within_narrow += 1
        if abs(alt - target_alt) <= hover_band_wide:
            within_wide += 1

        # record telemetry
        alt_hist.append(alt)
        vspd_hist.append(vspd)
        fuel_hist.append(fuel)
        thr_hist.append(throttle)

        if render:
            print(
                f"step={steps:03d} thr={throttle:.2f} "
                f"alt={alt:7.1f}m vspd={vspd:7.2f}m/s fuel={fuel:5.2f} "
                f"r={reward:7.2f} done={done} trunc={truncated}"
            )

        if csv_writer is not None:
            csv_writer.writerow([steps, throttle, alt, vspd, fuel, reward, done, truncated])

    # episode ended
    end_wall = time.time()
    wall_dur = end_wall - start_wall
    sim_dur = steps * env.step_sleep if hasattr(env, "step_sleep") else None

    alt_hist = np.array(alt_hist) if alt_hist else np.array([alt])
    vspd_hist = np.array(vspd_hist) if vspd_hist else np.array([vspd])
    fuel_hist = np.array(fuel_hist) if fuel_hist else np.array([fuel])
    thr_hist = np.array(thr_hist) if thr_hist else np.array([0.0])

    max_alt = float(np.max(alt_hist))
    min_alt = float(np.min(alt_hist))
    final_alt = float(alt_hist[-1])
    peak_up_v = float(np.max(vspd_hist))
    peak_down_v = float(np.min(vspd_hist))  # negative
    hover_time_narrow = within_narrow * (env.step_sleep if hasattr(env, "step_sleep") else 0.0)
    hover_time_wide = within_wide * (env.step_sleep if hasattr(env, "step_sleep") else 0.0)
    fuel_used = max(0.0, start_fuel - fuel_hist[-1])

    # Determine termination cause using env state and our logs
    crashed_flag = (final_alt <= 1.0) or (not env._has_command_module())
    timed_out_flag = (time.time() - env.start_time) > env.max_time if hasattr(env, "max_time") else False
    max_steps_flag = steps >= getattr(env, "max_steps", 10**9)

    reason = classify_termination(
        alt_history=alt_hist,
        reached_hover=reached_hover,
        crashed_flag=crashed_flag,
        timed_out_flag=timed_out_flag,
        max_steps_flag=max_steps_flag,
    )

    # Info from env (if it sent something)
    max_alt_from_env = info.get("max_altitude", None)
    if max_alt_from_env is not None:
        max_alt = max_alt_from_env  # trust env's tracked max if present

    metrics = {
        "steps": steps,
        "wall_time_s": wall_dur,
        "sim_time_s": sim_dur,
        "total_reward": total_reward,
        "max_alt_m": max_alt,
        "min_alt_m": min_alt,
        "final_alt_m": final_alt,
        "peak_up_v_mps": peak_up_v,
        "peak_down_v_mps": peak_down_v,
        "hover_time_narrow_s": hover_time_narrow,
        "hover_time_wide_s": hover_time_wide,
        "hover_ratio_narrow": within_narrow / steps if steps > 0 else 0.0,
        "hover_ratio_wide": within_wide / steps if steps > 0 else 0.0,
        "fuel_used_frac": fuel_used,
        "termination_reason": reason,
    }
    return metrics


# ------------ main ------------
def main():
    parser = argparse.ArgumentParser(description="Test PPO hover agent in KSP.")
    parser.add_argument("--model", type=str, default="ppo_hover_agent_final.zip",
                        help="Path to saved PPO model (.zip).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Print per-step telemetry.")
    parser.add_argument("--csv", type=str, default="",
                        help="Optional CSV file to log step telemetry across all episodes.")
    parser.add_argument("--hover", type=float, default=500.0, help="Target hover altitude (meters).")
    parser.add_argument("--narrow", type=float, default=10.0, help="Narrow hover band (+/- m).")
    parser.add_argument("--wide", type=float, default=50.0, help="Wide hover band (+/- m).")
    args = parser.parse_args()

    # Env (must match what model was trained on)
    env = HoverEnv()  # If constructor changed, pass same training args here.

    # Load trained model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    model = PPO.load(args.model, env=env)
    print(f"[INFO] Loaded model: {args.model}")

    # CSV logging (optional)
    csv_f = None
    csv_writer = None
    if args.csv:
        new_file = not os.path.exists(args.csv)
        csv_f = open(args.csv, "a", newline="")
        csv_writer = csv.writer(csv_f)
        if new_file:
            csv_writer.writerow([
                "step", "throttle", "altitude_m", "vertical_speed_mps",
                "fuel_frac", "reward", "done", "truncated"
            ])

    # Run episodes
    summary = []
    for ep in range(1, args.episodes + 1):
        print(f"\n=== TEST EPISODE {ep}/{args.episodes} ===")
        metrics = run_hover_episode(
            model,
            env,
            target_alt=args.hover,
            hover_band_narrow=args.narrow,
            hover_band_wide=args.wide,
            render=args.render,
            csv_writer=csv_writer,
        )
        summary.append(metrics)
        print(
            f"[SUMMARY EP {ep}] time={metrics['wall_time_s']:.1f}s "
            f"reward={metrics['total_reward']:.2f} "
            f"max_alt={metrics['max_alt_m']:.1f}m "
            f"hover±{args.narrow}m={metrics['hover_ratio_narrow']*100:.1f}% "
            f"reason={metrics['termination_reason']}"
        )

    env.close()
    if csv_f:
        csv_f.close()

    # Aggregate summary if >1 episode
    if len(summary) > 1:
        avg_hover = np.mean([m["hover_ratio_narrow"] for m in summary])
        avg_reward = np.mean([m["total_reward"] for m in summary])
        avg_max_alt = np.mean([m["max_alt_m"] for m in summary])
        print("\n=== AGGREGATE SUMMARY ===")
        print(f"Episodes:            {len(summary)}")
        print(f"Avg total reward:    {avg_reward:.2f}")
        print(f"Avg max altitude:    {avg_max_alt:.1f} m")
        print(f"Avg hover% ±{args.narrow}m: {avg_hover*100:.1f}%")
        print("=========================\n")


if __name__ == "__main__":
    main()

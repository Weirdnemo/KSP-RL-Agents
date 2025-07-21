import os
import re
import argparse
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from ksp_hover_env import HoverEnv  # adjust if your env filename/class differs


# --------------------------------------------------
# Utility: find latest checkpoint (ppo_hover_step_*.zip)
# --------------------------------------------------
def find_latest_checkpoint(pattern_prefix="ppo_hover_step_"):
    """
    Return path to checkpoint with largest step count, or None.
    Filenames must look like: ppo_hover_step_0002000.zip
    """
    candidates = []
    for fname in os.listdir("."):
        if fname.startswith(pattern_prefix) and fname.endswith(".zip"):
            m = re.search(r"_(\d+)\.zip$", fname)
            if m:
                step = int(m.group(1))
                candidates.append((step, fname))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]  # filename with max step


# --------------------------------------------------
# Callback: periodic checkpoint + simple episode logging
# --------------------------------------------------
class HoverCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int = 2000, save_prefix="ppo_hover_step_", verbose=1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_prefix = save_prefix
        self._next_save = self.save_freq

    def _on_step(self) -> bool:
        # Episode logging (single-env assumption)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                max_alt = info.get("max_altitude", info.get("max_alt", None))
                if self.verbose > 0:
                    if max_alt is not None:
                        print(f"[EP LOG] max_alt={max_alt:.1f} m")

        # Periodic checkpoint
        if self.num_timesteps >= self._next_save:
            ckpt_name = f"{self.save_prefix}{self.num_timesteps:07d}.zip"
            self.model.save(ckpt_name)
            if self.verbose > 0:
                print(f"[CKPT] Saved {ckpt_name}")
            self._next_save += self.save_freq

        return True


# --------------------------------------------------
# Make environment
# --------------------------------------------------
def make_env():
    env = HoverEnv()        # supply args if you changed constructor
    env = Monitor(env)      # records ep_rew_mean etc. for SB3 logs
    return env


# --------------------------------------------------
# Train (with resume support)
# --------------------------------------------------
def train_hover_agent(
    total_timesteps: int,
    ckpt_freq: int = 2000,
    resume: bool = True,
    final_model_name: str = "ppo_hover_agent_final",
    latest_symlink: str = "ppo_hover_agent_latest.zip",
):
    env = make_env()

    model = None
    loaded_from: Optional[str] = None

    if resume:
        latest = find_latest_checkpoint()
        if latest is not None:
            print(f"[INFO] Resuming from checkpoint: {latest}")
            model = PPO.load(latest, env=env)
            loaded_from = latest

    if model is None:
        # fresh model
        print("[INFO] Starting new PPO hover model.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            n_epochs=10,
        )

    callback = HoverCheckpointCallback(save_freq=ckpt_freq, verbose=1)

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Final save
    model.save(final_model_name)
    print(f"[INFO] Final model saved: {final_model_name}.zip")

    # Also copy/alias as "latest"
    try:
        # SB3 appends ".zip"; ensure filename
        src = f"{final_model_name}.zip"
        if os.path.exists(src):
            # write bytes copy
            with open(src, "rb") as fin, open(latest_symlink, "wb") as fout:
                fout.write(fin.read())
            print(f"[INFO] Latest copy written: {latest_symlink}")
    except Exception as e:
        print(f"[WARN] Could not write latest copy: {e}")

    env.close()


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO hover agent for KSP.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps.")
    parser.add_argument("--ckpt-freq", type=int, default=2000, help="Checkpoint save frequency.")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh; do not resume from checkpoint.")
    args = parser.parse_args()

    train_hover_agent(
        total_timesteps=args.timesteps,
        ckpt_freq=args.ckpt_freq,
        resume=not args.no_resume,
    )

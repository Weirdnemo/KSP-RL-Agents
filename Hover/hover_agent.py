import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ksp_hover_env import HoverEnv


MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 20_000
SAVE_INTERVAL = 2_000  # Save model every 2k steps


class SaveOnStepCallback(BaseCallback):
    """
    Custom callback to save model every SAVE_INTERVAL steps and print episode summaries.
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Save model periodically
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"ppo_hover_agent_{self.num_timesteps}.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"[MODEL SAVE] Saved model to {path}")
        return True

    def _on_rollout_end(self) -> None:
        # Print episode summaries if available
        if "episode" in self.locals:
            ep_info = self.locals["episode"]
            self.episode_count += 1
            print(f"[EP {self.episode_count}] Reward={ep_info['r']:.2f}, Length={ep_info['l']}")


def train_agent(total_steps=TOTAL_TIMESTEPS):
    env = HoverEnv(step_sleep=0.2)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        n_epochs=10
    )

    callback = SaveOnStepCallback(save_freq=SAVE_INTERVAL, save_path=MODEL_DIR, verbose=1)

    print("[INFO] Starting training...")
    model.learn(total_timesteps=total_steps, callback=callback)
    print("[INFO] Training completed.")

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "ppo_hover_agent_final.zip")
    model.save(final_model_path)
    print(f"[MODEL SAVE] Final model saved at {final_model_path}")

    env.close()


if __name__ == "__main__":
    train_agent()

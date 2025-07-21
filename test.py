from ksp_env import KSPEnv

env = KSPEnv()
obs, _ = env.reset()
print("Initial Obs:", obs)
done = False
while not done:
    action = env.action_space.sample()  # Random action for testing
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Alt={obs[0]:.1f}, Reward={reward:.2f}")
    done = terminated or truncated

env.close()

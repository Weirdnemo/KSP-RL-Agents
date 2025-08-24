# 🚀 KSP RL Agents

**Reinforcement Learning agents trained to launch, hover, and land rockets inside Kerbal Space Program (KSP).**

---

## 🌌 Overview

This project explores **autonomous rocket control** in KSP using **Reinforcement Learning (RL)**.
Agents are trained via [kRPC](https://krpc.github.io/krpc/) to interact with KSP in real-time, learning to:

* 🔼 **Launch Agent** → perform smooth vertical ascents
* 🛰 **Hover Agent** → stabilize around a target altitude
* 🔽 **Landing Agent** → execute powered landings with minimal error

All agents are built using **Stable-Baselines3 (PPO)**, with custom KSP environments exposing physics and telemetry.

---

## 📂 Repository Structure

```
KSP-RL-Agents/
│
├── Helper/                    # Utility functions and plotting scripts
│   └── plotting.py
|   └── flight_recorder
|       └── flight_recorder.py
|       └── plot.py
│
├── KSP_Hover_Agent/           # Hover agent training & testing
│   ├── hover_agent.py         # PPO hover agent
│   ├── hover_test.py          # Test scripts for hover agent
│   └── ksp_hover_env.py       # Custom KSP hover environment
│
├── KSP_Landing_Agent/         # Landing agent training & testing
│   ├── ksp_land_env.py        # Custom KSP landing environment
│   ├── test_land_agent.py     # Evaluate trained landing agent
│   └── train_land_agent.py    # Training loop for landing agent
│
├── KSP_Launching_Agent/       # Launching agent training & testing
│   ├── agent.py               # PPO launch agent
│   ├── ksp_env.py             # Custom KSP launch environment
│   └── test.py                # Evaluate trained launch agent
│
├── showcase/                  # Demos and visualizations
│   ├── earth_1.11.mp4         # Demo video: Launching agent
│   ├── moon_1.11.mp4          # Demo video: Landing agent
│   └── flight_dashboard.png
│
├── requirement.txt            # Python dependencies
└── README.md                  # Project description (this file)
```

---

## ⚙️ Installation

1. **Install KSP**

   * Tested with KSP1

2. **Install kRPC mod**

   * Place in `GameData/`, start the server inside KSP

3. **Set up Python environment**

   ```bash
   git clone https://github.com/Weirdnemo/KSP-RL-Agents.git
   cd KSP-RL-Agents
   pip install -r requirement.txt
   ```

Requirements include:

* `stable-baselines3`
* `gym`
* `krpc`
* `numpy`

---

## 🧠 How It Works

* **Environment:** Custom Gym wrappers around KSP via kRPC (altitude, velocity, orientation, fuel states)
* **Agent:** PPO learns throttle, pitch, and burn profiles
* **Rewards:** Shaped to encourage stable control and minimize error

| Agent     | Goal                                | Reward Signal          |
| --------- | ----------------------------------- | ---------------------- |
| Launching | Reach target altitude efficiently   | Time & fuel efficiency |
| Hovering  | Hold altitude without overshoot     | Distance from setpoint |
| Landing   | Touch down safely, minimal velocity | Smoothness + precision |

---

## 🎥 Demos

*(See `/showcase/` for full-resolution videos and captures)*

---

## 📊 Results

* Trained over **tens of thousands of episodes**
* Agents converge to **stable, repeatable behaviors** in all three tasks
* Performance beats naive scripted control (PID-only baselines)

---

## 🔮 Next Steps

* Multi-objective training: launch → hover → land in a single trajectory
* More robust environments (randomized gravity, payloads, wind)
* Compare PPO with SAC, TD3, A2C
* Long-term: full **autonomous orbital insertion** in KSP

---

## 🏷 Credits

* Built by [Weirdnemo](https://github.com/Weirdnemo) over 2–3 months
* Powered by [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) + [kRPC](https://krpc.github.io/krpc/)

---

## 📜 License

Open source under your preferred license (default: MIT).

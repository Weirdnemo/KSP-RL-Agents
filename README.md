    git clone https://github.com/your-username/ksp-hover-agent.git
    cd ksp-hover-agent
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Run

1.  **Start KSP:** Launch Kerbal Space Program.
2.  **Load a vessel:** In KSP, load a rocket capable of vertical flight (e.g., a simple rocket with a single engine and fuel tank).
3.  **Connect kRPC:** Ensure the kRPC server is running within KSP (usually starts automatically with the mod).
4.  **Run the training script:**
    ```bash
    python train.py
    ```
    This will start the training process. The agent will interact with the rocket in KSP.

5.  **Run the evaluation script (after training):**
    ```bash
    python evaluate.py
    # 🚀 KSP Hover Agent

This project trains a reinforcement learning (PPO) agent to control a rocket in Kerbal Space Program (KSP) using the kRPC API.  
The agent learns to reach and hover around a target altitude of **200m** while minimizing overshoot and maintaining stability.

---

## 📹 Demo Video
(youtube)

<p align="center">
  <a href="https://youtu.be/rWP7ViwXaMM" target="_blank">
    <img src="/showcase/Frontface.png" alt="Watch the video" width="70%">
  </a>
</p>

---

## 🖼 Screenshots

### Untrained Rocket
<p align="center">
  <img src="/showcase/untrained.png" alt="Untrained Rocket" width="60%">
</p>

### Trained Rocket
<p align="center">
  <img src="/showcase/trained-16k.png" alt="Trained Rocket" width="60%">
</p>

    python evaluate.py
    ```
    This will run the trained agent in KSP and display its performance.

---

## ⚙️ Project Structure

```
ksp-hover-agent/
├── agents/                 # Contains the PPO agent implementation
├── environments/           # Defines the KSP environment for training
├── models/                 # Stores trained models
├── showcase/               # Demo videos and screenshots
│   ├── Frontface.png
│   ├── untrained.png
│   └── trained-16k.png
├── utils/                  # Utility functions
├── train.py                # Script to train the agent
├── evaluate.py             # Script to evaluate the trained agent
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

*   [kRPC](https://krpc.github.io/krpc/) for providing the API to interact with KSP.
*   [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) for the PPO implementation.

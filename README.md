# RL Player Imitation: Behavioral Simulation via PPO

An advanced Reinforcement Learning project focused on training an intelligent agent to imitate complex player behaviors using **Proximal Policy Optimization (PPO)**. This repository showcases the implementation of custom OpenAI Gym environments, data-driven reward shaping, and behavioral analysis.

## 🚀 Project Overview
The goal of this project is to bridge the gap between raw tracking data and autonomous decision-making. By processing real-world player and ball movements, the agent learns to navigate and react within a custom-built physics environment.

## 🛠 Key Features
- **Custom Gym Environment:** Designed `Player14Env`, a bespoke reinforcement learning environment tailored for high-dimensional state spaces.
- **PPO Implementation:** Leveraged **Stable Baselines3** to implement the PPO algorithm, ensuring stable and efficient policy convergence.
- **Data Engineering Pipeline:** Developed scripts to clean and transform raw `.json` tracker data into structured `.csv` formats for training.
- **Behavioral Analysis:** Included comprehensive evaluation scripts and visual analytics (via Matplotlib & TensorBoard) to monitor reward progress and agent performance.

## 💻 Tech Stack
- **Languages:** Python
- **RL Frameworks:** Stable Baselines3, OpenAI Gym
- **Data Science:** NumPy, Pandas, Matplotlib
- **Logging:** TensorBoard

## 📈 Results
The "Improved Player 14" model demonstrates significant convergence in reward optimization, showing the agent's ability to replicate spatial positioning and movement patterns derived from the initial dataset.

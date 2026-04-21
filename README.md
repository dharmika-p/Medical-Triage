# 🏥 LLM-Guided Safe Reinforcement Learning for Medical Triage under Partial Observability

## 📌 Overview

This project implements a **Safe Reinforcement Learning framework** for medical triage under **partial observability**, enhanced with **LLM-based guidance**.

We compare two RL algorithms:

* **Proximal Policy Optimization (PPO)**
* **Deep Q-Network (DQN)**

The system ensures that **high-risk patients are handled safely** while maximizing overall reward.

---

## 🎯 Objectives

* Model a **medical triage system** as a POMDP
* Integrate **LLM-based safety guidance**
* Compare **PPO vs DQN performance**
* Minimize **unsafe decisions (safety violations)**

---

## ⚙️ Technologies Used

* Python
* Gymnasium (custom environment)
* Stable-Baselines3 (PPO, DQN)
* NumPy, Matplotlib
* HuggingFace Transformers (LLM simulation)

---

## 🧠 Methodology

### 1. Environment (POMDP)

* Patient severity is partially observable
* States include noisy observations
* Actions:

  * `0` → Low priority
  * `1` → Medium priority
  * `2` → High priority

---

### 2. LLM Guidance

A simulated LLM provides safety-aware recommendations:

* Encourages correct triage decisions
* Penalizes unsafe actions (e.g., ignoring high severity)

---

### 3. Safe Reward Design

* Positive reward for correct triage
* Strong penalty for unsafe decisions
* Encourages safe and optimal policies

---

## 📊 Results

### 🔹 PPO vs DQN Performance

* PPO achieves **stable and higher rewards**
* DQN shows **unstable and negative rewards**

### 🔹 Safety

* Both models achieved **0 safety violations**
* Due to strong safety penalty shaping

### 🔹 Key Insight

Even with zero violations:

* **DQN performs poorly** due to inefficient action selection
* **PPO learns balanced and optimal policies**

---

## 📈 Visualizations Included

* Reward Comparison
* Cumulative Reward
* Smoothed Reward
* Safety Violations
* Action Distribution (Bar Graph)

---

## 📌 Final Metrics (Example)

```
PPO Avg Reward: 2.96
DQN Avg Reward: -4.08
PPO Violations: 0
DQN Violations: 0
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install stable-baselines3 gymnasium numpy matplotlib transformers
```

### 2. Run Notebook

Open in Google Colab:

* Upload `MedicalTriage.ipynb`
* Click **Run All**

---

## 📁 Project Structure

```
Medical-Triage/
│
├── MedicalTriage.ipynb
├── README.md
```

---

## 💡 Future Improvements

* Real clinical dataset integration
* Advanced LLM (GPT-based) decision support
* Multi-agent triage system
* Real-time hospital simulation

---

## ⭐ Conclusion

This project demonstrates how **LLM-guided safe reinforcement learning** can significantly improve decision-making in **critical healthcare systems**, ensuring both **performance and safety**.

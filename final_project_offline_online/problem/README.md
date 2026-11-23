# Offline-to-Online Reinforcement Learning: CS 185/285 Final Project

Final project for **UC Berkeley CS 185/285: Deep Reinforcement Learning, Decision Making, and Control (Spring 2026)**, Option II: *Offline-to-online RL*.

In offline-to-online RL, a policy is first trained on a fixed dataset and then fine-tuned with a limited budget of online environment interaction. This report covers:

- naive offline-to-online baselines
- techniques that make the transition more stable
- two more advanced methods, **QC-FQL** and **QAM**
- additional methods I explored

All experiments run on OGBench.

> 🚧 = section still in progress (missing runs or text).

---

## 1. Overview

### Motivation

| | Offline RL | Online RL |
|---|---|---|
| **Pros** | No environment interaction; reuses existing data | Can explore beyond the data; training data matches the policy |
| **Cons** | Limited by dataset coverage; distribution shift at test time | Needs a lot of (possibly unsafe) interaction |

Offline-to-online RL tries to get the best of both. The main difficulty is the switch itself: the training distribution moves from the offline dataset to the policy's own experience, which can make performance collapse.

### Setup

| | |
|---|---|
| Tasks | `cube-single-play-singletask-task1-v0` (sanity check), `cube-double-play-singletask-task1-v0`, `antsoccer-arena-navigate-singletask-task1-v0` (state-based OGBench) |
| Budget | 500k offline steps, 100k online steps, 2 seeds |
| Training-time evaluation | 25 episodes every 50k steps (the curves below; shaded area = min/max over seeds) |
| Final evaluation | 100 episodes on the final checkpoint (marked † in the tables) |

### Implemented methods

| Method | Idea | Code |
|---|---|---|
| SAC+BC | Gaussian SAC actor with a behavior-cloning penalty α | [`sacbc_agent.py`](src/agents/sacbc_agent.py) |
| FQL | Flow-matching BC policy distilled into a one-step Q-maximizing policy | [`fql_agent.py`](src/agents/fql_agent.py) |
| Offline-data retention | Pre-fill the online replay buffer with dataset transitions | [`train_offline_online.py`](src/scripts/train_offline_online.py) |
| WSRL | Collect warm-up rollouts with the frozen offline policy before updating | [`train_offline_online.py`](src/scripts/train_offline_online.py) |
| IFQL | Expectile value learning, flow policy, best-of-N sampling at test time | [`ifql_agent.py`](src/agents/ifql_agent.py) |
| DSRL | RL in the latent-noise space of a flow BC policy | [`dsrl_agent.py`](src/agents/dsrl_agent.py) |
| QSM | Diffusion policy whose score is aligned with ∇ₐQ | [`qsm_agent.py`](src/agents/qsm_agent.py) |
| QC-FQL | FQL over action chunks (Q-chunking) | [`qcfql_agent.py`](src/agents/qcfql_agent.py) |
| QAM | Q-learning with adjoint matching on a flow policy | [`qam_agent.py`](src/agents/qam_agent.py) |
| ACIFQL | IFQL over action chunks | [`acifql_agent.py`](src/agents/acifql_agent.py) |

Every network uses LayerNorm between the linear layer and the activation ([`pytorch_util.py`](src/infrastructure/pytorch_util.py)).

---

## 2. Naive offline-to-online RL

The simplest approach: train offline, then keep training the same agent on freshly collected online data.

![Naive offline-to-online on cube-single](results/s3_naive_cube-single.png)

**What happens at the offline → online switch (naive FQL):**

![Naive FQL transition metrics](results/s3_fql_naive_transition.png)

- On cube-single, FQL saturates offline (100%), dips to 88% after the switch, and recovers to 96%.
- On cube-double, naive fine-tuning **collapses** from 48% to 8%. At the switch, the **Q-loss drops sharply** and the **mean Q-value jumps** from about −140 to −75. The distillation loss barely changes.

**SAC+BC (offline phase only):**

![SAC+BC offline partial](results/s3_sacbc_offline_partial.png)

> 🚧 **Pending:** SAC+BC run through the full 500k offline + 100k online steps, 2 seeds. My current runs stop at 350k offline steps.

**Questions**

1. *What happens to the loss (and its components) and the Q-values at the transition, and why?*
   > 🚧 *Answer.*
2. *Did SAC+BC and FQL behave differently in the offline vs. online phases? Why?*
   > 🚧 *Answer (needs the complete SAC+BC run).*

---

## 3. Improving offline-to-online RL

### 3.1 Retaining offline data

![Offline data in online buffer, cube-double](results/s4_fql_cube-double_offline_data.png)

- Keeping **150k–300k** offline transitions prevents the collapse and lifts cube-double to about 90%. Keeping all 500k adapts more slowly (76%), because the online data is diluted.

> 🚧 **Pending:** antsoccer with 3 amounts of offline data (2 seeds).

### 3.2 Warm-start RL (WSRL)

![WSRL warm-up, cube-double](results/s4_fql_cube-double_wsrl.png)

- WSRL alone also prevents the collapse, and longer warm-ups help (20k steps → 78%). It is weaker than retaining offline data here.

> 🚧 **Pending:** antsoccer with 3 WSRL warm-up lengths (2 seeds).

### 3.3 Combining both

![Combined, cube-double](results/s4_fql_cube-double_combined.png)

<details>
<summary>cube-single sanity check: every variant stays at ~100%</summary>

![cube-single combined](results/s4_fql_cube-single_combined.png)
![cube-single WSRL](results/s4_fql_cube-single_wsrl.png)
![cube-single offline data](results/s4_fql_cube-single_offline_data.png)

</details>

### 3.4 FQL on antsoccer

| BC coefficient sweep (offline) | Best α = 10, 150k offline data + WSRL 5k |
|:-:|:-:|
| ![alpha sweep](results/s4_fql_antsoccer_alpha.png) | ![antsoccer online](results/s4_fql_antsoccer_online.png) |

- antsoccer is very sensitive to α: only α = 10 learns (65% / 54% offline over 100 episodes). Fine-tuning raises the final success to **63%**.

### 3.5 IFQL + rejection sampling

![IFQL](results/s4_ifql.png)

- IFQL saturates cube-single and reaches about 74% offline on antsoccer. On antsoccer it drops after the switch (42–54%), with either 150k or 500k retained transitions.

<p align="center"><img src="assets/videos/ifql_antsoccer_600k_sd0.gif" width="300"><br><sub>IFQL on antsoccer after fine-tuning (600k steps)</sub></p>

> 🚧 **Pending:** full cube-double run (2 seeds). I only have the online-phase evaluations of one seed (92% at 600k).

### 3.6 DSRL

| cube-single / cube-double | antsoccer |
|:-:|:-:|
| ![DSRL sigma](results/s4_dsrl_sigma.png) | ![DSRL sigma antsoccer](results/s4_dsrl_sigma_antsoccer.png) |

- σ<sub>z</sub> = 0.8 works best on cube-double and σ<sub>z</sub> = 1.0 on antsoccer. Fine-tuning brings cube-double to 98% (2 seeds) but antsoccer only to 22%.

| DSRL: cube-double (600k) | DSRL: antsoccer (600k) |
|:-:|:-:|
| <img src="assets/videos/dsrl_cube-double_600k.gif" width="300"> | <img src="assets/videos/dsrl_antsoccer_600k.gif" width="300"> |

> 🚧 **Pending:** complete σ<sub>z</sub> sweeps (several runs stop at 250k–350k steps).

### 3.7 QSM

![QSM partial](results/s4_qsm_cube-single.png)

> 🚧 **Pending:** tuned QSM (η, α ∈ {10, 30, 100}) on cube-single, cube-double and antsoccer. The current run is unstable (loss spike at ~120k steps) and stops at 150k.

**Questions**

1. *Offline data: what changes in the data distribution we sample from when we retain offline data?*
   > 🚧 *Answer.*
2. *WSRL: how does the data distribution differ from retaining offline data, and why would that be better?*
   > 🚧 *Answer.*
3. *Rejection sampling: what is the intuition, what does it force the action distribution to look like, and what policy do we approximate as N → ∞?*
   > 🚧 *Answer.*
4. *DSRL: what are the benefits of a fixed BC policy plus a latent-noise policy for efficiency, exploration and access to the model?*
   > 🚧 *Answer.*
5. *QSM: with π(a|s) ∝ exp(η Q(s,a)), why does the objective L(π) make sense?*
   > 🚧 *Answer.*

---

## 4. QC-FQL: Flow Q-Learning with action chunking

**Method.** QC-FQL applies action chunking (Q-chunking; Li, Zhou & Levine, 2025) to FQL.

- The policy outputs a chunk of **H = 5** consecutive actions, executed open-loop.
- The critic scores the whole chunk, Q(s, a<sub>t:t+H</sub>), and is trained with an **H-step backup**. This spreads the sparse task reward faster than a 1-step backup.
- As in FQL, a flow-matching BC policy models the chunk distribution in the dataset. A one-step policy is distilled from it while maximizing Q (α = 300).
- Online fine-tuning uses 150k retained offline transitions and 5k WSRL steps.

> 🚧 *To expand: losses and design decisions (H, critic ensemble size, Q aggregation, α).*

**Results on cube-double**

![QC-FQL vs FQL with Q-values](results/s5_qcfql_h5_vs_fql_double.png)

- QC-FQL reaches **97% †** (94% / 100% over two seeds), above the 95% project target.
- It is *weaker* offline (about 20% vs 48% for FQL) but jumps to about 90% within the first 50k online steps.
- During fine-tuning, the Q-values rise smoothly, with no overestimation blow-up.

| 100k steps (offline) | 500k steps (end of offline) | 600k steps (end of online) |
|:-:|:-:|:-:|
| <img src="assets/videos/qcfql_cube-double_100k_offline.gif" width="260"> | <img src="assets/videos/qcfql_cube-double_500k_end-offline.gif" width="260"> | <img src="assets/videos/qcfql_cube-double_600k_end-online.gif" width="260"> |
| fumbles the cubes; episodes time out | partial progress, still slow | all 4 episodes solved in under 100 steps |

<sub>From the QC-FQL ablation run with α = 222 (seed 8).</sub>

**Results on antsoccer**

![QC-FQL H1 vs FQL antsoccer](results/s5_qcfql_h1_vs_fql_antsoccer.png)

- Chunking hurts here: H = 5 gets only 5% †. H = 1 matches FQL offline but drops during fine-tuning (48% † vs 63% †). Dribbling a ball seems to need reactive closed-loop control, and open-loop chunks remove it.

> 🚧 **Pending:** second seed for QC-FQL on antsoccer.

<details>
<summary>Ablations: chunk length, critic ensemble size and α on cube-double</summary>

![QC-FQL variants](results/double_qcfql.png)
![QC-FQL H1 vs FQL cube-double](results/s5_qcfql_h1_vs_fql_double.png)

</details>

---

## 5. QAM: Q-learning with Adjoint Matching

**Method.** QAM (Li & Levine, 2026) trains a flow policy toward the Boltzmann policy π(a|s) ∝ π<sub>BC</sub>(a|s)·exp(η Q(s,a)), without backpropagating through the flow.

- The critic's action gradient ∇ₐQ, scaled by the inverse temperature η, is the terminal condition of an **adjoint ODE**.
- The adjoint is integrated backward along the flow trajectory, which gives a regression target for the velocity field at every flow step.
- My implementation uses action chunks of H = 5, η = 1, 2 critics with mean aggregation, and 10 flow steps.

> 🚧 *To expand: the adjoint-matching loss and design decisions.*

**Results on cube-double**

![QAM vs FQL on cube-double](results/s5_qam_cube-double.png)

- QAM learns faster than FQL early in offline training (44% at 300k), but ends the offline phase at 24%.
- Online it jumps to 88% within 50k steps and ends at **84%** (2 seeds). Q-values rise steadily, a little more slowly than FQL's.

<p align="center"><img src="assets/videos/qam_cube-double_600k.gif" width="300"><br><sub>QAM on cube-double after fine-tuning (600k steps)</sub></p>

> 🚧 **Pending:** QAM on antsoccer (early runs were still at 0% after 50k steps), plus an inverse-temperature sweep.

---

## 6. Additional methods

### 6.1 ACIFQL: action-chunked IFQL

IFQL with 5-step action chunks: expectile value learning over chunks, and best-of-N sampling of whole chunks at test time.

![ACIFQL on cube-double](results/s5_acifql_cube-double.png)

- Offline success stays between 20% and 48%. The Q- and V-estimates keep decreasing during offline training and recover only once online data arrives.
- It ends at 60% after fine-tuning (seed 10). A second seed reached 84%, but only its online-phase evaluations were logged.

> 🚧 **Pending:** 2 complete seeds; antsoccer.

### 6.2 Other variants

> 🚧 *Space for further methods or ablations.*

<details>
<summary>Action-chunked methods side by side on cube-double</summary>

![chunked methods](results/double_methods.png)

</details>

---

## 7. Comparison of all methods

Success rate at step 600k (500k offline + 100k online). † = 100-episode final evaluation. Other values are the last 25-episode training-time evaluation.

| Method | cube-double | antsoccer |
|---|:-:|:-:|
| FQL, naive fine-tuning | 8% | 🚧 |
| FQL + 150k offline data | 88% † | 🚧 |
| FQL + 150k offline data + WSRL 5k | 86% † | **63% †** |
| SAC+BC | 🚧 | 🚧 |
| IFQL (τ = 0.9) | 92% (1 seed) | 42% |
| DSRL (σ<sub>z</sub> = 0.8 / 1.0) | 98% | 22% |
| QSM | 🚧 | 🚧 |
| QC-FQL (H = 5) | **97% †** | 5% † (1 seed) |
| QC-FQL (H = 1) | 🚧 | 48% † (1 seed) |
| QAM (H = 5) | 84% | 🚧 |
| ACIFQL (H = 5) | 60% (1 seed) | 🚧 |
| *Project target* | *> 95%* | *> 45%* |

### cube-double

![cube-double all methods](results/final_cube-double_all_methods.png)

### antsoccer

![antsoccer all methods](results/final_antsoccer_all_methods.png)

> 🚧 **Pending for the final plots:** SAC+BC and QSM on both tasks, a second IFQL seed on cube-double, QAM and ACIFQL on antsoccer.

---

## 8. Discussion

> 🚧 *To write.*

- **Stabilizing the transition:** retaining part of the offline data was the most effective single fix for the collapse at the offline → online switch. WSRL helps as well, but less.
- **Task dependence:** action-chunked methods (QC-FQL, QAM) adapt very quickly on cube-double, a long-horizon manipulation task. On antsoccer they underperform plain FQL, where reactive control matters more.
- **Limitations:** some baselines have a single seed, and the 25-episode training-time evaluations are noisy.
- **Future work:** 🚧

## References

- Ball et al., *Efficient Online Reinforcement Learning with Offline Data* (RLPD), 2023
- Hansen-Estruch et al., *IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies*, 2023
- Li & Levine, *Q-learning with Adjoint Matching* (QAM), 2026
- Li, Zhou & Levine, *Reinforcement Learning with Action Chunking*, 2025
- Park et al., *OGBench: Benchmarking Offline Goal-Conditioned RL*, ICLR 2025
- Park, Li & Levine, *Flow Q-Learning*, 2025
- Psenka et al., *Learning a Diffusion Model Policy from Rewards via Q-Score Matching*, 2025
- Wagenmaker et al., *Steering Your Diffusion Policy with Latent Space Reinforcement Learning* (DSRL), 2025
- Zhou et al., *Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data* (WSRL), 2025

Starter code and project description: [berkeleydeeprlcourse/homework_spring2026](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/final_project_offline_online).

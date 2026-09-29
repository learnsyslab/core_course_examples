# Optimal Control and Decision-Making 
This repository contains the examples shown in the course 'Optimal Control and Decision-Making' by Prof. Angela P. Schoellig at TUM. You can try them yourself following the steps below. The examples are implemented in Python and use Jupyter notebooks for in-depth exploration of the concepts covered in the lectures. 

## Course Overview
These examples complement the theoretical concepts covered in the lectures by providing interactive implementations. You'll find implementations of dynamic programming, model predictive control, reinforcement learning, and more - applied to the [mountain car problem](https://en.wikipedia.org/wiki/Mountain_car_problem). 

![Figure 1: The mountain car reaches the top of the hill using nonlinear model predictive control (MPC)](figure/MountainCarAnimation.gif)

## Prerequisites
- Basic knowledge of Python programming
- Familiarity with numerical methods and linear algebra
- Understanding of control theory fundamentals
- Previous experience with Jupyter notebooks is helpful but not required

## Repository Structure
```
├── Dockerfile
├── ex1_DP/                   # DP examples
│   ├── dynamic_programming.ipynb
│   └── ...
├── ex2_LQR/                  # LQR examples
│   ├── linear_quadratic_regulator.ipynb
│   └── ...
└── ... # More examples
```

## Course Content
Each chapter folder contains one or more notebooks. The table lists the main methods covered in each of them.

| Chapter | Notebook | Main methods |
|:--|:--|:--|
| **1. Dynamic Programming**<br>`ex1_DP/` | [1.1 Dynamic Programming](ex1_DP/1.1_dynamic_programming.ipynb) | mountain car environment and system dynamics, dynamic programming |
| **2. Linear Quadratic Regulator**<br>`ex2_LQR/` | [2.1 Linear Quadratic Regulator](ex2_LQR/2.1_linear_quadratic_regulator.ipynb) | finite-horizon LQR, infinite-horizon LQR |
| **3. Optimization Fundamentals**<br>`ex3_OPT/` | [3.1 Optimization Fundamentals](ex3_OPT/3.1_optimization_fundamentals.ipynb) | *unconstrained:* gradient descent, Newton's method, line search<br>*constrained:* KKT conditions, quadratic programming (QP), sequential quadratic programming (SQP) |
| **4. Iterative LQR**<br>`ex4_iLQR/` | [4.1 Iterative LQR](ex4_iLQR/4.1_iterative_linear_quadratic_regulator.ipynb) | iterative LQR (iLQR) |
| **5. Model Predictive Control**<br>`ex5_MPC/` | [5.1 Model Predictive Control](ex5_MPC/5.1_model_predictive_control_part1.ipynb) | open-loop optimal control, linear MPC, nonlinear MPC |
| | [5.2 Tracking and Robust MPC](ex5_MPC/5.2_model_predictive_control_part2.ipynb) | tracking MPC, robust MPC |
| **6. Model Learning and Learning-based Control**<br>`ex6_SysID/` | [6.1 Model Learning](ex6_SysID/6.1_SysID.ipynb) | linear regression (LR), Bayesian linear regression (BLR) |
| | [6.2 Learning-based MPC](ex6_SysID/6.2_Learning_Based_MPC.ipynb) | MPC with an LR model, robust MPC with a BLR model |
| | [6.3 GP Learning and GP-MPC](ex6_SysID/6.3_GP_Learning_and_GP_MPC.ipynb) | Gaussian process (GP) regression, BLR vs. GP, GP-MPC |
| | [6.4 Data-Enabled Predictive Control](ex6_SysID/6.4_Data_Enabled_Predictive_Control.ipynb) | Willems' fundamental lemma, DeePC, regularized DeePC for nonlinear systems |
| **7. Reinforcement Learning**<br>`ex7_RL/` | [7.0 Stochastic Shortest Path](ex7_RL/7.0_ssp.ipynb) | stochastic shortest path (SSP) problem |
| | [7.1 Model-based RL](ex7_RL/7.1_mbrl.ipynb) | value iteration, policy iteration |
| | [7.2 Model-free RL](ex7_RL/7.2_mfrl.ipynb) | Monte Carlo method, Q-learning |
| **8. Deep Reinforcement Learning**<br>`ex8_DRL/` | [8.1 Deep Reinforcement Learning](ex8_DRL/8.1_drl.ipynb) | proximal policy optimization (PPO) |

## Setup

### Docker Desktop
These instructions use Docker Desktop to create a containerized environment using Docker that works on any common operating systems (OS), e.g., Linux, Windows, and Mac OS. If you are on Linux you may instead directly install the Docker Engine, however, we recommend using Docker Desktop. 

Install Docker Desktop for your OS: 
https://docs.docker.com/desktop/ 

### Visual Studio Code
These instructions are for Visual Studio (VS) Code and have only been tested with VS Code. 

Download and install VS Code for your OS:
https://code.visualstudio.com/Download

VS Code lets us conveniently open a directory inside a Docker container. For this, install the Dev Containers extension in VS Code. See the instructions here: https://code.visualstudio.com/docs/devcontainers/tutorial 

### Git & GitHub
For version control we are using Git. Install Git on your system (if you don't have it already): https://git-scm.com/downloads  

We host this repository on GitHub. Make sure that you have an account to pull it (and file issues, create pull requests, etc.): https://docs.github.com/en/get-started  

You will have to set up your account to connect to GitHub via SSH: https://docs.github.com/en/authentication/connecting-to-github-with-ssh. In particular, you will have to add a SSH key to your GitHub account (if you haven't done so already). 

If you are unfamiliar with Git, check out this brief overview: https://education.github.com/git-cheat-sheet-education.pdf 

## Installation 
First, pull the repository
```
cd <YOUR DESIRED DIRECTORY>
git clone git@github.com:utiasDSL/core_course_examples.git
```
Open VS Code. Press the `F1` key, select `Dev Containers: Open Folder in Container` in the search bar, and select the cloned repository as the folder. The repository contains a `Dockerfile`, which is like a recipe how to set up all the required dependencies in a container. The creation of the container may take a couple of minutes.  

## Usage
In VS Code's Explorer, expand the folder `ex1_DP` and double-click on the Jupyter notebook `1.1_dynamic_programming.ipynb`.

This will open up the first example on dynamic programming and will prompt you to install additional extensions, e.g., the Jupyter and Python extensions (if you haven't installed them already). To run the cells in the notebook, you will have to select a Kernel. As the kernel, select the Python environment provided by the Docker container. Detailed instructions on how to use Jupyter notebooks in VS Code can be found here: https://code.visualstudio.com/docs/datascience/jupyter-notebooks 

Over time we will push updates to this repository, e.g., fixes and additional examples. You can receive them by using
``` 
git pull 
```

If you run into any issues, feel free to create an issue on GitHub. We recommend creating your own fork of the repository if you are planning to do development on your own, see: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks 

## Authors
Haocheng Zhao, Lukas Brunke, SiQi Zhou, and Angela P. Schoellig from the Learning Systems and Robotics Lab @ TUM
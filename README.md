# Visual CartPole RL

This project showcases a Deep Reinforcement Learning agent trained to solve the `CartPole-v1` environment from Gymnasium using only visual input (pixels from the screen). The goal is to achieve a complete workflow from environment preprocessing and custom model building to training and evaluation.

### Core Technologies
* **Gymnasium:** For the simulation environment.
* **PyTorch:** For building the custom CNN policy.
* **OpenCV:** For image preprocessing.
* **Stable Baselines3:** For implementing the RL algorithm.
* **Google Colab & GitHub:** For development and version control.

---
## Project Status & Progress

### Phase 1: Environment and Preprocessing (✅ Complete)
* **Objective:** Transform the standard `CartPole-v1` environment into one that uses screen pixels as its state.
* **Key Activities:**
    * Set up the project structure with directories for notebooks, models, and media.
    * Developed an image preprocessing pipeline using OpenCV to crop, grayscale, and resize the rendered frames.
    * Created a custom `gym.ObservationWrapper` to seamlessly integrate the preprocessing pipeline with the environment.
* **Outcome:** A tested wrapped Gymnasium environment ready for visual-based training. The completed work is available in `notebooks/1_Environment_and_Preprocessing.ipynb`.

### Phase 2: Model Building and Training (✅ Complete)
* **Objective:** Define a custom CNN and train a PPO agent to solve the visual CartPole task.
* **Key Activities:**
    * Defined a custom CNN feature extractor using PyTorch to interpret env images.
    * Configured a PPO agent from Stable Baselines3 to use the custom network.
    * Ran a 250,000 timestep training session, which demonstrated initial learning followed by a performance plateau.
* **Outcome:** A trained model (`ppo_visual_cartpole_250k.zip`) saved and ready for evaluation. The completed work is in `notebooks/2_Training_the_Agent.ipynb`.

---
## Repository Structure

-   **/notebooks**: Contains the Jupyter notebooks for each phase of the project.
-   **/models**: Will store the final trained agent models.
-   **/media**: Will store output GIFs and videos of the trained agent.
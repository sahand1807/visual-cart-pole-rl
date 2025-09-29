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

---
## Repository Structure

-   **/notebooks**: Contains the Jupyter notebooks for each phase of the project.
-   **/models**: Will store the final trained agent models.
-   **/media**: Will store output GIFs and videos of the trained agent.
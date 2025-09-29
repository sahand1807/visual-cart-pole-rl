import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import cv2

def preprocess_frame(frame):
    """
    Preprocesses a single frame from the CartPole environment.

    Args:
        frame (np.array): The raw RGB frame from env.render().

    Returns:
        np.array: The processed grayscale frame.
    """
    # 1. Crop the screen to focus on the cart and pole.
    # The exact values [150:350, :] might need tweaking based on the render,
    # but these are generally good for CartPole-v1.
    processed_frame = frame[150:350, :]

    # 2. Convert the image to grayscale.
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2GRAY)

    # 3. Resize the image to 84x84 pixels.
    processed_frame = cv2.resize(
        processed_frame, (84, 84), interpolation=cv2.INTER_AREA
    )

    # 4. Add a channel dimension for the CNN.
    # The CNN expects an input shape of (Height, Width, Channels).
    # Since we have a grayscale image, we add a 1-channel dimension.
    return processed_frame[:, :, None]


# ==============================================================================
# GYMNASIUM WRAPPER CLASS
# ==============================================================================
class ImageWrapper(gym.ObservationWrapper):
    """
    A custom Gymnasium wrapper to preprocess image observations and modify the
    observation space.
    """
    def __init__(self, env):
        super().__init__(env)
        # Define the new observation space.
        # It's an 84x84 image with 1 channel (grayscale).
        # The values range from 0 to 255 (uint8).
        self.observation_space = Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        """
        Processes the observation before it is returned by the environment.
        """
        # The original `obs` is the state vector, which will be ignored.
        # We render the environment to get the pixels.
        frame = self.env.render()
        # Apply the cv2 preprocessing function.
        return preprocess_frame(frame)

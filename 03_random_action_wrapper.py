import gym
import random
from typing import TypeVar

Action = TypeVar("Action")

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.3):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random")
            return self.env.action_space.sample()

        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))
    env = gym.wrappers.Monitor(env, "recording", force=True)

    obs = env.reset()
    total_reward = 0.0

    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break

    print("Reward got: %.2f" %total_reward)
import cv2
import gym
import gym.spaces
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    '''
    The preceding wrapper presses the FIRE button in environments that require that
    for the game to start. In addition to pressing FIRE, this wrapper checks for several
    corner cases that are present in some games.
    '''

    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)

        if done:
            self.env.reset()

        return obs


class MaxAndSkipEnv(gym.Wrapper):
    '''
    This wrapper combines the repetition of actions during K frames and pixels from
    two consecutive frames.
    '''

    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None

        for _ in range(self._skip):

            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self._obs_buffer.append(obs)
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    '''
    The goal of this wrapper is to convert input observations from the emulator, which
    normally has a resolution of 210×160 pixels with RGB color channels, to a grayscale
    84×84 image. It does this using a colorimetric grayscale conversion (which is closer
    to human color perception than a simple averaging of color channels), resizing the
    image, and cropping the top and bottom parts of the result.
    '''

    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, 'Unknown resolution.'

        img = img[:, :, 0] * 0.299 + \
              img[:, :, 1] * 0.587 + \
              img[:, :, 2] * 0.114

        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])

        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    '''
    This class creates a stack of subsequent frames along the first dimension and returns
    them as an observation. The purpose is to give the network an idea about the
    dynamics of the objects, such as the speed and direction of the ball in Pong or how
    enemies are moving. This is very important information, which it is not possible to
    obtain from a single image.
    '''

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space

        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=self.dtype
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    '''
    This simple wrapper changes the shape of the observation from HWC (height, width,
    channel) to the CHW (channel, height, width) format required by PyTorch. The
    input shape of the tensor has a color channel as the last dimension, but PyTorch's
    convolution layers assume the color channel to be the first dimension.
    '''

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    '''
    converts observation data from bytes to
    floats, and scales every pixel's value to the range [0.0...1.0]
    '''

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)

    return ScaledFloatFrame(env)

import sys
from tabnanny import verbose
from unicodedata import name
import gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from CustomEnv import CustomEnv, deck
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

import time
vecEnv = CustomEnv()
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

custom_objects = {}
if newer_python_version:
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
if __name__ == '__main__':
    # vecEnv = gym.make("CartPole-v1")
    # vecEnv = SubprocVecEnv([gym.make("CartPole-v1") for _ in range(2)])
    vecEnv = make_vec_env(CustomEnv, n_envs=8, vec_env_cls=SubprocVecEnv)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./logs/',
                                             name_prefix='rl_model')

    model = PPO('MlpPolicy', vecEnv, verbose=0)
    model = PPO.load("./model (4)", env=CustomEnv(),
                     custom_objects=custom_objects, verbose=1)
    start_time = time.time()

    # model.learn(total_timesteps=13375000)
    model.learn(total_timesteps=13375000,
                callback=checkpoint_callback)
    print(model.learning_rate)

    print("training end", time.time()-start_time)
    # model = DQN.load("model (3).zip")
    # model.save("model")

    vecEnv = CustomEnv()

    obs = vecEnv.reset()

    print(len(obs))
    for i in range(130):
        action, _state = model.predict(obs, deterministic=True)
        # start_time = time.time()
        obs, reward, done, info = vecEnv.step(action)
        # print(info[0])
        # print(reward[0])
        vecEnv.render()

        if done:
            obs = vecEnv.reset()

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
if __name__ == '__main__':
    # vecEnv = gym.make("CartPole-v1")
    # vecEnv = SubprocVecEnv([gym.make("CartPole-v1") for _ in range(2)])
    vecEnv = make_vec_env(CustomEnv, n_envs=8, vec_env_cls=SubprocVecEnv)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./logs/',
                                             name_prefix='rl_model')

    model = PPO('MlpPolicy', vecEnv, verbose=0)
    # model = DQN.load("./model (2)")
    start_time = time.time()
    # model.learn(total_timesteps=13375000)
    model.learn(total_timesteps=13375000, callback=checkpoint_callback)
    print("training end", time.time()-start_time)
    # model = DQN.load("model (3).zip")
    model.save("model")

    obs = vecEnv.reset()
    for i in range(130):
        action, _state = model.predict(obs, deterministic=True)
        # start_time = time.time()
        obs, reward, done, info = vecEnv.step(action)
        print(info[0])
        print(reward[0])
        if done[0]:
            obs = vecEnv.reset()

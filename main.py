from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from CustomEnv import CustomEnv, deck
from stable_baselines3.common.env_util import make_vec_env
vecEnv = make_vec_env(CustomEnv, 11)
env = CustomEnv()
# It will check your custom environment and output additional warnings if needed
# check_env(env)
checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./logs/',
                                         name_prefix='rl_model')

model = DQN('MlpPolicy', vecEnv, verbose=0)
model.learn(total_timesteps=13375000, callback=checkpoint_callback)
# model.learn(total_timesteps=10000)
# model = DQN.load("model (1).zip")
model.save("model")
obs = env.reset()
for i in range(5):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    print(reward)
    print("")
    print("")
    print("")
    print("")
    print("")
    env.render()
    if done:
        obs = env.reset()

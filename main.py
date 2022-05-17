from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from CustomEnv import CustomEnv, deck

env = CustomEnv()
# It will check your custom environment and output additional warnings if needed
# check_env(env)
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./logs/',
                                         name_prefix='rl_model')
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1070000, callback=checkpoint_callback)
model.save("model")
obs = env.reset()
for i in range(5):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        obs = env.reset()

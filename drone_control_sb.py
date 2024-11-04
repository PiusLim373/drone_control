#!/usr/bin python3
from new_gym import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import datetime

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq  # Save frequency in timesteps
        self.save_path = save_path    # Path to save the model

    def _on_step(self) -> bool:
        # Check if the current step is a multiple of the save frequency
        if self.n_calls % self.save_freq == 0:
            # Save the model
            self.model.save(f"{self.save_path}/ppo_model_{self.num_timesteps}")
            if self.verbose > 0:
                print(f"Model saved at timestep {self.num_timesteps}")
        return True

if __name__ == "__main__":
    def make_env():
        def _init():
            env = DroneControlGym(render=False)
            env = Monitor(env)
            return env
        return _init 
    num_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # temp = DroneControlGym(render=True)
    # temp = Monitor(temp)
    # env = DummyVecEnv([lambda: temp])

    policy_kwargs = dict(net_arch=[128, 128, 128, 128])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=1024, batch_size=64, n_epochs=20, verbose=1, tensorboard_log="./tensorboard_logs/")

    # Set the save frequency and path
    save_freq = 100000  # Save every 5000 timesteps
    save_path = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_drone_ppo"
    callback = SaveModelCallback(save_freq=save_freq, save_path=save_path, verbose=1)

    model.learn(total_timesteps=16_000_000, callback=callback, tb_log_name="PPO_Quadcopter")
    print("done")
    model.save("ppo_drone_control")


''' 
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 159          |
|    ep_rew_mean          | -323         |
| time/                   |              |
|    fps                  | 48           |
|    iterations           | 13           |
|    time_elapsed         | 545          |
|    total_timesteps      | 26624        |
| train/                  |              |
|    approx_kl            | 0.0054851263 |
|    clip_fraction        | 0.0119       |
|    clip_range           | 0.2          |
|    entropy_loss         | -2.67        |
|    explained_variance   | 0.1          |
|    learning_rate        | 0.0003       |
|    loss                 | 3.65e+03     |
|    n_updates            | 120          |
|    policy_gradient_loss | -0.01        |
|    value_loss           | 9.87e+03     |
------------------------------------------


-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 139         |
|    ep_rew_mean          | -328        |
| time/                   |             |
|    fps                  | 44          |
|    iterations           | 25          |
|    time_elapsed         | 570         |
|    total_timesteps      | 25600       |
| train/                  |             |
|    approx_kl            | 0.009221831 |
|    clip_fraction        | 0.0734      |
|    clip_range           | 0.2         |
|    entropy_loss         | -2.55       |
|    explained_variance   | 0.371       |
|    learning_rate        | 0.0003      |
|    loss                 | 2.38e+03    |
|    n_updates            | 360         |
|    policy_gradient_loss | -0.0241     |
|    value_loss           | 5.9e+03     |
-----------------------------------------

250k step
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 275          |
|    ep_rew_mean          | -41.2        |
| time/                   |              |
|    fps                  | 294          |
|    iterations           | 245          |
|    time_elapsed         | 852          |
|    total_timesteps      | 250880       |
| train/                  |              |
|    approx_kl            | 0.0049841637 |
|    clip_fraction        | 0.0102       |
|    clip_range           | 0.2          |
|    entropy_loss         | -1.67        |
|    explained_variance   | -0.285       |
|    learning_rate        | 0.0003       |
|    loss                 | 23.4         |
|    n_updates            | 3660         |
|    policy_gradient_loss | -0.00557     |
|    value_loss           | 228          |
------------------------------------------

10m
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 286        |
|    ep_rew_mean          | 26.9       |
| time/                   |            |
|    fps                  | 225        |
|    iterations           | 9766       |
|    time_elapsed         | 44271      |
|    total_timesteps      | 10000384   |
| train/                  |            |
|    approx_kl            | 0.03237743 |
|    clip_fraction        | 0.102      |
|    clip_range           | 0.2        |
|    entropy_loss         | -0.762     |
|    explained_variance   | 0.725      |
|    learning_rate        | 0.0003     |
|    loss                 | 62.6       |
|    n_updates            | 195300     |
|    policy_gradient_loss | -0.0306    |
|    value_loss           | 411        |
----------------------------------------

'''
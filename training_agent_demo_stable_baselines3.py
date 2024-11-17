#!/usr/bin python3
from drone_control_gym import *
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
            self.model.save(f"{self.save_path}/autosave_step_{self.num_timesteps}")
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
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=1024, batch_size=64, n_epochs=20, verbose=1, tensorboard_log="./saves/")

    # Set the save frequency and path
    save_freq = 100000  # Save every 100000 timesteps
    save_path = os.path.join("saves", "autosave_sb_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    callback = SaveModelCallback(save_freq=save_freq, save_path=save_path, verbose=1)

    model.learn(total_timesteps=16_000_000, callback=callback, tb_log_name="sb_tensorboard_log")
    print("done")
    model.save("ppo_drone_control")

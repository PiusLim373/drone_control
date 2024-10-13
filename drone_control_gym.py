#!/usr/bin/python3
import gym
import numpy as np
import mujoco
import mujoco_viewer
import os
import random
import copy
import logging
from custom_utils import *

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)
# Global variables
RANGE_LIMIT = 10
RESOLUTION = 10
GOAL_TOLERANCE = 0.1  # if the drone is within 0.1m of the goal, it is considered to have reached the goal
FULL_THROTTLE = 7
DRONE_MODEL_PATH = os.path.join(os.getcwd(), "asset/skydio_x2/scene.xml")
ROLL_TARGET = 5  # degrees
PITCH_TARGET = 5  # degrees
ROLL_THRESHOLD = 90  # degrees
PITCH_THRESHOLD = 90  # degrees
HEIGHT_LOWER_LIMIT = 0.05  # m
IDLE_POSITION_THRESHOLD = 0.05  # m
GYRO_SMOOTH_THRESHOLD = 0.05  # Threshold for angular velocity (rad/s)
ACC_SMOOTH_THRESHOLD = 0.1  # Threshold for linear acceleration (m/s^2)

(
    ACTION_COST,
    IDLE_COST,
    GOAL_REWARD,
    OUT_OF_BOUND_REWARD,
    FLIPPED_REWARD,
    SMOOTH_MOTION_REWARD,
) = (-0.1, -0.2, 1.0, -1.0, -2.0, 0.2)
ACTIONS = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]


class DroneControlGym(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(DRONE_MODEL_PATH)
        self.drone = mujoco.MjData(self.model)
        # Get the index of the IMU sensors (accelerometer and gyro) from the XML definition
        self.gyro_index = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")]
        self.acc_index = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc")]

        # Initialize IMU readings
        self.drone_acc = []
        self.drone_gyro = []
        self.sensor_attributes = []

        mujoco.mj_step(self.model, self.drone)
        self.motor_states = [
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
        ]
        self.goal_pose = [
            random.uniform(0.0, RANGE_LIMIT),
            random.uniform(0.0, RANGE_LIMIT),
            random.uniform(0.1, RANGE_LIMIT),
        ]  # randomly initialize goal pose (x, y, z) within RANGE_LIMIT
        self.drone_pose = [0.0, 0.0, 0.0]
        self.last_drone_pose = [0.0, 0.0, 0.0]
        self.drone_rpy = None
        self.last_drone_rpy = None
        self.drone_motor_thrust = None
        self.distance_to_goal = None
        self.drone_last_ctrl = copy.deepcopy(self.drone.ctrl)
        self.step_count = 1
        self.has_finished = False

    def _update_drone_data_from_sim(self):
        rpy_angles = quaternion_to_rpy(self.drone.xquat[1])
        self.drone_rpy = np.array(rpy_angles)
        self.drone_pose = np.array(self.drone.xpos[1])
        self.drone_acc = self.drone.sensordata[self.acc_index : self.acc_index + 3]  # Accelerometer (x, y, z)
        self.drone_gyro = self.drone.sensordata[self.gyro_index : self.gyro_index + 3]  # Gyroscope (x, y, z)

    def _update_motor_thrust(self):
        duty_cycle = [0] * 4
        thrust = [0] * 4
        for index, individual_motor_state in enumerate(self.motor_states):
            duty_cycle[index] = average(individual_motor_state)
            thrust[index] = duty_cycle[index] * FULL_THROTTLE
        self.drone.ctrl[:] = np.array(thrust)

    def _calculate_goal_attributes(self):
        # Calculate vector difference between goal and drone's current position
        vector_to_goal = np.array(self.goal_pose) - np.array(self.drone_pose)
        self.distance_to_goal = np.linalg.norm(vector_to_goal)  # Calculate distance
        if self.distance_to_goal > 0:
            dx, dy, dz = vector_to_goal / self.distance_to_goal  # Normalize
        else:
            dx, dy, dz = 0.0, 0.0, 0.0  # If already at the goal
        return np.array([dx, dy, dz, self.distance_to_goal])

    def _calculate_reward(self):
        # calculate reward based on the current state of the drone and if the drone has reached the goal
        # Check if the roll and pitch are close to zero (upright)
        if (
            self.distance_to_goal < GOAL_TOLERANCE
            and abs(self.drone_rpy[0]) < ROLL_TARGET
            and abs(self.drone_rpy[1]) < PITCH_TARGET
        ):
            logging.debug("The drone is staying upright at the goal.")
            return True, GOAL_REWARD
        else:
            reward = 0
            finished = False
            # if drone is flipped (or collided?), return FLIPPED_REWARD
            if abs(self.drone_rpy[0]) > ROLL_THRESHOLD or abs(self.drone_rpy[1]) > PITCH_THRESHOLD:
                logging.debug("drone has flipped")
                reward += FLIPPED_REWARD
                finished = True
            # if drone is out of RANGE_LIMIT return OUT_OF_BOUND_REWARD, not sure need this or not
            if (
                self.drone_pose[0] > RANGE_LIMIT
                or self.drone_pose[1] > RANGE_LIMIT
                or self.drone_pose[2] < HEIGHT_LOWER_LIMIT
            ):
                logging.debug("drone is out of bound")
                reward += OUT_OF_BOUND_REWARD
                finished = True
            # check if current duty cycle is equal to previous duty cycle and position is same
            if np.array_equal(self.drone.ctrl, self.drone_last_ctrl) and np.all(
                np.abs(self.drone_pose - self.last_drone_pose) < IDLE_POSITION_THRESHOLD
            ):
                logging.debug("drone is idle in place")
                reward += IDLE_COST
            else:
                logging.debug("drone took valid action")
                reward += ACTION_COST
                # Check for smooth angular motion (low angular velocity)
                if np.all(np.abs(self.drone_gyro) < GYRO_SMOOTH_THRESHOLD):
                    logging.debug("drone is rotating smoothly")
                    reward += SMOOTH_MOTION_REWARD
                # Check for smooth linear motion (low acceleration)
                if np.all(np.abs(self.drone_acc) < ACC_SMOOTH_THRESHOLD):
                    logging.debug("drone is moving smoothly")
                    reward += SMOOTH_MOTION_REWARD
            return finished, reward

    def get_pose(self):
        self._update_drone_data_from_sim()
        return {
            "x": self.drone_pose[0],
            "y": self.drone_pose[1],
            "z": self.drone_pose[2],
            "roll": self.drone_rpy[0],
            "pitch": self.drone_rpy[1],
            "yaw": self.drone_rpy[2],
        }

    def get_goal(self):
        return {"x": self.goal_pose[0], "y": self.goal_pose[1], "z": self.goal_pose[2]}

    def get_imu_reading(self):
        return {
            "accelerometer": {
                "x": self.drone_acc[0],
                "y": self.drone_acc[1],
                "z": self.drone_acc[2],
            },
            "gyro": {
                "x": self.drone_gyro[0],
                "y": self.drone_gyro[1],
                "z": self.drone_gyro[2],
            },
        }

    def reset(self):
        # Reset simulation data
        mujoco.mj_resetData(self.model, self.drone)
        # Randomly initialize a new goal pose
        self.goal_pose = [
            random.uniform(0.0, RANGE_LIMIT),
            random.uniform(0.0, RANGE_LIMIT),
            random.uniform(0.1, RANGE_LIMIT),
        ]
        # Reset motor states
        self.motor_states = [
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
        ]
        # Reset IMU readings
        self.drone_acc = []
        self.drone_gyro = []
        self.sensor_attributes = []
        # Reset the drone pose and orientation to defaults
        self.drone_pose = [0.0, 0.0, 0.0]
        self.last_drone_pose = [0.0, 0.0, 0.0]
        self.drone_rpy = None
        self.drone_motor_thrust = None
        self.distance_to_goal = None
        self.drone_last_ctrl = np.zeros(4)
        self.step_count = 1
        self.has_finished = False
        # Step the simulation once to apply the reset thrust values
        mujoco.mj_step(self.model, self.drone)
        # Return the initial observation and goal
        return self.get_all_state(), ACTIONS

    def step(self, action):
        logging.debug(f"drone stepped with action [{action[0]}] [{action[1]}]")
        logging.debug(f"                             X")
        logging.debug(f"                          [{action[3]}] [{action[2]}]")

        # using action given, pop the first motor state and append the new motor state for each motor
        for index, individual_action in enumerate(action):
            self.motor_states[index].pop(0)
            self.motor_states[index].append(individual_action)
        self._update_motor_thrust()
        mujoco.mj_step(self.model, self.drone)
        self.step_count += 1

        self._update_drone_data_from_sim()
        self.drone_motor_thrust = np.array(self.drone.actuator_force)

        self.goal_attributes = self._calculate_goal_attributes()  # return list of [dx, dy, dz, d]
        self.has_finished, self.reward = self._calculate_reward()  # return bool and float

        logging.debug(f"Step: {self.step_count}")
        logging.debug(f"Reward: {self.reward}")
        logging.debug(f"Finished episode: {self.has_finished}")
        logging.debug(f"Goal attributes: {self.goal_attributes}")
        logging.debug(f"Current drone RPY: {self.drone_rpy}")
        logging.debug(f"Current drone motor thrust: {self.drone_motor_thrust}")
        logging.debug(f"Current drone accelerations:{self.drone_acc}")
        logging.debug(f"Current drone gyro velocities{self.drone_gyro}")

        return self.get_all_state()

    def get_all_state(self):
        return (
            self.reward,
            self.has_finished,
            self.goal_attributes,
            self.drone_rpy,
            self.drone_motor_thrust,
            self.drone_acc,
            self.drone_gyro,
            ACTIONS,
        )

    def render(self):
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.drone)
        while self.viewer.is_alive:
            self.viewer.render()


if __name__ == "__main__":
    # Sample Usage
    gym_env = DroneControlGym()

    while gym_env.step_count < 25:
        gym_env.step(ACTIONS[0])

    while gym_env.step_count < 50:
        gym_env.step(ACTIONS[15])

    while gym_env.step_count < 55:
        gym_env.step(ACTIONS[3])

    while gym_env.step_count < 70:
        gym_env.step(ACTIONS[15])

    while gym_env.step_count < 75:
        gym_env.step(ACTIONS[12])

    while gym_env.step_count < 120:
        gym_env.step(ACTIONS[15])

    while gym_env.step_count < 350:
        gym_env.step(ACTIONS[0])

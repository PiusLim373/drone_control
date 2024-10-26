#!/usr/bin/python3
import gym
import numpy as np
import mujoco
import mujoco_viewer
import os
import random
import logging
from custom_utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)
# Global variables
RANGE_LIMIT = 1
RANGE_BUFFER = 1
RESOLUTION = 1
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
APPROACH_MULTIPLIER = 3

(
    ACTION_COST,
    IDLE_COST,
    GOAL_REWARD,
    OUT_OF_BOUND_REWARD,
    FLIPPED_REWARD,
    SMOOTH_MOTION_REWARD,
) = (-1, -10, 3000, -3000, -3000, 1)
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
    def __init__(self, goal_pose=None, render=False):
        # Load the drone model and initialize the simulation
        self.model = mujoco.MjModel.from_xml_path(DRONE_MODEL_PATH)
        self.drone = mujoco.MjData(self.model)

        # Get the index of the IMU sensors (accelerometer and gyro) from the XML definition
        self.gyro_index = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")]
        self.acc_index = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc")]
        self.goal_id = self.model.geom("goal").id  # Get the index of the point geom
        self.renderflag = render
        if self.renderflag:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.drone)

        self.drone_acc = []
        self.drone_gyro = []
        self.drone_motor_thrust = None
        self.drone_position = [0.0, 0.0, 0.0]
        self.drone_rpy = None
        self.last_drone_rpy = None
        self.last_drone_motor_thrust = None
        self.last_drone_position = [0.0, 0.0, 0.0]
        self.motor_states = [
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
        ]

        # randomly initialize goal pose (x, y, z) within RANGE_LIMIT
        if goal_pose == None:
            self.goal_pose = [
                random.uniform(-RANGE_LIMIT / 2, RANGE_LIMIT / 2),
                random.uniform(-RANGE_LIMIT / 2, RANGE_LIMIT / 2),
                random.uniform(0.3, RANGE_LIMIT),
            ]
        else:
            self.goal_pose = goal_pose

        self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)  # Set the position of the point
        self.distance_to_goal = None
        self.last_distance_to_goal = None

        self.reward_counters = {"idle": 0, "action": 0, "approach": 0, "away": 0}

        self.has_finished = False
        mujoco.mj_step(self.model, self.drone)
        self.step_count = 1

    def _update_drone_data_from_sim(self):
        # Get the drone's current pose, orientation, and sensor readings from the simulation
        rpy_angles = quaternion_to_rpy(self.drone.xquat[1])
        self.drone_rpy = np.array(rpy_angles)
        self.drone_position = np.array(self.drone.xpos[1])
        self.drone_acc = self.drone.sensordata[self.acc_index : self.acc_index + 3]  # Accelerometer (x, y, z)
        self.drone_gyro = self.drone.sensordata[self.gyro_index : self.gyro_index + 3]  # Gyroscope (x, y, z)

    def _update_motor_thrust(self):
        # Calculate the average duty cycle for each motor and set the thrust values
        duty_cycle = [0] * 4
        thrust = [0] * 4
        for index, individual_motor_state in enumerate(self.motor_states):
            duty_cycle[index] = average(individual_motor_state)
            thrust[index] = duty_cycle[index] * FULL_THROTTLE
        self.drone.ctrl[:] = np.array(thrust)

    def _calculate_goal_attributes(self):
        if self.drone_rpy is None:
            self.drone_rpy = [0.0, 0.0, 0.0]  # Set to [0, 0, 0] if not initialized

        # Calculate vector difference between goal and drone's current position
        vector_to_goal_world = np.array(self.goal_pose) - np.array(self.drone_position)
        drone_rot_mat = rpy_to_mat(self.drone_rpy)

        # Transform the goal vector from the world frame to the drone's frame
        vector_to_goal_drone_frame = np.dot(
            np.linalg.inv(drone_rot_mat), vector_to_goal_world
        )  # Inverse R because we want world to drone frame

        # Calculate distance to goal in the drone's frame
        self.distance_to_goal = np.linalg.norm(vector_to_goal_drone_frame)

        # Normalize the vector to goal if the distance is greater than 0
        if self.distance_to_goal > 0:
            dx, dy, dz = vector_to_goal_drone_frame / self.distance_to_goal  # Normalize
        else:
            dx, dy, dz = 0.0, 0.0, 0.0  # If already at the goal
        return np.array([dx, dy, dz, self.distance_to_goal])
    
    
    def _rpy_to_rotation_matrix(self, roll, pitch, yaw):
        # Convert roll, pitch, yaw to rotation matrix
        roll, pitch, yaw = np.radians([roll, pitch, yaw])

        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # The final rotation matrix is R = R_z * R_y * R_x
        R = np.dot(R_z, np.dot(R_y, R_x))
        
        return R


    def _rpy_to_rotation_matrix(self, roll, pitch, yaw):
        # Convert roll, pitch, yaw to rotation matrix
        roll, pitch, yaw = np.radians([roll, pitch, yaw])

        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        # The final rotation matrix is R = R_z * R_y * R_x
        R = np.dot(R_z, np.dot(R_y, R_x))
        
        return R

    def _calculate_reward(self):
        # calculate reward based on the current state of the drone and if the drone has reached the goal
        # Check if the roll and pitch are close to zero (upright)
        if self.distance_to_goal < GOAL_TOLERANCE:
            logging.info("The drone is staying upright at the goal.")
            return True, GOAL_REWARD
        else:
            reward = 0
            finished = False

            # check if current duty cycle is equal to previous duty cycle and position is same
            if np.all(np.abs(self.drone_position - self.last_drone_position) < IDLE_POSITION_THRESHOLD):
                logging.debug("drone is idle in place")
                reward += IDLE_COST
                self.reward_counters["idle"] += 1
            else:
                logging.debug("drone took valid action")
                reward += ACTION_COST
                self.reward_counters["action"] += 1

                # if drone is flipped (or collided?), return FLIPPED_REWARD
                # REWARD = ACTION + FLIPPED
                if abs(self.drone_rpy[0]) > ROLL_THRESHOLD or abs(self.drone_rpy[1]) > PITCH_THRESHOLD:
                    logging.debug("drone has flipped")
                    reward += FLIPPED_REWARD
                    finished = True

                # Apply smooth motion rewards only if not flipped or out of bounds
                if not finished:
                    # Check for smooth angular motion (low angular velocity)
                    # REWARD = ACTION + SMOOTH
                    if self.distance_to_goal and self.last_distance_to_goal:
                        if self.distance_to_goal < self.last_distance_to_goal:
                            logging.debug("drone is approaching goal")
                            reward += APPROACH_MULTIPLIER / (self.distance_to_goal + 0.003)
                            self.reward_counters["approach"] += 1
                        elif self.distance_to_goal >= self.last_distance_to_goal:
                            logging.debug("drone is getting further from goal")
                            reward -= APPROACH_MULTIPLIER
                            self.reward_counters["away"] += 1

            return finished, reward

    def get_pose(self):
        # Returns the current pose of the drone (x, y, z, roll, pitch, yaw)
        self._update_drone_data_from_sim()
        return {
            "x": self.drone_position[0],
            "y": self.drone_position[1],
            "z": self.drone_position[2],
            "roll": self.drone_rpy[0],
            "pitch": self.drone_rpy[1],
            "yaw": self.drone_rpy[2],
        }

    def get_goal(self):
        # Returns the current goal position (x, y, z)
        return {"x": self.goal_pose[0], "y": self.goal_pose[1], "z": self.goal_pose[2]}

    def get_imu_reading(self):
        # returns the current accelerometer and gyroscope readings
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

    def get_all_state(self):
        combined_array = np.concatenate(
            (self.goal_attributes, self.drone_rpy, self.drone_motor_thrust)
        )
        observations = combined_array.tolist()
        return (
            self.reward,
            self.has_finished,
            observations,
        )

    def reset(self, goal_pose=None):
        # Reset simulation data
        mujoco.mj_resetData(self.model, self.drone)
        # Randomly initialize a new goal pose
        if goal_pose == None:
            self.goal_pose = [
                random.uniform(-RANGE_LIMIT / 2, RANGE_LIMIT / 2),
                random.uniform(-RANGE_LIMIT / 2, RANGE_LIMIT / 2),
                random.uniform(0.3, RANGE_LIMIT),
            ]
        else:
            self.goal_pose = goal_pose
        self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)
        self.goal_attributes = self._calculate_goal_attributes()

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
        # Reset the drone pose and orientation to defaults
        self.drone_position = [0.0, 0.0, 0.0]
        self.last_drone_position = [0.0, 0.0, 0.0]
        self.drone_rpy = None
        self.distance_to_goal = None
        self.last_distance_to_goal = None
        self.drone_last_ctrl = np.zeros(4)
        self.step_count = 1
        self.has_finished = False
        # Step the simulation once to apply the reset thrust values
        mujoco.mj_step(self.model, self.drone)
        self._update_drone_data_from_sim()
        self.drone_motor_thrust = np.array(self.drone.actuator_force)

        self.reward_counters = {"idle": 0, "action": 0, "approach": 0, "away": 0}

        # Return the initial observation and goal
        combined_array = np.concatenate(
            (self.goal_attributes, self.drone_rpy, self.drone_motor_thrust)
        )
        observations = combined_array.tolist()
        return observations

    def step(self, action):
        # Called by learning agent, step through the simulation with the given action, and get some new state back
        logging.debug(
            f"                         [{action[0]}] [{action[1]}]\n"
            f"                                  drone stepped with action   X\n"
            f"                                                           [{action[3]}] [{action[2]}]"
        )
        self.last_distance_to_goal = self.distance_to_goal
        # using action given, pop the first motor state and append the new motor state for each motor
        for index, individual_action in enumerate(action):
            self.motor_states[index].pop(0)
            self.motor_states[index].append(individual_action)
        self.last_drone_motor_thrust = self.drone_motor_thrust
        self._update_motor_thrust()

        # step through the simulator and get data
        mujoco.mj_step(self.model, self.drone)
        if self.renderflag:
            self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)  # Set the position of the point
            self.viewer.render()
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

    def render(self):
        # Render for visualization, press Esc to continue
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.drone)
        while self.viewer.is_alive:
            self.viewer.render()

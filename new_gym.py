#!/usr/bin/python3
import gymnasium as gym
from gymnasium import spaces
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
RESOLUTION = 10
FULL_THROTTLE = 7
DRONE_MODEL_PATH = os.path.join(os.getcwd(), "asset/skydio_x2/scene.xml")
ROLL_TARGET = 5  # degrees
PITCH_TARGET = 5  # degrees
ROLL_THRESHOLD = 170  # degrees
PITCH_THRESHOLD = 170  # degrees
HEIGHT_LOWER_LIMIT = 0.2  # m
IDLE_POSITION_THRESHOLD = 0.0005  # m
GYRO_SMOOTH_THRESHOLD = 0.05  # Threshold for angular velocity (rad/s)
ACC_SMOOTH_THRESHOLD = 0.1  # Threshold for linear acceleration (m/s^2)

TILT_THRESHOLD = 45  # degrees
MAX_TIMESTEPS = 300
MAX_IDLE_STEP = 50
GOAL_REWARD = 1
MAX_APPROACH_REWARD = 5
AWAY_MULTIPLIER = 1
IDLE_PENALTY = 0.1
FLIPPED_PENALTY = -2
GROUND_PENALTY = -10
GOAL_ZONE_MULTIPLIER = 10000  # Scaling factor within goal zone
APPROACH_MULTIPLIER = 5000  # Scaling factor for distance reward
TILT_PENALTY_MULTIPLER = 0.1  # Penalty scaling for tilt (flip avoidance)
TIME_TARGET = 10

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
    def __init__(self, level=1, goal_pose=None, render=False):
        super(DroneControlGym, self).__init__()
        # Load the drone model and initialize the simulation
        self.model = mujoco.MjModel.from_xml_path(DRONE_MODEL_PATH)
        self.drone = mujoco.MjData(self.model)

        self.action_space = spaces.Discrete(16)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        # Define z ranges for different levels
        self.z_ranges = {
            1: (0.3, 0.7),
            2: (0.3, 1.0),
            3: (0.3, 1.5),
            4: (0.3, 2.5),
            5: (0.3, 4.0),
            6: (0.3, 7.5)
        }
        
        # Define xy ranges for different levels
        self.xy_ranges = {
            1: 0.6,
            2: 0.9,
            3: 1.5,
            4: 2.5,
            5: 3.5,
            6: 5  # Adjust upper limit as needed
        }

        # Define goal tolerance for different levels
        self.tolerances = {
            1: 0.3,
            2: 0.25,
            3: 0.15,
            4: 0.1,
            5: 0.025,
            6: 0.01
        }

        self.current_level = level  # Track the current level
        
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    0,
                    -90,
                    -90,
                    -90,
                    0,
                    0,
                    0,
                    0,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    0.0,
                ]
            ),
            high=np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    np.inf,
                    90,
                    90,
                    90,
                    FULL_THROTTLE,
                    FULL_THROTTLE,
                    FULL_THROTTLE,
                    FULL_THROTTLE,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ]
            ),
            dtype=np.float32,
        )
        # Get the index of the IMU sensors (accelerometer and gyro) from the XML definition
        self.gyro_index = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")]
        self.acc_index = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc")]
        self.goal_id = self.model.geom("goal").id  # Get the index of the point geom
        self.drone_point_id = self.model.geom("drone").id  # Get the index of the point geom
        # Initialize variables
        self._init_vars()
        # self.goal_pose = None
        # self.drone_position = None
        if goal_pose == None:
            self._generate_goal()
        else:
            self.goal_pose = goal_pose

        self.renderflag = render
        if self.renderflag:
            self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)  # Set the position of the point
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.drone)

        self.success_count = 0
        self.fail_count = 0

        mujoco.mj_step(self.model, self.drone)
        self.step_count = 1

    def _update_drone_data_from_sim(self):
        # Get the drone's current pose, orientation, and sensor readings from the simulation
        rpy_angles = quaternion_to_rpy(self.drone.xquat[1])
        self.drone_rpy = np.array(rpy_angles)
        self.drone_position = np.array(self.drone.xpos[1])
        self.drone_acc = self.drone.sensordata[self.acc_index : self.acc_index + 3]  # Accelerometer (x, y, z)
        # print(f"acc: {self.drone_acc}")
        self.drone_gyro = self.drone.sensordata[self.gyro_index : self.gyro_index + 3]  # Gyroscope (x, y, z)
        # print(f"gyro: {self.drone_gyro}")

        if np.all(self.goal_vector_drone_frame) and np.all(self.last_goal_vector_drone_frame):
            self.drone_linvel = (self.goal_vector_drone_frame - self.last_goal_vector_drone_frame) / 0.01

    def _update_motor_thrust(self, is_reset=False, action=None):
        if not is_reset:
            for index, individual_action in enumerate(action):
                self.motor_states[index].pop(0)
                self.motor_states[index].append(individual_action)
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
        goal_vector_world = np.array(self.goal_pose) - np.array(self.drone_position)
        drone_rot_mat = rpy_to_mat(self.drone_rpy)

        # calculate goal unit vector in world frame
        goal_unit_vector_world = goal_vector_world / np.linalg.norm(goal_vector_world)

        # Get drone's Z-axis in world coordinates from its orientation
        # Assuming drone_orientation is a rotation matrix (3x3)
        drone_z_axis = drone_rot_mat[:, 2]  # Z-axis is the third column of the rotation matrix

        # Calculate the dot product between the drone's Z-axis and the unit goal vector
        # Ensure dot_product is between -1 and 1 (should be if normalized)
        dot_product = np.clip(np.dot(drone_z_axis, goal_unit_vector_world), -1.0, 1.0)

        # Calculate the angle between the two vectors in radians
        self.drone_angle_from_goal = np.arccos(dot_product)

        # Transform the goal vector from the world frame to the drone's frame
        # Inverse R because we want world to drone frame
        self.goal_vector_drone_frame = np.dot(np.linalg.inv(drone_rot_mat), goal_vector_world)

        # Calculate distance to goal in the drone's frame
        self.distance_to_goal = np.linalg.norm(self.goal_vector_drone_frame)

        # Normalize the vector to goal if the distance is greater than 0
        if self.distance_to_goal > 0:
            dx, dy, dz = self.goal_vector_drone_frame / self.distance_to_goal  # Normalize
        else:
            dx, dy, dz = 0.0, 0.0, 0.0  # If already at the goal
        return np.array([dx, dy, dz, self.distance_to_goal])

    def _generate_goal(self):
        """Randomly initialize goal pose (x, y, z) based on current level."""
        # Decide whether to use the current level or the previous level
        if self.current_level == 1 or random.random() < 0.6:  # 60% chance for last level
            level_used = self.current_level - 1 if self.current_level > 1 else self.current_level
        else:  # 40% chance for current level
            level_used = self.current_level

        # Get ranges and tolerance from the determined level
        xy_range = self.xy_ranges[level_used]
        z_range = self.z_ranges[level_used]
        self.goal_tolerance = self.tolerances[level_used]

        # Randomly generate x and y values within the selected xy rang
        self.goal_pose = [
            random.uniform(-xy_range / 2, xy_range / 2),
            random.uniform(-xy_range / 2, xy_range / 2),
            random.uniform(*z_range),
        ]

    def _calculate_reward(self):
        global XY_RANGE, RANGE_LIMIT, Z_RANGE, GOAL_TOLERANCE, TIME_TARGET, MAX_TIMESTEPS
        # calculate reward based on the current state of the drone and if the drone has reached the goal
        # Check if the roll and pitch are close to zero (upright)
        reward = 0
        terminated = False
        truncated = False

        if np.all(self.last_distance_to_goal):
            # When drone is within goal zone
            if self.distance_to_goal < self.goal_tolerance:
                self.time_in_goal += 1
                self.reward_counters["goal"] += 1
                reward += 20 * (self.time_in_goal)
                if self.last_distance_to_goal > self.distance_to_goal:
                    self.reward_counters["approach"] += 1
                    reward += min(200 * (self.last_distance_to_goal - self.distance_to_goal), 0.5)
                else:
                    self.reward_counters["away"] += 1
                    # reward += max(200 * (self.last_distance_to_goal - self.distance_to_goal), -0.5)

                # if self.time_in_goal > 0:
                #     self.success_count += 1
                #     XY_RANGE = min(XY_RANGE * 1.1, 3)  # XY range increases
                #     Z_RANGE = min(Z_RANGE * 1.01, 3)  # Z range increases
                #     GOAL_TOLERANCE = max(GOAL_TOLERANCE * 0.99, 0.15)  # Goal zone get smaller
                #     print(
                #         f"LEVEL {self.success_count}: XY_RANGE: {XY_RANGE}, Z_RANGE: {Z_RANGE}, GOAL_TOLERANCE: {GOAL_TOLERANCE}"
                #     )

                # level up challenges after being in goal for certain time steps
                # if self.time_in_goal >= TIME_TARGET:
                # self.fail_count = 0
                # logging.info("Drone has reached goal")
                # reward += 100
                # self.success_count += 1
                # XY_RANGE = min(XY_RANGE * 1.1, 3)  # XY range increases
                # Z_RANGE = min(Z_RANGE * 1.01, 3)  # Z range increases
                # GOAL_TOLERANCE = max(GOAL_TOLERANCE * 0.99, 0.15)  # Goal zone get smaller
                # TIME_TARGET += 1
                # while abs(np.linalg.norm(self.goal_pose - self.drone_position)) < GOAL_TOLERANCE:
                #     self._generate_goal()
                # MAX_TIMESTEPS = self.step_count + (self.distance_to_goal * 200)
                # print(
                #     f"LEVEL {self.success_count}: XY_RANGE: {XY_RANGE}, Z_RANGE: {Z_RANGE}, GOAL_TOLERANCE: {GOAL_TOLERANCE}"
                # )
                # finished = True  # Mission completed
            else:
                # penalty for being idle unless in goal
                if np.all(np.abs(self.drone_position - self.last_drone_position) < IDLE_POSITION_THRESHOLD):
                    pass
                else:
                    reward += 0.3 / (self.distance_to_goal + 0.003)
                    if self.last_distance_to_goal > self.distance_to_goal:
                        self.reward_counters["approach"] += 1
                        reward += min(100 * (self.last_distance_to_goal - self.distance_to_goal), 0.5)
                    else:
                        self.reward_counters["away"] += 1
                        reward += max(100 * (self.last_distance_to_goal - self.distance_to_goal), -0.5)
                    if abs(self.drone_rpy[0]) > TILT_THRESHOLD or abs(self.drone_rpy[1]) > TILT_THRESHOLD:
                        # Excess tilt penalty
                        self.reward_counters["tilt"] += 1
                        reward += max(-0.02 * (abs(self.drone_rpy[0]) + abs(self.drone_rpy[1])), -0.2)
                    else:
                        reward += 0.2

        if self.drone_position[2] < 0.1:
            # Penalty for being too near to ground
            logging.info("Drone has fallen to ground")
            self.fail_count += 1
            self.reward_counters["altitude"] += 1
            reward -= 100
            terminated = True
        else:
            # Check for high gyro velocity
            if abs(self.drone_gyro[0]) > 10.0 or abs(self.drone_gyro[1]) > 10.0 or abs(self.drone_gyro[2]) > 5.0:
                # logging.info("Drone is rotating too fast.")
                self.reward_counters["spin"] += 1
                reward += max(-0.05 * max(abs(self.drone_gyro)), -0.5)
            else:
                # reward for staying alive nicely
                reward += 1

        # # Check for flipping
        if abs(self.drone_rpy[0]) > ROLL_THRESHOLD or abs(self.drone_rpy[1]) > PITCH_THRESHOLD:
            logging.info("Drone has flipped.")
            reward += -100  # Severe penalty for flipping
            self.fail_count += 1
            terminated = True

        # Check for out of bound
        if self.distance_to_goal > 5.0:
            logging.info("Drone is out of bound.")
            reward += -100
            self.fail_count += 1
            terminated = True

        # Regenerate goal pose if too many fails
        if self.fail_count >= 10:
            logging.info("Failed 10 times, regenerating goal")
            self._generate_goal()
            self.fail_count = 0

        if self.step_count >= MAX_TIMESTEPS:
            # logging.info("Timed out")
            self.fail_count += 1
            truncated = True

        return terminated, truncated, reward

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
            (
                self.goal_attributes,
                self.drone_rpy,
                self.drone_motor_thrust,
                self.drone_acc,
                self.drone_gyro,
                self.drone_linvel,
                np.array([self.drone_position[2]]),
            )
        )
        observations = combined_array
        return (observations, self.reward, self.terminated, self.truncated, self.reward_counters)

    def reset(self, seed=None, goal_pose=None):
        global XY_RANGE, RANGE_LIMIT, Z_RANGE, GOAL_TOLERANCE, TIME_TARGET
        if self.time_in_goal > 0:
            self.success_count += 1
            logging.info(f"Been in goal {self.time_in_goal / 100} seconds, regenerating next reset")
            XY_RANGE = min(XY_RANGE * 1.1, 3)  # XY range increases
            Z_RANGE = min(Z_RANGE * 1.01, 3)  # Z range increases
            GOAL_TOLERANCE = max(GOAL_TOLERANCE * 0.99, 0.15)  # Goal zone get smaller
            self._generate_goal()
            print(
                f"LEVEL {self.success_count}: XY_RANGE: {XY_RANGE}, Z_RANGE: {Z_RANGE}, GOAL_TOLERANCE: {GOAL_TOLERANCE}"
            )

        # Reset simulation data
        mujoco.mj_resetData(self.model, self.drone)
        # Randomly initialize a new goal pose
        if goal_pose == None:
            pass
        else:
            self.goal_pose = goal_pose
        self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)

        # Initialize variables:
        self._init_vars()
        # Step the simulation once to apply the reset thrust values
        self._update_motor_thrust(is_reset=True, action=None)
        mujoco.mj_step(self.model, self.drone)
        self._update_drone_data_from_sim()
        self.goal_attributes = self._calculate_goal_attributes()
        self.drone_motor_thrust = np.array(self.drone.actuator_force)

        # Return the initial observation and goal
        combined_array = np.concatenate(
            (
                self.goal_attributes,
                self.drone_rpy,
                self.drone_motor_thrust,
                self.drone_acc,
                self.drone_gyro,
                self.drone_linvel,
                np.array([self.drone_position[2]]),
            )
        )
        observations = combined_array.tolist()
        return combined_array, self.reward_counters

    def _init_vars(self):
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        self.time_in_goal = 0
        self.reward_counters = {"goal": 0, "approach": 0, "away": 0, "idle": 0, "altitude": 0, "tilt": 0, "spin": 0}
        self.drone_acc = []
        self.drone_gyro = []
        self.drone_motor_thrust = None
        self.drone_position = self.drone.geom_xpos[self.model.body("x2").id]
        self.last_drone_position = self.drone.geom_xpos[self.model.body("x2").id]
        self.drone_rpy = None
        self.goal_vector_drone_frame = None
        self.last_goal_vector_drone_frame = None
        self.last_drone_motor_thrust = None
        self.distance_to_goal = None
        self.last_distance_to_goal = None
        self.drone_angle_from_goal = None
        self.last_drone_angle_from_goal = None
        self.drone_linvel = [0, 0, 0]
        self.drone_angvel = None
        # self.motor_states = [
        #     [0] * RESOLUTION,
        #     [0] * RESOLUTION,
        #     [0] * RESOLUTION,
        #     [0] * RESOLUTION,
        # ]
        self._generate_goal()
        self.motor_states = [
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        ]

    def step(self, action):
        throttle_values = ACTIONS[action]
        # Called by learning agent, step through the simulation with the given action, and get some new state back
        logging.debug(
            f"                         [{throttle_values[0]}] [{throttle_values[1]}]\n"
            f"                                  drone stepped with action   X\n"
            f"                                                           [{throttle_values[3]}] [{throttle_values[2]}]"
        )
        self.last_distance_to_goal = self.distance_to_goal
        self.last_drone_angle_from_goal = self.drone_angle_from_goal
        self.last_drone_position = np.copy(self.drone_position)
        self.last_drone_motor_thrust = self.drone_motor_thrust
        # using action given, pop the first motor state and append the new motor state for each motor

        self._update_motor_thrust(is_reset=False, action=throttle_values)

        # step through the simulator and get data
        mujoco.mj_step(self.model, self.drone)
        if self.renderflag:
            self.drone.geom_xpos[self.drone_point_id] = np.copy(self.drone_position)
            self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)  # Set the position of the point
            self.viewer.render()
        self.step_count += 1
        self._update_drone_data_from_sim()
        self.drone_motor_thrust = np.array(self.drone.actuator_force)

        self.goal_attributes = self._calculate_goal_attributes()  # return list of [dx, dy, dz, d]
        self.terminated, self.truncated, self.reward = self._calculate_reward()  # return bool and float
        logging.debug(f"Step: {self.step_count}")
        logging.debug(f"Reward: {self.reward}")
        logging.debug(f"Finished episode: {self.terminated}")
        logging.debug(f"Finished episode: {self.truncated}")
        logging.debug(f"Goal attributes: {self.goal_attributes}")
        logging.debug(f"Current drone RPY: {self.drone_rpy}")
        logging.debug(f"Current drone motor thrust: {self.drone_motor_thrust}")
        logging.debug(f"Current drone accelerations:{self.drone_acc}")
        logging.debug(f"Current drone gyro velocities{self.drone_gyro}")

        return self.get_all_state()

    def increase_difficulty(self):
        """Increase level for curriculum learning."""
        if self.current_level < 6:  # Prevent exceeding the maximum level
            self.current_level += 1
            logging.info(f"Adjusted difficulty to level: {self.current_level}")
        else:
            logging.info("Maximum level reached. No further difficulty adjustment.")
            
    def decrease_difficulty(self):
        """Decrease level for curriculum learning if performance is poor."""
        if self.current_level > 1:  # Prevent going below level 1
            self.current_level -= 1
            logging.info(f"Decreased difficulty to level: {self.current_level}")
        else:
            logging.info("Minimum level reached. No further difficulty adjustment.")
    
    def render(self):
        # Render for visualization, press Esc to continue
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.drone)
        while self.viewer.is_alive:
            self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)
            self.drone.geom_xpos[self.drone_id] = np.copy(self.drone_position)
            self.viewer.render()

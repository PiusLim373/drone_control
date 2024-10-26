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
XY_RANGE = 0.2
Z_RANGE = 0.7
RANGE_LIMIT = 1
RANGE_BUFFER = 1
RESOLUTION = 1
GOAL_TOLERANCE = 0.3  # if the drone is within 0.5m of the goal, it is considered to have reached the goal
FULL_THROTTLE = 6
DRONE_MODEL_PATH = os.path.join(os.getcwd(), "asset/skydio_x2/scene.xml")
ROLL_TARGET = 5  # degrees
PITCH_TARGET = 5  # degrees
ROLL_THRESHOLD = 90  # degrees
PITCH_THRESHOLD = 90  # degrees
HEIGHT_LOWER_LIMIT = 0.2  # m
IDLE_POSITION_THRESHOLD = 0.0001  # m
GYRO_SMOOTH_THRESHOLD = 0.05  # Threshold for angular velocity (rad/s)
ACC_SMOOTH_THRESHOLD = 0.1  # Threshold for linear acceleration (m/s^2)

TILT_THRESHOLD = 30  # degrees
MAX_TIMESTEPS = 300
MAX_IDLE_STEP = 50
GOAL_REWARD = 20
MAX_APPROACH_REWARD = 5
AWAY_MULTIPLIER = 1
IDLE_PENALTY = -3
FLIPPED_PENALTY = -2000
GROUND_PENALTY = -20
GOAL_ZONE_MULTIPLIER = 1000  # Scaling factor within goal zone
APPROACH_MULTIPLIER = 500  # Scaling factor for distance reward
TILT_PENALTY_MULTIPLER = 1  # Penalty scaling for tilt (flip avoidance)

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
        self.drone_id = self.model.geom("drone").id  # Get the index of the point geom
        self.renderflag = render
        if self.renderflag:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.drone)

        self.success_count = 0
        self.time_in_goal = 0
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
                random.uniform(-XY_RANGE / 2, XY_RANGE / 2),
                random.uniform(-XY_RANGE / 2, XY_RANGE / 2),
                random.uniform(0.3, Z_RANGE),
            ]
        else:
            self.goal_pose = goal_pose
        self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)  # Set the position of the point
        self.distance_to_goal = None
        self.last_distance_to_goal = None
        self.drone_angle_from_goal = None
        self.last_drone_angle_from_goal = None

        self.reward_counters = {"goal": 0, "approach": 0, "away": 0, "idle": 0, "ground": 0, "tilt": 0}

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
        goal_vector_drone_frame = np.dot(np.linalg.inv(drone_rot_mat), goal_vector_world)

        # Calculate distance to goal in the drone's frame
        self.distance_to_goal = np.linalg.norm(goal_vector_drone_frame)

        # Normalize the vector to goal if the distance is greater than 0
        if self.distance_to_goal > 0:
            dx, dy, dz = goal_vector_drone_frame / self.distance_to_goal  # Normalize
        else:
            dx, dy, dz = 0.0, 0.0, 0.0  # If already at the goal
        return np.array([dx, dy, dz, self.distance_to_goal])

    def _calculate_reward(self):
        global XY_RANGE, RANGE_LIMIT, Z_RANGE
        # calculate reward based on the current state of the drone and if the drone has reached the goal
        # Check if the roll and pitch are close to zero (upright)
        reward = 0
        finished = False

        if np.all(self.last_distance_to_goal):
            # only counter here, not for reward
            if self.last_distance_to_goal > self.distance_to_goal:
                self.reward_counters["approach"] += 1
            else:
                self.reward_counters["away"] += 1

            # When drone is within goal zone
            if self.distance_to_goal < GOAL_TOLERANCE:
                self.time_in_goal += 1
                self.reward_counters["goal"] += 1
                # flat reward for being in goal
                reward += GOAL_REWARD
                # scale reward for staying in goal
                reward += 5 * (self.time_in_goal)
                # scale reward for closer to center of goal
                reward += GOAL_ZONE_MULTIPLIER * (self.last_distance_to_goal - self.distance_to_goal)

                # level up challenges after being in goal for certain time steps
                if self.time_in_goal >= 50:
                    self.success_count += 1
                    XY_RANGE *= 1.02  # XY range increases
                    Z_RANGE *= 1.02  # Z range increases
                    GOAL_TOLERANCE * 0.98  # Goal zone get smaller
                    self.goal_pose = [
                        random.uniform(-XY_RANGE / 2, XY_RANGE / 2),
                        random.uniform(-XY_RANGE / 2, XY_RANGE / 2),
                        random.uniform(0.2, Z_RANGE),
                    ]
                    print(
                        f"LEVEL {self.success_count}: XY_RANGE: {XY_RANGE}, Z_RANGE: {Z_RANGE}, GOAL_TOLERANCE: {GOAL_TOLERANCE}"
                    )
                    finished = True  # Mission completed
            else:
                # positive reward for moving toward goal, negative reward for away
                reward += APPROACH_MULTIPLIER * (self.last_distance_to_goal - self.distance_to_goal)

                # penalty for being idle unless in goal
                if np.all(np.abs(self.drone_position - self.last_drone_position) < IDLE_POSITION_THRESHOLD):
                    logging.debug("Drone is idle.")
                    self.reward_counters["idle"] += 1
                    reward += (self.reward_counters["idle"]) * IDLE_PENALTY
                # if self.drone_angle_from_goal < self.last_drone_angle_from_goal:
                #     # Cap the alignment reward
                #     logging.debug("Drone is aligning towards the goal.")
                #     reward += 0.5 * (self.drone_angle_from_goal)
                #     self.reward_counters["aligned"] += 1

        if np.all(self.drone_rpy):
            if abs(self.drone_rpy[0]) > TILT_THRESHOLD or abs(self.drone_rpy[1]) > TILT_THRESHOLD:
                # Excess tilt penalty
                self.reward_counters["tilt"] += 1
                reward += -TILT_PENALTY_MULTIPLER * (abs(self.drone_rpy[0]) + abs(self.drone_rpy[1]))

        if self.drone_position[2] < HEIGHT_LOWER_LIMIT:
            # penalty for being too near to ground
            self.reward_counters["ground"] += 1
            reward += GROUND_PENALTY

        if self.step_count >= MAX_TIMESTEPS:
            logging.info("Timed out")
            finished = True

        # Case 5: Check for flipping or collision
        if abs(self.drone_rpy[0]) > ROLL_THRESHOLD or abs(self.drone_rpy[1]) > PITCH_THRESHOLD:
            logging.info("Drone has flipped.")
            reward += FLIPPED_PENALTY  # Severe penalty for flipping
            finished = True

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
        # combined_array = np.concatenate(
        #     (self.goal_attributes, self.drone_rpy, self.drone_motor_thrust, self.drone_acc, self.drone_gyro)
        # )
        combined_array = np.concatenate((self.goal_attributes, self.drone_rpy, self.drone_motor_thrust))
        observations = combined_array.tolist()
        return (self.reward, self.has_finished, observations)

    def reset(self, goal_pose=None):
        # Reset simulation data
        mujoco.mj_resetData(self.model, self.drone)
        # Randomly initialize a new goal pose
        if goal_pose == None:
            pass
        else:
            self.goal_pose = goal_pose
        self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)
        self.goal_attributes = self._calculate_goal_attributes()
        self.time_in_goal = 0
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
        self.drone_angle_from_goal = None
        self.last_drone_angle_from_goal = None
        self.drone_last_ctrl = np.zeros(4)
        self.step_count = 1
        self.has_finished = False
        # Step the simulation once to apply the reset thrust values
        mujoco.mj_step(self.model, self.drone)
        self._update_drone_data_from_sim()
        self.drone_motor_thrust = np.array(self.drone.actuator_force)

        self.reward_counters = {"goal": 0, "approach": 0, "away": 0, "idle": 0, "ground": 0, "tilt": 0}

        # Return the initial observation and goal
        combined_array = np.concatenate((self.goal_attributes, self.drone_rpy, self.drone_motor_thrust))
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
        self.last_drone_angle_from_goal = self.drone_angle_from_goal
        self.last_drone_position = np.copy(self.drone_position)
        # using action given, pop the first motor state and append the new motor state for each motor
        for index, individual_action in enumerate(action):
            self.motor_states[index].pop(0)
            self.motor_states[index].append(individual_action)
        self.last_drone_motor_thrust = self.drone_motor_thrust
        self._update_motor_thrust()

        # step through the simulator and get data
        mujoco.mj_step(self.model, self.drone)
        if self.renderflag:
            self.drone.geom_xpos[self.drone_id] = np.copy(self.drone_position)
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
            self.drone.geom_xpos[self.goal_id] = np.copy(self.goal_pose)
            self.drone.geom_xpos[self.drone_id] = np.copy(self.drone_position)
            self.viewer.render()

#!/usr/bin/python3
import gym
import numpy as np
import mujoco
import mujoco_viewer
import os
import random
from custom_utils import *

# Global variables
RANGE_LIMIT = 10
RESOLUTION = 10
GOAL_TOLERANCE = 0.1  # if the drone is within 0.1m of the goal, it is considered to have reached the goal
FULL_THROTTLE = 7
DRONE_MODEL_PATH = os.path.join(os.getcwd(), "asset/skydio_x2/scene.xml")


ACTION_COST, IDLE_COST, GOAL_REWARD, OUT_OF_BOUND_REWARD, FLIPPED_REWARD = -0.1, -0.2, 1.0, -1.0, -2.0
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
        self.drone_pose = None
        self.drone_rpy = None
        self.drone_motor_thrust = None
        self.distance_to_goal = None
        self.has_finished = False

        self.step_count = 1

    def _update_drone_pose_from_sim(self):
        rpy_angles = quaternion_to_rpy(self.drone.xquat[1])
        self.drone_rpy = list(rpy_angles)
        self.drone_pose = list(self.drone.xpos[1])

    def _update_motor_thrust(self):
        duty_cycle = [0] * 4
        thrust = [0] * 4
        for index, individual_motor_state in enumerate(self.motor_states):
            duty_cycle[index] = average(individual_motor_state)
            thrust[index] = duty_cycle[index] * FULL_THROTTLE
        self.drone.ctrl[:] = np.array(thrust)

    def _calculate_goal_attributes(self):
        # Calculate the distance between drone_pose and goal_pose
        self.distance_to_goal = np.linalg.norm(np.array(self.goal_pose) - np.array(self.drone_pose))
    
        # Calculate the vector difference between goal and current position
        vector_to_goal = np.array(self.goal_pose) - np.array(self.drone_pose)
    
        # Get the absolute distances in x, y, and z directions 
        dx, dy, dz = vector_to_goal
    
        # Return the unit vector components [dx, dy, dz] and the distance to goal
        return [dx, dy, dz, self.distance_to_goal]

    def _calculate_reward(self):
        # calculate reward based on the current state of the drone and if the drone has reached the goal
        if self.distance_to_goal < GOAL_TOLERANCE:
            # drone mustn't be flipped or collided also, please implement this
            return True, GOAL_REWARD
        else:
            # if drone is flipped (or collided?), return FLIPPED_REWARD
            # if drone is out of RANGE_LIMIT return OUT_OF_BOUND_REWARD, not sure need this or not
            # else return ACTION_COST
            # how about IDLE_COST?
            reward = 0  # placeholder, please implement this
            return False, reward

    def get_pose(self):
        self._update_drone_pose_from_sim()
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
    
    def reset(self):
        # Reset the simulation state to the initial configuration
        mujoco.mj_resetData(self.model, self.drone)
        
        # # Reset drone position and orientation manually (redundant for now)
        # self.drone.qpos[:3] = np.array([0.0, 0.0, 0.1])  # Set (x, y, z) position
        # self.drone.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # Set quaternion (w, x, y, z) for orientation

        # Reset motor states
        self.motor_states = [
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
        ]

        # Randomly initialize a new goal pose
        self.goal_pose = [
            random.uniform(0.0, RANGE_LIMIT),
            random.uniform(0.0, RANGE_LIMIT),
            random.uniform(0.1, RANGE_LIMIT),
        ]

        # Reset the drone pose and orientation
        self.drone_pose = None
        self.drone_rpy = None
        self.distance_to_goal = None
        
        # Reset the thrust to 0 for all motors
        self.drone.ctrl[:] = np.zeros(4)
    
        # Step the simulation to apply the reset thrust
        mujoco.mj_step(self.model, self.drone)

        self.has_finished = False  # Reset the has_finished flag
        self.step_count = 1 # Reset the step count

        # Update the drone pose after reset
        self._update_drone_pose_from_sim()

        # Return the initial observation and goal
        return self.get_pose(), self.get_goal()


    def step(self, action):
        # using action given, pop the first motor state and append the new motor state for each motor
        for index, individual_action in enumerate(action):
            self.motor_states[index].pop(0)
            self.motor_states[index].append(individual_action)
        self._update_motor_thrust()
        mujoco.mj_step(self.model, self.drone)
        self.step_count += 1

        self._update_drone_pose_from_sim()
        self.drone_motor_thrust = list(self.drone.actuator_force)

        print(f"quadcopter xyz: {self.drone_pose}")
        print(f"quadcopter rpy: {self.drone_rpy}")

        self.goal_attributes = self._calculate_goal_attributes()  # return list of [dx, dy, dz, d]
        self.has_finished, self.reward = self._calculate_reward()  # return bool and float

        # eg 2.0, True, [0.3, 0.4, 0.1, 5.0], [33.0, 44.0, 55.0], [3.5, 3.5, 3.6, 3.7]
        return self.reward, self.has_finished, self.goal_attributes, self.drone_rpy, self.drone_motor_thrust

    def get_last_state(self):
        return self.reward, self.has_finished, self.goal_attributes, self.drone_rpy, self.drone_motor_thrust

    def render(self):
        viewer = mujoco_viewer.MujocoViewer(self.model, self.drone)
        print("Request to render image, press Ecs to continue")
        while viewer.is_alive:
            viewer.render()


if __name__ == "__main__":
    # Sample Usage
    gym_env = DroneControlGym()

    print(f"step: {gym_env.step_count}")
    while gym_env.step_count < 25:
        print(f"step: {gym_env.step_count}, drone pose: {gym_env.get_pose()} state: {gym_env.step(ACTIONS[0])}")
    gym_env.render()

    print(f"step: {gym_env.step_count}")
    while gym_env.step_count < 70:
        print(f"step: {gym_env.step_count}, drone pose: {gym_env.get_pose()} , state: {gym_env.step(ACTIONS[15])}")
    gym_env.render()

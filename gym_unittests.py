import unittest
import mujoco  
import numpy as np
import math
from drone_control_gym import DroneControlGym, ACTIONS  # Importing the drone control environment and action set

class TestDroneControlGym(unittest.TestCase):
    def setUp(self):
        """Set up the environment for testing."""
        self.gym_env = DroneControlGym()
        print(f"Running test: {self._testMethodName}")
        
    def rpy_to_quaternion(self, rpy):
        roll, pitch, yaw = [math.radians(angle) for angle in rpy]
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qw, qx, qy, qz]
    
    def set_drone_pose(self, position, rpy):
        # Set the drone position
        self.gym_env.drone_position = np.array(position)
        self.gym_env.drone.xpos[1][:3] = self.gym_env.drone_position  # Update drone position in MuJoCo

        # Set the drone orientation (rpy: roll, pitch, yaw)
        self.gym_env.drone_rpy = np.array(rpy)
        quat = self.rpy_to_quaternion(rpy)  # Convert rpy to quaternion (make sure this is defined elsewhere)
        self.gym_env.drone.xquat[1][:] = quat  # Update drone orientation in MuJoCo

    def test_drone_reaches_goal(self):
        # Set the goal position
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        # Set the drone's position and orientation (rpy: [roll, pitch, yaw])
        self.set_drone_pose([1.0, 1.0, 1.0], [0, 0, 0])  
        # Ensure the simulation processes the new pose
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        # The drone should be considered as having reached the goal
        self.assertTrue(self.gym_env.has_finished) # Should give termination signal
        self.assertEqual(self.gym_env.reward, 100.0) # Goal Reward: 100
        
    def test_goal_tolerance(self):
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        self.set_drone_pose([1.0, 1.09, 1.0], [4.99, 4.99, 0])  # Within tolerance of 0.1 within roll, pitch threshold of 5 degrees
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertTrue(self.gym_env.has_finished)
        self.assertEqual(self.gym_env.reward, 100.0) 

    def test_goal_tolerance_exceed_position_rpy(self):
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        self.set_drone_pose([1.0, 1.01, 1.0], [5, 0, 0])  # At tolerance of 0.1 and roll, pitch threshold of 5 degrees
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertFalse(self.gym_env.has_finished) # Should not give termination signal
        self.assertEqual(self.gym_env.reward, 0)  # Reward: action cost + smooth flight reward = (-2) + 2 = 0
        
    def test_goal_tolerance_exceed_rpy(self):
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        self.set_drone_pose([1.0, 1.0, 1.0], [5, 0, 0])  # Within tolerance of 0.1 with zero orientation
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertFalse(self.gym_env.has_finished)
        self.assertEqual(self.gym_env.reward, 0)
        
    def test_goal_tolerance_exceed_position(self):
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        self.set_drone_pose([1.0, 1.1, 1.0], [0, 0, 0])  # Within tolerance of 0.1 with zero orientation
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertFalse(self.gym_env.has_finished)
        self.assertEqual(self.gym_env.reward, 0)
        
    def test_drone_flips_over(self):
        self.set_drone_pose([0, 0, 1], [90.01, 0.0, 0.0])  # Roll beyond threshold of 90 degree
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertTrue(self.gym_env.has_finished) # Should give termination signal
        self.assertEqual(self.gym_env.reward, -52.0) # Reward: Flipped reward + action cost = (-50) + (-2) = -52
        
    def test_drone_flips_over_01(self):
        self.set_drone_pose([0, 0, 1], [0.0, 90.0, 0.0])  # Roll at exact threshold of 90 degree while not exceeding
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertFalse(self.gym_env.has_finished) # Should not give termination signal
        self.assertEqual(self.gym_env.reward, 0) # Reward: action cost + smooth flight reward = (-2) + 2 = 0
           
    def test_drone_goes_out_of_bounds(self):
        self.set_drone_pose([6.0, 6.0, 11.0], [0.0, 0.0, 0.0])  # Position at out of bounds limit of X: 6, Y: 6, Z: 11
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertTrue(self.gym_env.has_finished) # Should give termination signal
        self.assertEqual(self.gym_env.reward, -52.0) # Reward: Out of bounds reward + action cost  = (-50) + (-2) = -52

    def test_drone_goes_out_of_bounds_01(self):
        self.set_drone_pose([5.99, 5.99, 10.99], [0.0, 0.0, 0.0])  # Position before out of bounds limit of X: 6, Y: 6, Z: 11
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertFalse(self.gym_env.has_finished) # Should not give termination signal
        self.assertEqual(self.gym_env.reward, 0) # Reward: action cost + smooth flight reward = (-2) + 2 = 0
        
    def test_drone_goes_out_of_bounds_flips_over(self):
        self.set_drone_pose([10.01, 0, 1], [90.01, 0.0, 0.0])  # Both position and roll angle exceed thresholds
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertTrue(self.gym_env.has_finished) # Should give termination signal
        self.assertEqual(self.gym_env.reward, -102.0) # Reward: Out of bounds reward + flipped reward + action cost  = (-50) + (-50) + (-2) = -102
    
    def test_random_goal_at_boundary(self):
        self.gym_env.goal_pose = np.array([10.0, 10.0, 10.0])
        self.set_drone_pose([10.0, 10.0, 10.0], [0, 0, 0])  
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertTrue(self.gym_env.has_finished) # Should give termination signal
        self.assertEqual(self.gym_env.reward, 100.0) # Goal Reward: 100

    def test_idle(self):
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])  
        # Run the simulation with zero thrust for 50 steps
        for _ in range(50):
            self.gym_env.step(ACTIONS[0])  # Apply zero thrust continuously
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertFalse(self.gym_env.has_finished) # Should give termination signal
        self.assertEqual(self.gym_env.reward, -4.0) # Idle Cost: -4
        
    def test_max_thrust_all_motors(self):
        # Apply maximum thrust to all motors (ACTIONS[15] represents full thrust on all 4 motors)
        for _ in range(50):
            self.gym_env.step(ACTIONS[15])  # Apply maximum thrust continuously
        self.gym_env.has_finished, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertFalse(self.gym_env.has_finished)
        self.assertGreater(self.gym_env.drone_position[2], 0.5) # Ensure the drone has moved upwards (in the z-direction)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

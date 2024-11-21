import unittest
import mujoco  
import numpy as np
import math
from drone_control_gym import *

class TestDroneControlGym(unittest.TestCase):
    def setUp(self):
        """Set up the environment for testing."""
        self.gym_env = DroneControlGym()
        print(f"Running test: {self._testMethodName}")
        
        # Initialize last_distance_to_goal (initialize it as a numpy array with the same shape)
        self.gym_env.last_distance_to_goal = np.array([1.0])
        
        # Mock the drone's gyroscope and accelerometer data
        self.gym_env.drone_gyro = [0.0, 0.0, 0.0]  # Default values for gyro (x, y, z)
        self.gym_env.drone_acc = [0.0, 0.0, 0.0]  # Default values for accelerometer (x, y, z)
        
        # Set default pose for the drone to avoid errors in reward calculation
        self.set_drone_pose([1.0, 1.0, 1.0], [0, 0, 0])  # Position and orientation

    def rpy_to_quaternion(self, rpy):
        roll, pitch, yaw = [math.radians(angle) for angle in rpy]
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qw, qx, qy, qz]
    
    def set_drone_pose(self, position, rpy):
        """Set the drone position and orientation."""
        self.gym_env.drone_position = np.array(position)
        self.gym_env.drone.xpos[1][:3] = self.gym_env.drone_position  # Update drone position in MuJoCo

        # Set the drone orientation (rpy: roll, pitch, yaw)
        self.gym_env.drone_rpy = np.array(rpy)
        quat = self.rpy_to_quaternion(rpy)  # Convert rpy to quaternion
        self.gym_env.drone.xquat[1][:] = quat  # Update drone orientation in MuJoCo

    def test_drone_reaches_goal(self):
        """Test if the drone can reach the goal."""
        # Set the goal position
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        # Set the drone's position and orientation
        self.set_drone_pose([1.0, 1.0, 1.0], [0, 0, 0])  
        # Ensure the simulation processes the new pose
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        goal_reward = min(GOAL_ZONE_MULTIPLIER * self.gym_env.time_in_goal, 10)
        approach_reward = min(APPROACH_MULTIPLIER * (self.gym_env.last_distance_to_goal - self.gym_env.distance_to_goal), 3.0)
        expected_reward = goal_reward + approach_reward  +0.2 + 1   # Goalreward + approaching reward + alive reward 
        # Check if the drone has not yet terminated and is in goal
        self.assertFalse(has_terminated or has_truncated)  # Should not give termination signal
        self.assertEqual(self.gym_env.reward, expected_reward)  # Goal Reward
        
    def test_goal_tolerance(self):
        """Test if the drone is within the goal tolerance."""
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        # Set the drone's position just within the goal tolerance
        self.set_drone_pose([1.0, 1.29, 1.0], [0, 0, 0])  # Within tolerance of 0.1, roll/pitch threshold 5 degrees
        # Step the simulation
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        # Calculate expected reward (within goal tolerance, so should get goal reward)
        goal_reward = min(GOAL_ZONE_MULTIPLIER * self.gym_env.time_in_goal, 10)
        approach_reward = min(APPROACH_MULTIPLIER * (self.gym_env.last_distance_to_goal - self.gym_env.distance_to_goal), 3.0)
        expected_reward = goal_reward + approach_reward +0.2 + 1   # Goalreward + approaching reward + alive reward
        # Check if the drone is within goal tolerance
        self.assertFalse(has_terminated or has_truncated)  # Should not give termination signal
        # Check that the reward matches the expected value for reaching the goal
        self.assertEqual(self.gym_env.reward, expected_reward)

    def test_goal_tolerance_exceed_position(self): 
        """Test if the drone is outside the goal tolerance."""
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        # Set the drone's position outside the goal tolerance
        self.set_drone_pose([1.0, 1.9, 1.0], [0, 0, 0])  # Exceeds tolerance of 0.1, zero orientation
        # Step the simulation
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        approach_reward = min(APPROACH_MULTIPLIER * (self.gym_env.last_distance_to_goal - self.gym_env.distance_to_goal), 3.0)
        expected_reward = approach_reward +0.2 + 1   # Approaching reward + alive reward
        self.assertFalse(has_terminated or has_truncated)  # No termination signal for out-of-tolerance position
        # Check if the drone has not finished, since it's outside the goal tolerance
        self.assertEqual(self.gym_env.reward, expected_reward)
        
    def test_drone_flips_over(self):
        """Test if the drone flips when roll exceeds threshold (170 degrees)."""
        self.set_drone_pose([1.0, 1.0, 1.0], [170.01, 0.0, 0.0])  # Roll beyond 170 degrees threshold
        # Step the simulation
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        # Expected reward 
        tilt_reward = max(TILT_MULTIPLIER * (abs(self.gym_env.drone_rpy[0]) + abs(self.gym_env.drone_rpy[1])), -0.2)
        away_reward = max((APPROACH_MULTIPLIER/2) * (self.gym_env.last_distance_to_goal - self.gym_env.distance_to_goal), -3.0)
        expected_reward = FLIP_PENALTY + tilt_reward + away_reward + + 1   # Flip penalty + tilt penalty + away penalty + alive reward
        # Check if the drone has flipped (should trigger termination)
        self.assertTrue(has_terminated)  # Should be terminated due to flip
        self.assertFalse(has_truncated)  
        self.assertEqual(self.gym_env.reward, expected_reward)  

    def test_drone_flips_over_01(self):
        """Test if the drone remains alive at roll threshold (170 degrees)."""
        self.set_drone_pose([1.0, 1.0, 1.0], [0.0, 170.0, 0.0])  # Roll exactly at 170 degrees threshold
        # Step the simulation
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
         # Expected reward 
        tilt_reward = max(TILT_MULTIPLIER * (abs(self.gym_env.drone_rpy[0]) + abs(self.gym_env.drone_rpy[1])), -0.2)
        away_reward = max((APPROACH_MULTIPLIER/2) * (self.gym_env.last_distance_to_goal - self.gym_env.distance_to_goal), -3.0)
        expected_reward = tilt_reward + away_reward + + 1   # tilt penalty + away penalty + alive reward
        # Check that the drone has not flipped (should not be terminated)
        self.assertFalse(has_terminated or has_truncated)  # Should not trigger termination
        self.assertEqual(self.gym_env.reward, expected_reward)  
    
    def test_drone_out_of_bounds(self):
        """Test if the drone terminates when it goes out of bounds."""
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        self.set_drone_pose([2.0, 2.0, 4.0], [0.0, 0.0, 0.0])  # Out of bounds 
        # Step the simulation
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        # Expected reward 
        away_reward = max((APPROACH_MULTIPLIER/2) * (self.gym_env.last_distance_to_goal - self.gym_env.distance_to_goal), -3.0)
        expected_reward = OUT_OF_BOUND_PENALTY + away_reward + 0.2 + 1# Ouf of bounds penalty + away penalty + alive reward
        # Check if the drone is out of bounds (should trigger termination)
        self.assertTrue(has_terminated)  # Should be terminated due to out of bounds
        self.assertFalse(has_truncated)  
        self.assertEqual(self.gym_env.reward, expected_reward)  
        
    def test_drone_out_of_bounds_flip(self):
        """Test if the drone terminates when it goes out of bounds and flipped at the same time."""
        self.gym_env.goal_pose = np.array([1.0, 1.0, 1.0])
        self.set_drone_pose([2.0, 2.0, 4.0], [0.0, 170.01, 0.0])  # Out of bounds 
        # Step the simulation
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        # Expected reward 
        tilt_reward = max(TILT_MULTIPLIER * (abs(self.gym_env.drone_rpy[0]) + abs(self.gym_env.drone_rpy[1])), -0.2)
        away_reward = max((APPROACH_MULTIPLIER/2) * (self.gym_env.last_distance_to_goal - self.gym_env.distance_to_goal), -3.0)
        expected_reward = OUT_OF_BOUND_PENALTY + FLIP_PENALTY + tilt_reward + away_reward + 1# Ouf of bounds penalty + Flipped penalty + tilt penalty + away penalty + alive reward
        # Check if termination is triggered
        self.assertTrue(has_terminated)  # Should trigger termination
        self.assertFalse(has_truncated)  
        self.assertEqual(self.gym_env.reward, expected_reward)  
    
    def test_drone_fallen(self):
        """Test if the drone terminates when it fallen to the ground."""
        self.set_drone_pose([1.0, 1.0, 0.09], [0.0, 0.0, 0.0])  # Fallen to the ground (< 0.1m)
        mujoco.mj_step(self.gym_env.model, self.gym_env.drone)
        # Update goal attributes and calculate rewards
        self.gym_env.goal_attributes = self.gym_env._calculate_goal_attributes()
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        # Expected reward 
        away_reward = max((APPROACH_MULTIPLIER/2) * (self.gym_env.last_distance_to_goal - self.gym_env.distance_to_goal), -3.0)
        expected_reward = GROUND_PENALTY + away_reward + 0.2 # Ground Penalty + away penalty + alive reward
        # Check if termination is triggered
        self.assertTrue(has_terminated)  # Should be terminated due to fallen to the ground
        self.assertFalse(has_truncated)   
        self.assertEqual(self.gym_env.reward, expected_reward)  
    
    def test_timeout(self):
        # Alternate between maximum thrust (ACTIONS[15]) and no thrust (ACTIONS[0]) for 300 steps
        for i in range(300):
            if i % 2 == 0:
                self.gym_env.step(15)  # Apply maximum thrust (ACTIONS[15])
            else:
                self.gym_env.step(0)  # Apply no thrust (ACTIONS[0])
        # Calculate rewards and check if the simulation has been truncated or terminated
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        # Check that the drone has not terminated or exceeded the time limit
        self.assertFalse(has_terminated)
        self.assertTrue(has_truncated)  # Should be truncated due to max timesteps
           
    def test_max_thrust_all_motors(self):
        # Apply maximum thrust to all motors (ACTIONS[15] represents full thrust on all 4 motors)
        for _ in range(50):
            self.gym_env.step(15)  # Apply maximum thrust continuously
        has_terminated, has_truncated, self.gym_env.reward = self.gym_env._calculate_reward()
        self.assertFalse(has_terminated or has_truncated)
        self.assertGreater(self.gym_env.drone_position[2], 0.5) # Ensure the drone has moved upwards (in the z-direction)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

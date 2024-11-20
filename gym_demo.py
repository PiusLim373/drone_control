#!/usr/bin/python3
from drone_control_gym import *

# Create an instance of the DroneControlGym environment
gym_env = DroneControlGym()
print(f"step: {gym_env.step_count}")  # the gym environment starts with step 1

# Run the simulation for 25 steps to allow the quadcopter to reach a stationary state
while gym_env.step_count < 25:
    print(f"step: {gym_env.step_count}, state: {gym_env.step(0)}")
# at the end of step 25, the quadcopter is at a stationary state and near position 0, 0, 0
print(f"Drone pose at step {gym_env.step_count}: {gym_env.get_pose()}")
# render the simulation, press Esc to continue
gym_env.render()

# Now run the simulation for 50 more steps with action that turns on all 4 rotors
while gym_env.step_count < 75:
    print(f"step: {gym_env.step_count}, state: {gym_env.step(15)}")
# at the end of step 75, the quadcopter has moved to a new position
print(f"Drone pose at step {gym_env.step_count}: {gym_env.get_pose()}")
# render the simulation, press Esc to continue
gym_env.render()

# Now run the simulation for 50 more steps with action that turns on the diagonal rotors (to rotate clockwise)
while gym_env.step_count < 125:
    print(f"step: {gym_env.step_count}, state: {gym_env.step(5)}")
# at the end of step 125, the quadcopter has rotated  pitch angle is changed
print(f"Drone pose at step {gym_env.step_count}: {gym_env.get_pose()}")
# render the simulation, press Esc to continue
gym_env.render()

# Now run the simulation for 15 more steps with action that turns on the back rotors (to rotate forward)
while gym_env.step_count < 140:
    print(f"step: {gym_env.step_count}, state: {gym_env.step(3)}")
# at the end of step 140, the quadcopter has rotated forward, pitch angle is changed
print(f"Drone pose at step {gym_env.step_count}: {gym_env.get_pose()}")
# render the simulation, press Esc to continue
gym_env.render()

# Now reset the environment after step 140
print("Resetting the environment after step 140")
gym_env.reset()  # Reset the environment to its initial state
print(f"After reset, step: {gym_env.step_count}")
print(f"Drone pose after reset: {gym_env.get_pose()}")

# Run the simulation for 25 steps to allow the quadcopter to reach a stationary state
while gym_env.step_count < 25:
    print(f"step: {gym_env.step_count}, state: {gym_env.step(0)}")
# at the end of step 25, the quadcopter is at a stationary state and near position 0, 0, 0
print(f"Drone pose at step {gym_env.step_count}: {gym_env.get_pose()}")
# render the simulation, press Esc to continue
gym_env.render()

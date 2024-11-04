#!/usr/bin/python3
import os
from new_gym import *
from torch.utils.tensorboard import SummaryWriter
from training_agent import *
tb_writer = SummaryWriter(log_dir="saves/")

# Function to print a few weights from the network
def print_weights(network, title):
    print(f"\n{title}:")
    for name, param in network.named_parameters():
        if param.requires_grad:
            # Print only the first few elements of the weights
            print(f"Layer: {name} | Weights: {param.data.flatten()[:5]}")  # Show first 5 weights

RENDER = False
INPUT_DIMS = 21
ACTIONS_DIMS = 16
LEARNING_RATE = 0.0003
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
BATCH_SIZE = 64
TRAIN_EVERY_N_STEPS = 1024
N_EPOCH = 10
N_GAMES = 10000
PERFORMANCE_THRESHOLD = 1000 #Set this based on desired success rate average score, i.e. 80% reach goal
CHECKPOINT_DIR = "saves/"

gym_env = DroneControlGym(render=RENDER)
agent = Agent(input_dims=INPUT_DIMS, action_dims=ACTIONS_DIMS, learning_rate=LEARNING_RATE, discount=DISCOUNT, gae_lambda=GAE_LAMBDA, ppo_clip=PPO_CLIP, batch_size=BATCH_SIZE, n_epoch=N_EPOCH, checkpoint_dir=CHECKPOINT_DIR)
learning_counter = 0
step_count = 0
all_score = []
best_score = -int(1e6)

# Print the weights of the actor network and critic network after backpropagation
print_weights(agent.actor, "Actor Network Weights at learning counter " + str(learning_counter))
print_weights(agent.critic, "Critic Network Weights at learning counter " + str(learning_counter))

# Run for N_GAMES episodes, train the network every 1024 steps
for i in range(N_GAMES):
    observation, _ = gym_env.reset()
    done = False
    score = 0

    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_new, reward, done, _, _ = gym_env.step(action)
        step_count += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        # update the networks every 1024 collected data, aka train
        if step_count % TRAIN_EVERY_N_STEPS == 0:
            critic_loss, actor_loss, total_loss = agent.learn()
            tb_writer.add_scalar("Actor Loss", actor_loss, i)
            tb_writer.add_scalar("Critic Loss", critic_loss, i)
            tb_writer.add_scalar("Total Loss", total_loss, i)
            step_count = 0
            learning_counter += 1

        observation = observation_new

    all_score.append(score)
    average_score = np.mean(all_score[-100:])
    average_score_1k = np.mean(all_score[-1000:])
    
    if i % 1000 == 0 and i > 0:  # Check every 1000 episodes
        print(f"Eposide {i} | Average Last 1k Rollout Score: {average_score_1k} | Current Level: {gym_env.current_level} ")
        if average_score >= PERFORMANCE_THRESHOLD:  # Increase level if performance is good
            gym_env.increase_difficulty()
            logging.info(f"Difficulty adjusted after episode {i}: New level is {gym_env.current_level}")
            
        elif average_score < PERFORMANCE_THRESHOLD / 2:  # Decrease level if performance is bad
            gym_env.decrease_difficulty()
            logging.info(f"Difficulty decreased after episode {i}: New level is {gym_env.current_level}")
            
        # Save the model every 1000 episodes
        agent.save_models(autosave=True)
        print(f"Model saved after episode {i}.")
    
    # log to tensorboard
    tb_writer.add_scalar("Score", score, i)
    tb_writer.add_scalar("Average Last 100 Rollout Score", average_score, i)
    
    print(f"Eposide {i} | Score: {score:.4f} | Average Last 100 Rollout Score: {average_score:.4f} | Current Level: {gym_env.current_level} | Learning Counter: {learning_counter} | Data collected: {step_count}")

    if average_score > best_score:
        best_score = average_score
        agent.save_models(autosave=True)
        print(f"New best score, autosaving model")
        
# End of the training, saving the model
agent.save_models()

# Print the weights of the actor network and critic network after training
print_weights(agent.actor, "Actor Network Weights at learning counter " + str(learning_counter))
print_weights(agent.critic, "Critic Network Weights at learning counter " + str(learning_counter))

            

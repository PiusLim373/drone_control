#!/usr/bin/python3
from torch.utils.tensorboard import SummaryWriter
from drone_control_gym import *
from training_agent import *
tb_writer = SummaryWriter(log_dir="saves/")

# Function to print a few weights from the network
def print_weights(network, title):
    print(f"\n{title}:")
    for name, param in network.named_parameters():
        if param.requires_grad:
            # Print only the first few elements of the weights
            print(f"Layer: {name} | Weights: {param.data.flatten()[:5]}")  # Show first 5 weights

RENDER = True
INPUT_DIMS = 11
ACTIONS_DIMS = 16
LEARNING_RATE = 0.0003
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
BATCH_SIZE = 64
TRAIN_EVERY_N_STEPS = 1024
N_EPOCH = 10
N_GAMES = 20 
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
    observation = gym_env.reset()
    done = False
    score = 0

    while not done:
        action, prob, val = agent.choose_action(observation)
        reward, done, observation_new = gym_env.step(ACTIONS[action])
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
    
    # log to tensorboard
    tb_writer.add_scalar("Score", score, i)
    tb_writer.add_scalar("Average Last 100 Rollout Score", average_score, i)
    
    print(f"Eposide {i} | Score: {score} | Average Last 100 Rollout Score: {average_score} | Learning Counter: {learning_counter} | Data collected: {step_count}")

    if average_score > best_score:
        best_score = average_score
        agent.save_models(autosave=True)
        print(f"New best score, autosaving model")
        
# End of the training, saving the model
agent.save_models()

# Print the weights of the actor network and critic network after training
print_weights(agent.actor, "Actor Network Weights at learning counter " + str(learning_counter))
print_weights(agent.critic, "Critic Network Weights at learning counter " + str(learning_counter))

            

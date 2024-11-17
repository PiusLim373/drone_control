#!/usr/bin/python3
from drone_control_gym import *
from neural_network import *


# Function to print a few weights from the network
def print_weights(network, title):
    print(f"\n{title}:")
    for name, param in network.named_parameters():
        if param.requires_grad:
            # Print only the first few elements of the weights
            print(f"Layer: {name} | Weights: {param.data.flatten()[:5]}")  # Show first 5 weights


LEARNING_RATE = 0.0003
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
BATCH_SIZE = 5
N_EPOCH = 5

gym_env = DroneControlGym()
memory = Memory(batch_size=BATCH_SIZE)
actor = ActorNetwork(
    n_actions=16, input_dims=25, learning_rate=LEARNING_RATE
)  # policy nn with 16 actions output (4 motors boolean control combinations)
critic = CriticNetwork(input_dims=25, learning_rate=LEARNING_RATE)  # value nn with 1 value output

# get initial states
observation, info = gym_env.reset()
print(f"Initial state: {observation}")  # initial state is a 25 element list: [dx, dy, dz, d, r, p, y, m1, m2, m3, m4]

learning_counter = 0

# run for 5 episodes:
for i in range(5):
    done = False
    score = 0
    step_count = 0

    while not done:
        # convert observation to tensor
        state = torch.tensor([observation], dtype=torch.float).to(actor.device)

        # get the prediction from actor network
        dist = actor(state)
        action = dist.sample()
        prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        # get the prediction from critic network
        val = torch.squeeze(critic(state)).item()

        observation_new, reward, terminated, truncated, info = gym_env.step(action)
        done = terminated or truncated
        step_count += 1
        score += reward
        memory.store_memory(state=observation, action=action, probs=prob, vals=val, reward=reward, done=done)

        # update the networks every 20 collected data, aka train
        if step_count % 20 == 0:

            # update the networks 5 times for each batch of data
            for _ in range(N_EPOCH):
                # generate minibatch of data from memory
                state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = (
                    memory.generate_batches()
                )

                # predicted values from the critic network
                values = vals_arr
                values = torch.tensor(values).to(actor.device)

                # calculate advantage using GAE
                advantage = np.zeros(len(reward_arr), dtype=np.float32)
                # Iterate over each time step except the last one
                for t in range(len(reward_arr) - 1):
                    discount = 1
                    advantage_t = 0
                    # For each future time step k, calculate the advantage
                    for k in range(t, len(reward_arr) - 1):
                        # Calculate the temporal difference (TD error) for time step k
                        td_error = reward_arr[k] + DISCOUNT * values[k + 1] * (1 - int(dones_arr[k])) - values[k]
                        # Accumulate the discounted TD error to compute the advantage
                        advantage_t += discount * td_error
                        # Update the discount factor for the next time step
                        discount *= DISCOUNT * GAE_LAMBDA
                    # Store the computed advantage for time step t
                    advantage[t] = advantage_t
                advantage = torch.tensor(advantage).to(actor.device)

                for batch in batches:
                    # let the actor network predict the again, and calculate the prob_ratio (r_theta) between the new and old prediction
                    states = torch.tensor(state_arr[batch], dtype=torch.float).to(actor.device)
                    old_probs = torch.tensor(old_prob_arr[batch]).to(actor.device)
                    actions = torch.tensor(action_arr[batch]).to(actor.device)
                    dist = actor(states)
                    new_action = dist.sample()
                    new_probs = torch.squeeze(dist.log_prob(new_action))

                    prob_ratio = new_probs.exp() / old_probs.exp()

                    # weighted prob (r_theta * advantage)
                    weighted_probs = prob_ratio * advantage[batch]

                    # apply ppo clip to the weighted prob
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantage[batch]

                    # calculate actor loss, actor loss = -min(r_theta * advantage, clipped_r_theta * advantage)
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    # let the crictic network predict the value, and calculate the critic loss
                    new_value = critic(states)
                    new_value = torch.squeeze(new_value)

                    # calculate the critic loss using mean square error
                    returns = advantage[batch] + values[batch]
                    critic_loss = (returns - new_value) ** 2
                    critic_loss = critic_loss.mean()

                    # calculate total loss
                    total_loss = actor_loss + 0.5 * critic_loss

                    # back progation and network update
                    actor.optimizer.zero_grad()
                    critic.optimizer.zero_grad()
                    total_loss.backward()
                    actor.optimizer.step()
                    critic.optimizer.step()
            memory.clear_memory()
        learning_counter += 1

        # Print the weights of the actor network and critic network after backpropagation
        print_weights(actor, "Actor Network Weights at learning counter " + str(learning_counter))
        print_weights(critic, "Critic Network Weights at learning counter " + str(learning_counter))

        observation = observation_new

        # terminate the episode if the step count exceeds 100
        if step_count > 100:
            print("Terminated")
            done = True

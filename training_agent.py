#!/usr/bin/python3
from neural_network import *


class Agent:
    def __init__(self, input_dims, action_dims, learning_rate=0.0003, discount=0.99, gae_lambda=0.95, critic_loss_coeff=0.5, entropy_coeff=0.01, ppo_clip=0.2, batch_size=64, n_epoch=10, checkpoint_dir="saves"):
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.learning_rate = learning_rate
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.checkpoint_dir = checkpoint_dir
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_coeff = entropy_coeff
        
        self.actor = ActorNetwork(n_actions=self.action_dims, input_dims=self.input_dims, learning_rate=self.learning_rate, chkpt_dir=self.checkpoint_dir)
        self.critic = CriticNetwork(input_dims=self.input_dims, learning_rate=self.learning_rate, chkpt_dir=self.checkpoint_dir)
        self.memory = Memory(batch_size=self.batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, autosave=False):
        print("... saving models ...")
        self.actor.save_checkpoint(autosave=autosave)
        self.critic.save_checkpoint(autosave=autosave)
    
    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()   
    
    def choose_action(self, observation):
        # convert observation to tensor
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        # get the prediction from actor network
        dist = self.actor(state)
        action = dist.sample()
        prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        # get the prediction from critic network
        val = torch.squeeze(self.critic(state)).item()
        
        return action, prob, val
    
    def learn(self):
        # initialize losses value, to return to tensorboard later on
        total_actor_loss = 0
        total_critic_loss = 0
        total_total_loss = 0
        
        # update the networks 15 times for each batch of data
        for _ in range(self.n_epoch):
            # generate minibatch of data from memory
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = (
                self.memory.generate_batches()
            )

            # predicted values from the critic network
            values = vals_arr

            # calculate advantage using GAE
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            # Iterate over each time step except the last one
            for t in range(len(reward_arr) - 1):
                discount = 1
                advantage_t = 0
                # For each future time step k, calculate the advantage
                for k in range(t, len(reward_arr) - 1):
                    # Calculate the temporal difference (TD error) for time step k
                    td_error = reward_arr[k] + self.discount * values[k + 1] * (1 - int(dones_arr[k])) - values[k]
                    # Accumulate the discounted TD error to compute the advantage
                    advantage_t += discount * td_error
                    # Update the discount factor for the next time step
                    discount *= self.discount * self.gae_lambda
                # Store the computed advantage for time step t
                advantage[t] = advantage_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            
            for batch in batches:
                # let the actor network predict the again, and calculate the prob_ratio (r_theta) between the new and old prediction
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                
                # get the new action prediction from the actor network, caclulate the prob_ratio
                dist = self.actor(states) 
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # weighted prob (r_theta * advantage)
                weighted_probs = prob_ratio * advantage[batch]

                # apply ppo clip to the weighted prob
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantage[batch]

                # calculate actor loss, actor loss = -min(r_theta * advantage, clipped_r_theta * advantage)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # let the crictic network predict the value, and calculate the critic loss
                new_value = self.critic(states)
                new_value = torch.squeeze(new_value)

                # calculate the critic loss using mean square error
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - new_value) ** 2
                critic_loss = critic_loss.mean()
                
                entropy = dist.entropy().mean()
                entropy_penalty = -self.entropy_coeff * entropy

                # calculate total loss
                total_loss = actor_loss + self.critic_loss_coeff * critic_loss + entropy_penalty

                # back progation and network update
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
                # Accumulate losses for averaging
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_total_loss += total_loss.item()
                
        self.memory.clear_memory()

        # Return average losses
        average_actor_loss = total_actor_loss / self.n_epoch
        average_critic_loss = total_critic_loss / self.n_epoch
        average_total_loss = total_total_loss / self.n_epoch

        return average_critic_loss, average_actor_loss, average_total_loss

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Categorical
from collections import Counter

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorNetwork, self).__init__()
        self.temperature = 1.5
        # Initialize with smaller values to prevent action distribution collapse
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),  # Remove softmax to prevent vanishing gradients
        )

        # Apply weight initialization
        self.apply(init_weights)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, state):
        conv_out = self.conv(state)
        conv_out = conv_out.view(conv_out.size(0), -1)
        action_logits = self.actor(conv_out)

        # Apply temperature scaling to increase exploration
        action_probs = torch.softmax(action_logits / self.temperature, dim=-1)
        dist = Categorical(action_probs)
        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CriticNetwork, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Apply weight initialization
        self.apply(init_weights)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, state):
        conv_out = self.conv(state)
        conv_out = conv_out.view(conv_out.size(0), -1)
        value = self.critic(conv_out)
        return value


class PPOAgent:
    def __init__(
        self,
        input_dims,
        n_actions,
        batch_size=32,
        alpha=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        n_epochs=10,
        base_entropy_coef=0.1,
        min_entropy_coef=0.001,
        entropy_decay=0.999
    ):

        # Added entropy 
        self.alpha = alpha
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.base_entropy_coef = base_entropy_coef
        self.min_entropy_coef = min_entropy_coef
        self.entropy_decay = entropy_decay
        self.batch_size = batch_size
        self.actions = []
        self.rewards = []
        # Dict to store level-specific information
        self.level_stats = {}

        self.current_level_key = None

        self.actor = ActorNetwork(input_dims, n_actions)
        self.critic = CriticNetwork(input_dims)
        self.memory = PPOMemory(batch_size)

        # Increased learning rate slightly
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)

        # Learning rate schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)

        # Add scheduler for entropy coefficient decay
        self.entropy_decay = 0.99995
        self.min_entropy = 0.001

        # Normalizing rewards
        self.reward_running_mean = 0
        self.reward_running_std = 1
        self.reward_alpha = 0.99  # Moving average weight

    def initialize_level(self, world, stage):
        level_key = f"{world}-{stage}"
        self.current_level_key = level_key
        if level_key not in self.level_stats:
            self.level_stats[level_key] = {
                'entropy_coef': self.base_entropy_coef,
                'avg_score': 0.0,
                'episode' : 0,
                'high_score': float('-inf'),
                'running_score': 0.0,
                'score_alpha': 0.95
            }
        return level_key

    def update_level_stats(self, level_key, episode_score):
        """Update level statistics and adjust entropy coefficient"""
        # Initialize the stats if they don't exist
        level_world, level_stage = level_key.split('-')
        if level_key not in self.level_stats:
            self.initialize_level(level_world, level_stage)
        # if level_key not in self.level_stats:
        #     self.level_stats[level_key] = {
        #         'entropy_coef': self.base_entropy_coef,
        #         'avg_score': 0.0,
        #         'episodes': 0,  # Changed from 'episode' to 'episodes'
        #         'high_score': float('-inf'),
        #         'running_score': 0.0,
        #         'score_alpha': 0.95
        #     }
                
        stats = self.level_stats[level_key]
        stats['episode'] += 1
        stats['high_score'] = max(stats['high_score'], episode_score)
        
        # Update running average score
        if stats['episode'] == 1:
            stats['running_score'] = episode_score
        else:
            stats['running_score'] = (stats['score_alpha'] * stats['running_score'] + 
                                    (1 - stats['score_alpha']) * episode_score)
        
        # Calculate relative performance (how close we are to the high score)
        relative_performance = stats['running_score'] / max(stats['high_score'], 1)
        
        # Adjust entropy coefficient based on relative performance
        # Higher entropy for lower performing levels
        target_entropy = self.base_entropy_coef * (1.0 - relative_performance)
        target_entropy = max(target_entropy, self.min_entropy_coef)
        
        # Smooth entropy adjustment
        stats['entropy_coef'] = (0.98 * stats['entropy_coef'] + 
                               0.02 * target_entropy)

        stats['entropy_coef'] = max(stats['entropy_coef'], self.min_entropy_coef)
        
        return stats['entropy_coef']
    
    def get_entropy_coef(self, level_key):
        """Get the current entropy coefficient for a specific level"""
        return self.level_stats[level_key]['entropy_coef']

    def decay_entropy(self, level_key):
        # self.entropy_coef = max(self.entropy_coef * self.entropy_decay, self.min_entropy)
        self.level_stats[level_key]['entropy_coef'] = max(self.level_stats[level_key]['entropy_coef'] * self.entropy_decay, self.min_entropy)

    def normalize_rewards(self, rewards):
        self.reward_running_mean = self.reward_alpha * self.reward_running_mean + (1 - self.reward_alpha) * np.mean(rewards)
        self.reward_running_std = self.reward_alpha * self.reward_running_std + (1 - self.reward_alpha) * np.std(rewards)
        return (rewards - self.reward_running_mean) / (self.reward_running_std + 1e-8)

    def store_transition(self, state, action, probs, vals, reward, done):
        # max_element, max_count = max(Counter(self.actions[-10:]).items(), key=lambda x: x[1])
        # if len(self.actions) &gt; 1 and action == max_element:
        #     reward -= 0.1
        # self.rewards.append(reward)
        # self.memory.store_memory(state, action, probs, vals, reward, done)

        if len(self.actions) > 1:  # Ensure there are enough past actions to compare
            counter = Counter(self.actions[-20:])
            max_element, max_count = max(counter.items(), key=lambda x: x[1], default=(None, 0))
    
            if action == max_element:
                reward -= 0.1  # Penalize for repeating the most frequent recent action

        self.rewards.append(reward)
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        if isinstance(observation, list):
            observation = np.array(observation)

        if len(observation.shape) == 3:
            observation = np.expand_dims(observation, 0)

        state = torch.from_numpy(observation).float().to(self.actor.device)

        dist = self.actor(state)        # probability distribution from actor
        value = self.critic(state)      # optimal value from critic

        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        current_entopy_coef = self.get_entropy_coef(self.level_key)
        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_probs_arr,
                vals_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.actor.device)
            # Normalize advantages
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # print(f"Advantage Mean: {advantage.mean().item()}, Std: {advantage.std().item()}")

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(
                    self.actor.device
                )
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                print(dist.shape)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * advantage[batch]
                )
                # print(weighted_probs.shape)
                # print(weighted_clipped_probs.shape)
                # Add entropy bonus
                entropy = dist.entropy().mean()
                # print(f"Entropy: {entropy}")
                actor_loss = (
                    -torch.min(weighted_probs, weighted_clipped_probs).mean()
                    - current_entopy_coef * entropy
                )

                rewards = self.memory.rewards
                rewards = self.normalize_rewards(rewards)

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.actor_scheduler.step()
                self.critic_scheduler.step()

            # Decay entropy coefficient
            # self.entropy_coef = max(
            #     self.entropy_coef * self.entropy_decay, self.min_entropy
            # )
            self.decay_entropy(self.level_key)

        self.memory.clear_memory()

    def save_models(self, save_path):
        """Save the model weights and hyperparameters."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "hyperparameters": {
                "batch_size": self.batch_size,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "policy_clip": self.policy_clip,
                "n_epochs": self.n_epochs,
                "base_entropy_coef": self.base_entropy_coef,
                "entropy_decay": self.entropy_decay,
                "min_entropy": self.min_entropy,
                "level_stats": self.level_stats
            },
        }
        torch.save(checkpoint, save_path+".pth")
        print(f"Model and hyperparameters saved to {save_path}")

    def load_models(self, load_path):
        """Load the model weights and hyperparameters."""
        checkpoint = torch.load(load_path+".pth")
        
        # Load model weights
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        # Restore hyperparameters
        hyperparams = checkpoint["hyperparameters"]
        self.batch_size = hyperparams["batch_size"]
        self.alpha = hyperparams["alpha"]
        self.gamma = hyperparams["gamma"]
        self.gae_lambda = hyperparams["gae_lambda"]
        self.policy_clip = hyperparams["policy_clip"]
        self.n_epochs = hyperparams["n_epochs"]
        self.base_entropy_coef = hyperparams["base_entropy_coef"]
        self.entropy_decay = hyperparams["entropy_decay"]
        self.min_entropy = hyperparams["min_entropy"]
        self.level_stats = hyperparams["level_stats"]

        print(f"Model and hyperparameters loaded from {load_path}")

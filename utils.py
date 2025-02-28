import time
import datetime
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from collections import defaultdict

def get_current_date_time_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


class Timer():
    def __init__(self):
        self.times = []

    def start(self):
        self.t = time.time()

    def print(self, msg=''):
        print(f"Time taken: {msg}", time.time() - self.t)

    def get(self):
        return time.time() - self.t
    
    def store(self):
        self.times.append(time.time() - self.t)

    def average(self):
        return sum(self.times) / len(self.times)

# def argParser():
#     parser = argparse.ArgumentParser(description="Train PPO on Super Mario Bros")
#     parser.add_argument("--env_name", type=str, default=None, help="Environment name")
#     parser.add_argument(
#         "--train", type=bool, default=True, help="Whether to train the agent"
#     )
#     parser.add_argument(
#         "--display", type=bool, default=False, help="Display environment while training"
#     )
#     parser.add_argument(
#         "--ckpt_ep", type=int, default=100, help="Checkpoint save interval in episodes"
#     )
#     parser.add_argument(
#         "--episodes", type=int, default=10000, help="Number of episodes to train"
#     )
#     parser.add_argument(
#         "--max_steps_per_episode", type=int, default=5000, help="Maximum steps per episode"
#     )
#     parser.add_argument(
#         "--batch_size", type=int, default=64, help="Batch size for training"
#     )
#     parser.add_argument("--alpha", type=float, default=0.0005, help="Learning rate")
#     parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
#     parser.add_argument(
#         "--reset", type=bool, default=False, help="Resets the training history"
#     )
#     parser.add_argument(
#         "--entropy_coef",
#         type=float,
#         default=None,
#         help="Define a particular entropy coefficient.",
#     )
#     parser.add_argument(
#         "--overwrite_hyperparams",
#         type=bool,
#         default=False,
#         help="Overwrite hyperparameters with the ones passed as arguments.",
#     )
#     parser.add_argument(
#         "--model_path", type=str, default=None, help="Define a particular model to use."
#     )
#     # Parse the arguments
#     args = parser.parse_args()
#     return args

def argParser():
    parser = argparse.ArgumentParser(description="Train PPO on Super Mario Bros")
    
    parser.add_argument("--env_name", type=str, default=None, help="Environment name")    
    parser.add_argument(
        "--train", 
        action="store_true",
        default=True,
        help="Whether to train the agent (default: True)"
    )
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Display environment while training (default: False)"
    )
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Resets the training history (default: False)"
    )
    parser.add_argument(
        "--overwrite_hyperparams", 
        action="store_true", 
        help="Overwrite hyperparameters with the ones passed as arguments (default: False)"
    )
    
    parser.add_argument("--ckpt_ep", type=int, default=50, help="Checkpoint save interval in episodes")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train")
    parser.add_argument("--max_steps_per_episode", type=int, default=5000, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--alpha", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--entropy_coef", 
        type=float, 
        default=None, 
        help="Define a particular entropy coefficient."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
        help="Define a particular model to use."
    )
    
    args = parser.parse_args()
    return args


def get_latest_checkpoint(checkpoint_dir):
    # List all files in the checkpoint directory
    checkpoint_files = os.listdir(checkpoint_dir)
    
    # Filter files that match the pattern 'model_<iter_number>_iter.pt'
    pattern = r"model_(\d+).pth"
    iter_numbers = []

    for file in checkpoint_files:
        match = re.match(pattern, file)
        if match:
            # Extract the iteration number and store it
            iter_numbers.append(int(match.group(1)))

    if iter_numbers:
        # Get the file corresponding to the highest iteration number
        latest_iter = max(iter_numbers)
        # latest_checkpoint = f"model_{latest_iter}_actor.pt"
        # return os.path.join(checkpoint_dir, latest_checkpoint)
        return f"models/model_{latest_iter}", latest_iter
    else:
        return None, 0  # No checkpoint found

class PerformanceTracker:
    def __init__(self, log_file="performance_log.csv", plot_file="performance_plot.png"):
        self.performance_data = {"episode": [], "score": [], "steps": []}
        self.log_file = log_file
        self.plot_file = plot_file

    def log_performance(self, episode, score, steps):
        """Logs the performance metrics for an episode."""
        self.performance_data["episode"].append(episode)
        self.performance_data["score"].append(score)
        self.performance_data["steps"].append(steps)

    def save_logs(self):
        """Saves performance data to a CSV file."""
        pd.DataFrame(self.performance_data).to_csv(self.log_file, mode='a', header=False, index=False)

    def plot_performance(self):
        """Plots and saves the performance metrics."""
        plt.figure(figsize=(12, 6))

        # Plot scores
        plt.subplot(1, 2, 1)
        plt.plot(self.performance_data["episode"], self.performance_data["score"], label="Score")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Score vs. Episode")
        plt.legend()

        # Plot steps
        plt.subplot(1, 2, 2)
        plt.plot(self.performance_data["episode"], self.performance_data["steps"], label="Steps", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Steps vs. Episode")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.plot_file)
        # plt.show()

class StageSelector:
    def __init__(self, worlds, stages, total_episodes):
        self.worlds = worlds
        self.stages = stages
        self.total_episodes = total_episodes
        
        # Calculate target episodes per level
        total_levels = len(worlds) * len(stages)
        self.target_episodes_per_level = total_episodes // total_levels
        
        # Track episodes per level
        self.level_episodes = defaultdict(int)
        
        # Track remaining episodes for each level
        self.remaining_episodes = {
            (w, s): self.target_episodes_per_level 
            for w in worlds 
            for s in stages
        }
    
    def select_stage(self):
        """Select the next stage based on training progress"""
        min_episodes = float('inf')
        selected_world = self.worlds[0]
        selected_stage = self.stages[0]

        for world in self.worlds:
            for stage in self.stages:
                episodes = self.level_episodes[(world, stage)]
                if episodes < min_episodes:
                    min_episodes = episodes
                    selected_world = world
                    selected_stage = stage

        self.level_episodes[(selected_world, selected_stage)] += 1
        return selected_world, selected_stage
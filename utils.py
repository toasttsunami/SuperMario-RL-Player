import time
import datetime
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

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


def get_latest_checkpoint(checkpoint_dir):
    # List all files in the checkpoint directory
    checkpoint_files = os.listdir(checkpoint_dir)
    
    # Filter files that match the pattern 'model_<iter_number>_iter.pt'
    pattern = r"model_(\d+)_actor\.pt"
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

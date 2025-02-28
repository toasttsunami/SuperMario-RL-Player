import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
import shutil
import argparse
from ppo import *
from utils import *

# Argument parser for configuration
parser = argparse.ArgumentParser(description="Train PPO on Super Mario Bros")
parser.add_argument("--env_name", type=str, default="SuperMarioBros-1-1-v3", help="Environment name")
parser.add_argument("--should_train", type=bool, default=True, help="Whether to train the agent")
parser.add_argument("--display", type=bool, default=False, help="Display environment while training")
parser.add_argument("--ckpt_save_interval", type=int, default=50, help="Checkpoint save interval in episodes")
parser.add_argument("--num_of_episodes", type=int, default=1000, help="Number of episodes to train")
parser.add_argument("--max_steps_per_episode", type=int, default=5000, help="Maximum steps per episode")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--alpha", type=float, default=0.00045, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--reset", type=bool, default=False, help="Resets the training history")

args = parser.parse_args()

# Configuration
ENV_NAME = args.env_name
SHOULD_TRAIN = args.should_train
DISPLAY = args.display
CKPT_SAVE_INTERVAL = args.ckpt_save_interval
NUM_OF_EPISODES = args.num_of_episodes
MAX_STEPS_PER_EPISODE = args.max_steps_per_episode
BATCH_SIZE = args.batch_size
ALPHA = args.alpha
GAMMA = args.gamma
RESET = args.reset
model_path = "models"

# Reset training history
if RESET:
    if(os.path.exists(os.path.join(os.getcwd(), model_path))):
        shutil.rmtree(os.path.join(os.getcwd(), model_path))
        print("Cleared training history.")

# Setup paths
os.makedirs(model_path, exist_ok=True)

# Setup environment
render_mode = 'human' if DISPLAY else None
env = gym_super_mario_bros.make(ENV_NAME, render_mode=render_mode, apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Initialize agent
agent = PPOAgent(
    input_dims=env.observation_space.shape,
    n_actions=env.action_space.n,
    batch_size=BATCH_SIZE,
    alpha=ALPHA,
    gamma=GAMMA
)

checkpoint_path, checkpoint_iter = get_latest_checkpoint(model_path)
if checkpoint_path is not None:
    agent.load_models(checkpoint_path)
else:
    print("No saved model found. Starting fresh.")

tracker = PerformanceTracker(log_file="performance_log.csv", plot_file="performance_plot.png")

try:
    for episode in range(checkpoint_iter+1, checkpoint_iter+NUM_OF_EPISODES+1, 1):
        observation, _ = env.reset()
        done = False
        score = 0
        step_count = 0
        
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action, prob, val = agent.choose_action(observation)
            # print(f"{episode} : {action} {prob} {val}")
            next_observation, reward, done, truncated, info = env.step(action)
            step_count += 1
            score += reward
            
            if SHOULD_TRAIN:
                agent.store_transition(observation, action, prob, val, reward, done)
            
            observation = next_observation
            
            # Learn if we have enough steps
            if len(agent.memory.states) >= agent.memory.batch_size and SHOULD_TRAIN:
                # print(f"fuck {episode}")
                agent.learn()

            # if step_count > 70:
            #     break
        
        print(f'Episode {episode}, Score: {score}, Steps: {step_count}')

        # Track Performance
        tracker.log_performance(episode, score, step_count)
        
        if SHOULD_TRAIN and (episode + 1) % CKPT_SAVE_INTERVAL == 0:
            save_path = os.path.join(model_path, f"model_{episode + 1}")
            agent.save_models(save_path)
            print(f"Saved checkpoint at episode {episode + 1}")
            tracker.save_logs()

except KeyboardInterrupt:
    print("Training interrupted by user ")
    # tracker.plot_performance()
except Exception as e:
    print(f"Error during training: {e} ")
finally:
    env.close()

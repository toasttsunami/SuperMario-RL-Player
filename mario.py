import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
import shutil
import random
import ppo
import utils
import traceback

# Argument parser for configuration
worlds = [1, 2, 3, 4, 5, 6, 7, 8]
stages = [1, 2, 3, 4]

args = utils.argParser()

# Configuration
TRAIN = args.train
DISPLAY = args.display
CKPT_EP = args.ckpt_ep
EPISODES = args.episodes
MAX_STEPS_PER_EPISODE = args.max_steps_per_episode
BATCH_SIZE = args.batch_size
ALPHA = args.alpha
GAMMA = args.gamma
# ENTROPY_COEF = args.entropy_coef if args.entropy_coef is not None else 0.1
ENTROPY_COEF = 0.1
RESET = args.reset
OVERWRITE_HYPERPARAMS = args.overwrite_hyperparams
model_path = "models" if args.model_path is None else args.model_path

# Reset training history
if RESET:
    # Remove model directory if it exists
    model_dir = os.path.join(os.getcwd(), model_path)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print("Cleared training history.")

    # Remove CSV file if it exists
    csv_file = "./performance_log.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print("Removed performance log file.")

# Setup paths
os.makedirs(model_path, exist_ok=True)

# Setup environment
render_mode = "human" if DISPLAY else None
stage_selector = utils.StageSelector(worlds, stages, EPISODES)
world, stage = stage_selector.select_stage()
checkpoint_iter = 0
env = gym_super_mario_bros.make(
    f"SuperMarioBros-{world}-{stage}-v3",
    render_mode=render_mode,
    apply_api_compatibility=True,
)
# print(SIMPLE_MOVEMENT)

env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = apply_wrappers(env)

# Initialize agent
agent = ppo.PPOAgent(
    input_dims=env.observation_space.shape,
    n_actions=env.action_space.n,
    batch_size=BATCH_SIZE,
    alpha=ALPHA,
    gamma=GAMMA,
    base_entropy_coef=ENTROPY_COEF if ENTROPY_COEF is not None else 0.1,
)

if args.model_path is None:
    checkpoint_path, checkpoint_iter = utils.get_latest_checkpoint(model_path)
    if checkpoint_path is not None:
        agent.load_models(checkpoint_path)
        if OVERWRITE_HYPERPARAMS:
            print("Overwriting hyperparameters with the ones passed as arguments.")
            agent.alpha = ALPHA
            agent.gamma = GAMMA
            # agent.entropy_coef = ENTROPY_COEF
    else:
        print("No saved model found. Starting fresh.")
else:
    agent.load_models(args.model_path)
    checkpoint_iter = int(args.model_path.split("_")[-1].split(".")[0])

tracker = utils.PerformanceTracker(
    log_file="performance_log.csv", plot_file="performance_plot.png"
)

print(f"Checkpoint iteration: {checkpoint_iter}")

try:
    for episode in range(checkpoint_iter, checkpoint_iter + EPISODES + 1, 1):
        # Change environment every 5 episodes
        if episode % 20 == 0:
            if episode > 0:
                env.close()
            world, stage = stage_selector.select_stage()
            new_env_name = f"SuperMarioBros-{world}-{stage}-v3"
            print(f"Changing environment to {new_env_name}")
            env = gym_super_mario_bros.make(
                new_env_name, render_mode=render_mode, apply_api_compatibility=True
            )
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = apply_wrappers(env)
            # agent.reset()  # Reset agent if necessary

        level_key = agent.initialize_level(world, stage)
        # level_key = f"{world}-{stage}"
        agent.level_key = level_key

        observation, _ = env.reset()
        done = False
        score = 0
        step_count = 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action, prob, val = agent.choose_action(observation)
            next_observation, reward, done, truncated, info = env.step(action)
            step_count += 1
            score += reward
            agent.actions.append(action)

            if TRAIN:
                agent.store_transition(observation, action, prob, val, reward, done)

            observation = next_observation

            # Learn if we have enough steps
            if len(agent.memory.states) >= agent.memory.batch_size and TRAIN:
                agent.learn()

        current_entropy = agent.update_level_stats(level_key, score)

        print(
            f"Episode {episode}, Stage {world}-{stage} Score: {score}, Steps: {step_count} entropy_coef : {current_entropy}"
        )

        # Track Performance
        tracker.log_performance(episode, score, step_count)
        

        if TRAIN and episode > 0 and episode % (CKPT_EP-1) == 0:
            save_path = os.path.join(model_path, f"model_{episode}")
            agent.save_models(save_path)
            print(f"Saved checkpoint at episode {episode}")
            tracker.save_logs()

except KeyboardInterrupt:
    print("\nTraining interrupted by user ")
    if TRAIN is True:
        save_path = os.path.join(model_path, f"model_{episode}")
        agent.save_models(save_path)
        print(f"Saved checkpoint at episode {episode}")
        tracker.save_logs()
except Exception as e:
    print(f"Error during training: {e} ")
    traceback.print_exc()
finally:
    env.close()

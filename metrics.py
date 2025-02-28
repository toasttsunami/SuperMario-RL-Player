import matplotlib.pyplot as plt
import pandas as pd

# Track performance
performance_data = {"episode": [], "score": [], "steps": []}

# Track performance
performance_data["episode"].append(episode)
performance_data["score"].append(score)
performance_data["steps"].append(step_count)

# Save performance data after every checkpoint save
if (episode + 1) % CKPT_SAVE_INTERVAL == 0:
    pd.DataFrame(performance_data).to_csv("performance_log.csv", index=False)

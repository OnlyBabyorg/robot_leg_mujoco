import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from bipedal_env import CustomBipedalEnv
import mujoco.viewer
import time
import sys
import os

def make_env():
    return CustomBipedalEnv("scene.xml")
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    if os.path.exists("./best_model/best_model.zip"):
        model_path = "./best_model/best_model"
        vecnorm_path = "./best_model/best_model_vecnormalize.pkl"
    elif os.path.exists("./bipedal_model_final.zip"):
        model_path = "./bipedal_model_final"
        vecnorm_path = "./vec_normalize.pkl"
    elif os.path.exists("./checkpoints/"):
        
        checkpoints = sorted([f for f in os.listdir("./checkpoints/") if f.endswith('.zip')])
        if checkpoints:
            model_path = f"./checkpoints/{checkpoints[-1].replace('.zip', '')}"
            print(f"Using checkpoint: {checkpoints[-1]}")
        else:
            print("Không tìm thấy model nào!")
            sys.exit(1)
    else:
        print(" Không tìm thấy model nào!")
        sys.exit(1)

print(f"🔬 Testing model: {model_path}")


env = CustomBipedalEnv("scene.xml")
vec_env = DummyVecEnv([make_env])


try:
    if os.path.exists("./vec_normalize.pkl"):
        vec_env = VecNormalize.load("./vec_normalize.pkl", vec_env)
        print(" Loaded VecNormalize stats")
    elif os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        print(" Loaded VecNormalize stats from best_model")
except Exception as e:
    print(f" Không load được VecNormalize: {e}")

vec_env.training = False
vec_env.norm_reward = False

try:
    model = PPO.load(model_path)
    print(f"✅ Loaded model from {model_path}")
except Exception as e:
    print(f" Lỗi load model: {e}")
    sys.exit(1)

obs = vec_env.reset()

env.reset()
obs_raw = env._get_obs()
obs = vec_env.normalize_obs(obs_raw)

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step = 0
    episode = 0
    max_steps = 2000  
    
    while viewer.is_running():
    
        action, _ = model.predict(obs, deterministic=True)
      
        obs_raw, reward, terminated, truncated, info = env.step(action)
      
        obs = vec_env.normalize_obs(obs_raw)
    
        if step % 50 == 0:
            z = env.data.qpos[2]
            knee_r = np.rad2deg(env.data.qpos[9])
            knee_l = np.rad2deg(env.data.qpos[12])
            thigh_r = np.rad2deg(env.data.qpos[8])
            thigh_l = np.rad2deg(env.data.qpos[11])
            print(f"Ep {episode} | Step {step} | H={z:.3f}m | "
                  f"Thigh R={thigh_r:.1f}° L={thigh_l:.1f}° | "
                  f"Knee R={knee_r:.1f}° L={knee_l:.1f}°")
 
        if terminated or step >= max_steps:
            if terminated:
                print(f"NGÃ sau {step} steps!")
            else:
                print(f" Hết {max_steps} steps - Reset!")
            
            obs = vec_env.reset()
            env.reset()
            obs_raw = env._get_obs()
            obs = vec_env.normalize_obs(obs_raw)
            step = 0
            episode += 1
        
        viewer.sync()
        time.sleep(0.02)  
        step += 1
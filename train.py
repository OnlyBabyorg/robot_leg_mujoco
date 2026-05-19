import os
import time
import torch
import warnings
<<<<<<< HEAD
from gymnasium.wrappers import TimeLimit
=======
>>>>>>> 56ebddc3229072649c19b5b552c39b0497e46b5c
warnings.filterwarnings("ignore")  

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor  
from bipedal_env import CustomBipedalEnv

def make_env():
    env = CustomBipedalEnv(xml_path="scene.xml")
    return Monitor(env)

def make_eval_env():
    env = CustomBipedalEnv(xml_path="scene.xml")
<<<<<<< HEAD
    env = TimeLimit(env, max_episode_steps=1000)
=======
>>>>>>> 56ebddc3229072649c19b5b552c39b0497e46b5c
    return Monitor(env)

if __name__ == "__main__":
    n_envs = 12
    
    print(f" Khởi tạo {n_envs} môi trường song song...")
    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method='fork')
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
  
    eval_env = SubprocVecEnv([make_eval_env], start_method='fork')
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./best_model/", exist_ok=True)
    os.makedirs("./checkpoints/", exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./best_model/',
        log_path='./logs/', 
        eval_freq=50000,
        deterministic=True, 
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path='./checkpoints/',
        name_prefix='bipedal_model'
    )
    
    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        device='cpu',
        verbose=1,
        learning_rate=1e-4,          
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,                 
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,               
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print(" Bắt đầu training... Kiểm tra CPU bằng 'htop' hoặc Task Manager.")
    start_time = time.time()
    
    model.learn(
        total_timesteps=3_000_000, 
        callback=[eval_callback, checkpoint_callback]
    )
    
    end_time = time.time()
    print(f"Hoàn tất training trong {(end_time - start_time)/3600:.2f} giờ!")
    
    model.save("bipedal_model_final")
    vec_env.save("vec_normalize.pkl")
<<<<<<< HEAD
    print(" Đã lưu model thành công!")
=======
    print(" Đã lưu model thành công!")
>>>>>>> 56ebddc3229072649c19b5b552c39b0497e46b5c

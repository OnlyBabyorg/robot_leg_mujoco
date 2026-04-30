import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from bipedal_env import CustomBipedalEnv

def make_env():
    return CustomBipedalEnv(xml_path="scene.xml")

if __name__ == "__main__":
   
    n_envs = 12
    
    print(f"Đang khởi tạo {n_envs} môi trường song song...")
    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method='fork')
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
  
    eval_env = SubprocVecEnv([make_env], start_method='fork')
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
    
    policy_kwargs = dict(net_arch=[64, 64])
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        device='cpu',  # CPU vẫn nhanh hơn cho mạng nhỏ
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 // n_envs * 2,  # Điều chỉnh theo số env
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("Bắt đầu training TỐI ƯU... Kiểm tra CPU bằng 'htop' hoặc Task Manager.")
    start_time = time.time()
    
    model.learn(
        total_timesteps=3_000_000, 
        callback=[eval_callback, checkpoint_callback]
    )
    
    end_time = time.time()
    print(f"Hoàn tất training trong {(end_time - start_time)/3600:.2f} giờ!")
    
    model.save("bipedal_model_final")
    vec_env.save("vec_normalize.pkl")
    print("Đã lưu model thành công!")
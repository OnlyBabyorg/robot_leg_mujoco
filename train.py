import os, time, torch, warnings
warnings.filterwarnings("ignore")
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from bipedal_env import CustomBipedalEnv
from asymmetric_extractor import AsymmetricExtractor

# --- Curriculum Callback ---
class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        # Mỗi 200k steps tăng target_vx
        if self.num_timesteps % 200_000 == 0:
            new_target = min(0.3, 0.1 + (self.num_timesteps // 200_000) * 0.05)
            self.training_env.env_method("set_target_vx", new_target)
            print(f"\n[Curriculum] target_vx set to {new_target:.2f}\n")
        return True

# --- Env constructors ---
def make_env():
    env = CustomBipedalEnv(xml_path="scene.xml")
    env = TimeLimit(env, max_episode_steps=1000)
    return Monitor(env)

def make_eval_env():
    env = CustomBipedalEnv(xml_path="scene.xml")
    env = TimeLimit(env, max_episode_steps=1000)
    return Monitor(env)

if __name__ == "__main__":
    n_envs = 12
    print(f"[*] Khởi tạo {n_envs} môi trường song song...")
    vec_env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method='fork')
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    eval_env = SubprocVecEnv([make_eval_env], start_method='fork')
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.)
    
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./best_model/", exist_ok=True)
    os.makedirs("./checkpoints/", exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=max(50000 // n_envs, 1),
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200000 // n_envs, 1),
        save_path='./checkpoints/',
        name_prefix='bipedal_model'
    )
    curriculum_callback = CurriculumCallback()

    policy_kwargs = dict(
        features_extractor_class=AsymmetricExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[],  # để trống
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MultiInputPolicy", 
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
    
    print("[*] Bắt đầu training...")
    start_time = time.time()
    model.learn(
        total_timesteps=5_000_000,
        callback=[eval_callback, checkpoint_callback, curriculum_callback]
    )
    end_time = time.time()
    print(f"Hoàn tất trong {(end_time - start_time)/3600:.2f} giờ!")
    model.save("bipedal_model_final")
    vec_env.save("vec_normalize.pkl")
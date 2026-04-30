import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from bipedal_env import CustomBipedalEnv
import mujoco.viewer
import time

# 1. Tạo Env gốc (KHÔNG Dùng DummyVecEnv vì Viewer cần Env gốc để sync data)
# Chúng ta tạo 2 thứ: Env gốc để viewer lấy data, và VecEnv để chạy predict
env = CustomBipedalEnv("scene.xml")

# 2. Tạo VecEnv và LOAD VecNormalize (BẮT BUỘC)
# Đây là bước bạn đã bỏ sót!
def make_env():
    return CustomBipedalEnv("scene.xml")

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("vec_normalize.pkl", vec_env) # <--- DÒNG SỬA QUAN TRỌNG
# Tắt cập nhật mean/std khi test (evaluation mode)
vec_env.training = False
vec_env.norm_reward = False

# 3. Load Model đã train
model = PPO.load("bipedal_model_final")

# 4. Reset Env gốc và VecEnv đồng bộ
obs = vec_env.reset() # Lấy obs đã được chuẩn hóa
env.reset()          # Reset env gốc cho Viewer

# 5. Mở Viewer (Lấy data từ env gốc)
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        # Predict sử dụng OBS ĐÃ CHUẨN HÓA
        action, _ = model.predict(obs, deterministic=True)
        
        # Thực thi action trên ENV GỐC
        env.step(action)
        
        # QUAN TRỌNG: Phải lấy OBS MỚI ĐÃ CHUẨN HÓA từ vec_env cho bước sau
        # Ta phải gọi vec_env.step() với action y hệt để nó tự cập nhật nội bộ
        # HOẶC dùng cách dễ hơn: Gọi env._get_obs() rồi tự chuẩn hóa bằng vec_env.normalize_obs
        obs_raw = env._get_obs()
        obs = vec_env.normalize_obs(obs_raw)
        
        # Kiểm tra trạng thái done (nếu ngã thì reset)
        if env._check_fallen():
            obs = vec_env.reset()
            env.reset()
        
        viewer.sync()
        time.sleep(0.04) # Giới hạn FPS ~100Hz, không ảnh hưởng vật lý
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class CustomBipedalEnv(gym.Env):
    def __init__(self, xml_path="scene.xml"):
        super(CustomBipedalEnv, self).__init__()    
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Tần số điều khiển
        self.frame_skip = 20  # Giữ nguyên 20
        self.dt = self.model.opt.timestep * self.frame_skip
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()
        self.prev_action = np.zeros(self.model.nu)
        
        num_obs = (self.model.nq - 2) + self.model.nv + self.model.nu + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32
        )

    def step(self, action):
        scaled_action = self.ctrl_ranges[:, 0] + (action + 1.0) * 0.5 * (self.ctrl_ranges[:, 1] - self.ctrl_ranges[:, 0])
        self.data.ctrl[:] = scaled_action
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_fallen()
        truncated = False 
        
        self.prev_action = action.copy()
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        qpos_noise = np.random.uniform(low=-0.05, high=0.05, size=self.model.nq)
        qvel_noise = np.random.uniform(low=-0.05, high=0.05, size=self.model.nv)
        
        qpos_noise[2] = 0 
        qpos_noise[3:7] = 0 
        
        self.data.qpos[:] = self.model.qpos0 + qpos_noise
        self.data.qvel[:] = qvel_noise
        
        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.model.nu)
        
        return self._get_obs(), {}

    def _get_obs(self):
        qpos_filtered = self.data.qpos[2:] 
        qvel = self.data.qvel
    
    # --- SỬA LẠI CÁCH LẤY GRAVITY VECTOR ---
    # Quaternion trong MuJoCo là (w, x, y, z)
        quat = self.data.qpos[3:7]
        qw, qx, qy, qz = quat
     
    # Ma trận xoay từ Global -> Local (Công thức chuẩn)
    # Hàng 3 của ma trận xoay (hướng trục Z toàn cục trong hệ Local)
        grav_x = 2 * (qx * qz - qw * qy)
        grav_y = 2 * (qy * qz + qw * qx)
        grav_z = 1 - 2 * (qx**2 + qy**2)
        projected_gravity = np.array([grav_x, grav_y, grav_z])
    
        prev_action_flat = self.prev_action.flatten()
    
        return np.concatenate([
        qpos_filtered,
        qvel,
        prev_action_flat,
        projected_gravity
        ]).astype(np.float32)

    def _compute_reward(self, action):
        z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        qw, qx, qy, qz = quat

    # === 1. Chiều cao (Target phụ thuộc vào góc gối) ===
    # Robot 2 chân gập gối thì chiều cao trung tâm (base) sẽ THẤP hơn.
    # Đặt target là 0.38 hoặc 0.39 thay vì 0.5.
        target_height = 0.39  # ĐIỀU CHỈNH THAM SỐ NÀY SAU KHI CHẠY 100 BƯỚC TEST
        height_reward = 15.0 * np.exp(-30.0 * (z - target_height)**2)

    # === 2. Thưởng đứng thẳng (Thân trên không nghiêng) ===
    # qw = 1.0 là đứng thẳng hoàn hảo
        upright_reward = 10.0 * (qw**2) 

    # === 3. PHẠT NẶNG nếu thân trên bị nghiêng (Pitch/Roll) ===
    # Dùng vector trọng lực chiếu lên trục X và Y của robot
        grav_x = 2 * (qx * qz - qw * qy)
        grav_y = 2 * (qy * qz + qw * qx)
    # Phạt nếu trọng lực không hướng thẳng xuống dưới (grav_z ~ -1)
        orientation_penalty = 15.0 * (grav_x**2 + grav_y**2)

    # === 4. THƯỞNG GẬP GỐI (SỬA LỖI LOGIC) ===
    # Lấy góc khớp gối (Hip3)
    # Lưu ý: Trong XML của bạn, hip1 là Roll, hip2 là Hông trước/sau, hip3 là Gối
        q_knee_r = self.data.qpos[9]   # hip3r
        q_knee_l = self.data.qpos[12]  # hip3l
    
    # Góc mong muốn (dương là gập về trước). 
    # Vì robot của bạn có thể đang hướng về phía khác, cần kiểm tra dấu.
    # Nếu gập gối làm chân đá về sau, target = 0.8 rad. 
    # Nếu gập gối làm chân đá về trước (đầu gối chúc xuống đất), target = 0.4 rad.
        target_knee = 0.2
    
    # THƯỞNG: Khoảng cách đến target càng nhỏ càng tốt
    # KHÔNG DÙNG abs() bên trong exp trừ khi muốn đối xứng tuyệt đối. Ở đây ta muốn đúng dấu.
        knee_diff_r = q_knee_r - target_knee
        knee_diff_l = q_knee_l - target_knee
        posture_reward = 5.0 * (np.exp(-8.0 * knee_diff_r**2) + np.exp(-8.0 * knee_diff_l**2))

    # === 5. ĐỐI XỨNG CHÂN (SỬA LẠI) ===
    # Chỉ thưởng khi 2 chân song song (góc bằng nhau)
        symmetry_reward = 3.0 * np.exp(-15.0 * (q_knee_r - q_knee_l)**2)

    # === 6. PHẠT TỐC ĐỘ CAO (Chống nhảy, hất chân mạnh) ===
    # Đây là chìa khóa để hết "nhảy". Phạt nặng vận tốc góc của khớp gối và hông.
    # self.data.qvel[6:12] là vận tốc các khớp chân (6 khớp)
        joint_velocities = self.data.qvel[6:12]
        velocity_penalty = 0.005 * np.sum(np.square(joint_velocities))
    
    # === 7. Chi phí điều khiển ===
        ctrl_cost = 0.001 * np.sum(np.square(action))
    
    # === TỔNG HỢP ===
        alive_bonus = 2.0 # Giảm alive bonus để nó tập trung vào dáng hơn
    
        reward = (
        alive_bonus +
        height_reward +
        upright_reward +
        posture_reward +
        symmetry_reward -
        orientation_penalty -           
        velocity_penalty -
        ctrl_cost)
        return reward

    def _check_fallen(self):
        z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        
        if z < 0.2 or abs(quat[0]) < 0.6:
            return True
        return False
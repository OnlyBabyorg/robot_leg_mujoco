import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class CustomBipedalEnv(gym.Env):
    def __init__(self, xml_path="scene.xml", target_knee=0.3, target_height=0.35):
        super(CustomBipedalEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.target_knee = target_knee
        self.target_height = target_height

        self.frame_skip = 20
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

        self.foot_r_id = self.model.geom('contact_foot_r').id
        self.foot_l_id = self.model.geom('contact_foot_l').id

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
    def _knee_too_low(self):
        knee_r_z = self.data.body('knee_right').xpos[2]
        knee_l_z = self.data.body('knee_left').xpos[2]
        return knee_r_z < 0.06 or knee_l_z < 0.06

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        qpos0_target = self.model.qpos0.copy()
        qpos0_target[7] = 0.0     # hip1r
        qpos0_target[8] = -0.25   # Đùi phải
        qpos0_target[9] = 0.2     # Gối phải
        qpos0_target[10] = 0.0    # hip1l
        qpos0_target[11] = -0.25  # Đùi trái
        qpos0_target[12] = 0.2    # Gối trái

        noise_scale = 0.02
        qpos_noise = np.random.uniform(-noise_scale, noise_scale, size=self.model.nq)
        qpos_noise[0:7] = 0
        qvel_noise = np.random.uniform(-0.01, 0.01, size=self.model.nv)
        qvel_noise[0:6] = 0

        self.data.qpos[:] = qpos0_target + qpos_noise
        self.data.qvel[:] = qvel_noise

        return self._get_obs(), {}

    def _get_obs(self):
        qpos_filtered = self.data.qpos[2:]
        qvel = self.data.qvel

        quat = self.data.qpos[3:7]
        qw, qx, qy, qz = quat
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

    def _has_contact(self, geom_id):
        """Kiểm tra geom có đang tiếp xúc với vật khác không."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == geom_id or contact.geom2 == geom_id) and contact.dist <= 0.0:
                return True
        return False

    def _compute_reward(self, action):
        z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        qw, qx, qy, qz = quat
        thigh_r, knee_r = self.data.qpos[8], self.data.qpos[9]
        thigh_l, knee_l = self.data.qpos[11], self.data.qpos[12]
        vx, vy = self.data.qvel[0], self.data.qvel[1]

    # --- Chiều cao mục tiêu (TĂNG LÊN) ---
        target_height = 0.40
        height_reward = 15.0 * np.exp(-12.0 * (z - target_height)**2)
        if z < 0.32:  # Phạt sớm và mạnh hơn
            height_reward -= 10.0 * (0.32 - z)**2

    # --- Thăng bằng thân ---
        if abs(qw) < 0.75:
            upright_reward = -8.0
        else:
            upright_reward = 6.0 * (abs(qw) - 0.75)
        grav_x = 2 * (qx * qz - qw * qy)
        grav_y = 2 * (qy * qz + qw * qx)
        orientation_penalty = 8.0 * (grav_x**2 + grav_y**2)

    # --- Tốc độ (mục tiêu = 0) ---
        target_vx = 0.0
        if abs(vx) > 0.05:
            speed_reward = -8.0 * (abs(vx) - 0.05)**2
        else:
            speed_reward = 3.0 * np.exp(-30.0 * vx**2)
        lateral_penalty = 2.0 * vy**2

    # --- Đối xứng & lệch đùi ---
        symmetry_penalty = 10.0 * (thigh_r - thigh_l)**2 + 5.0 * (knee_r - knee_l)**2
        thigh_deviation = 2.0 * (thigh_r**2 + thigh_l**2)

    # --- Phạt thẳng gối (GIỮ NGUYÊN) ---
        straight_knee_penalty = 0.0
        for knee in [knee_r, knee_l]:
            if abs(knee) < 0.15:
                straight_knee_penalty += 20.0 * (0.15 - abs(knee))**2

    # --- Phạt gập gối quá sâu (MỚI: ngưỡng 0.5 rad) ---
        deep_squat_penalty = 0.0
        for knee in [knee_r, knee_l]:
            if knee > 0.5:
                deep_squat_penalty += 3.0 * (knee - 0.5)**2

    # --- Phạt công suất khớp ---
        torque = self.data.actuator_force
        joint_vel = self.data.qvel[6:12]
        power_penalty = 0.0005 * np.sum(np.abs(torque * joint_vel))

    # --- Thưởng bước luân phiên ---
        contact_r = self._has_contact(self.foot_r_id)
        contact_l = self._has_contact(self.foot_l_id)
        no_fly_reward = 1.5 * (1 if (contact_r != contact_l) else 0)

    # --- Phạt đứng yên hoàn toàn ---
        joint_vel_sum = np.sum(np.square(joint_vel))
        if joint_vel_sum < 0.01:
            inactivity_penalty = 5.0
        else:
            inactivity_penalty = 0.0

    # --- Phạt gối quá thấp (chạm đất) ---
        knee_too_low_penalty = 0.0
        if self._knee_too_low():
            knee_too_low_penalty = 20.0  # Phạt cực nặng

    # --- Phạt giật cục ---
        action_change = np.sum(np.square(action - self.prev_action))
        smoothness_penalty = 0.5 * action_change

    # --- Tổng hợp ---
        reward = (height_reward
              + upright_reward
              - orientation_penalty
              + speed_reward
              - lateral_penalty
              - symmetry_penalty
              - thigh_deviation
              - straight_knee_penalty
              - deep_squat_penalty
              - power_penalty
              + no_fly_reward
              - inactivity_penalty
              - knee_too_low_penalty
              - smoothness_penalty
              + 0.3)
        return reward

    def _check_fallen(self):
        z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        if z < 0.2 or abs(quat[0]) < 0.6:
            return True
        return False

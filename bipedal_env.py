import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from rewards import compute_reward
from privileged_info import get_privileged_info

class CustomBipedalEnv(gym.Env):
    def __init__(self, xml_path="scene.xml", target_knee=0.3, target_height=0.35):
        super(CustomBipedalEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.target_knee = target_knee
        self.target_height = target_height

        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip

        # Lưu thông tin cho privileged info & domain randomization
        self.body_id = self.model.body('base_link').id
        self.nominal_mass = self.model.body_mass[self.body_id]
        self.body_mass = self.nominal_mass
        self.friction = 1.0
        self.ground_geom_id = self.model.geom('floor').id
        self.joint_ids = [self.model.joint(j).id for j in [
            'hip1l', 'hip1r', 'hip2l', 'hip2r', 'hip3l', 'hip3r'
        ]]
        self.nominal_damping = [self.model.dof_damping[j_id] for j_id in self.joint_ids]
        self.current_damping = self.nominal_damping.copy()   # <-- thêm dòng này

        # Observation space dạng Dict
        propro_dim = (self.model.nq - 2) + self.model.nv + self.model.nu + 3
        priv_dim = 2 + len(self.joint_ids)
        self.observation_space = spaces.Dict({
            "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(propro_dim,), dtype=np.float32),
            "privileged": spaces.Box(low=-np.inf, high=np.inf, shape=(priv_dim,), dtype=np.float32)
        })

        # Lưu trạng thái ban đầu
        mujoco.mj_resetData(self.model, self.data)
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()
        self.prev_action = np.zeros(self.model.nu)
        self.last_action = np.zeros(self.model.nu)

        # Foot contact geom ids
        self.foot_r_id = self.model.geom('contact_foot_r').id
        self.foot_l_id = self.model.geom('contact_foot_l').id

        # Địa chỉ qpos cho reward
        self.joint_qposadr = {
            'hip1r': self.model.jnt_qposadr[self.model.joint('hip1r').id],
            'hip1l': self.model.jnt_qposadr[self.model.joint('hip1l').id],
            'hip2r': self.model.jnt_qposadr[self.model.joint('hip2r').id],
            'hip2l': self.model.jnt_qposadr[self.model.joint('hip2l').id],
            'hip3r': self.model.jnt_qposadr[self.model.joint('hip3r').id],
            'hip3l': self.model.jnt_qposadr[self.model.joint('hip3l').id],
        }

        self.target_vx = 0.1   # khởi đầu cho curriculum

    def set_target_vx(self, vx):
        self.target_vx = vx

    def step(self, action):
        scaled_action = self.ctrl_ranges[:, 0] + (action + 1.0) * 0.5 * (self.ctrl_ranges[:, 1] - self.ctrl_ranges[:, 0])
        self.data.ctrl[:] = scaled_action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        self.last_action = action.copy()
        reward, reward_info = compute_reward(self)

        terminated = self._check_fallen()
        truncated = False

        self.prev_action = action.copy()
        info = {'reward_components': reward_info}
        return obs, reward, terminated, truncated, info

    def _knee_too_low(self):
        knee_r_z = self.data.body('knee_right').xpos[2]
        knee_l_z = self.data.body('knee_left').xpos[2]
        return knee_r_z < 0.06 or knee_l_z < 0.06

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Tạm tắt domain randomization
        self.body_mass = self.nominal_mass
        self.model.body_mass[self.body_id] = self.nominal_mass
        self.friction = 1.0
        self.model.geom_friction[self.ground_geom_id] = 1.0
        for i, j_id in enumerate(self.joint_ids):
            damping = self.nominal_damping[i]
            self.model.dof_damping[j_id] = damping
            self.current_damping[i] = damping

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.prev_action = np.zeros(self.model.nu)
        self.last_action = np.zeros(self.model.nu)
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

        propro = np.concatenate([
            qpos_filtered,
            qvel,
            prev_action_flat,
            projected_gravity
        ]).astype(np.float32)

        priv = get_privileged_info(self)
        return {"proprio": propro, "privileged": priv}

    def _has_contact(self, geom_id):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == geom_id or contact.geom2 == geom_id) and contact.dist <= 0.0:
                return True
        return False

    def _check_fallen(self):
        z = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        if z < 0.2 or abs(quat[0]) < 0.6 or self._knee_too_low():
            return True
        return False
import numpy as np

REWARD_WEIGHTS = {
    'height': 10.0,
    'balance': -5.0,
    'velocity': 2.0,
    'lateral_velocity': -2.0,
    'asymmetry': -3.0,
    'energy': -0.001,
    'alive': 0.1,
}

def compute_reward(env):
    data = env.data
    qpos = data.qpos
    qvel = data.qvel
    action = env.last_action

    right_roll = qpos[env.joint_qposadr['hip1r']]
    left_roll  = qpos[env.joint_qposadr['hip1l']]
    right_hip  = qpos[env.joint_qposadr['hip2r']]
    left_hip   = qpos[env.joint_qposadr['hip2l']]
    right_knee = qpos[env.joint_qposadr['hip3r']]
    left_knee  = qpos[env.joint_qposadr['hip3l']]

    torso_z = qpos[2]
    height_error = abs(torso_z - env.target_height)
    reward_height = np.exp(-10.0 * height_error) * REWARD_WEIGHTS['height']

    quat = qpos[3:7]
    qw = quat[0]
    balance_penalty = max(0.0, 0.6 - abs(qw)) * REWARD_WEIGHTS['balance']

    vx = qvel[0]
    target_vx = 0.3
    vel_reward = -abs(vx - target_vx) * REWARD_WEIGHTS['velocity']

    vy = qvel[1]
    lat_penalty = -abs(vy) * REWARD_WEIGHTS['lateral_velocity']

    asym_roll = abs(right_roll - left_roll)
    asym_thigh = abs(right_hip - left_hip)
    asym_penalty = (asym_roll + asym_thigh) * REWARD_WEIGHTS['asymmetry']

    energy_penalty = -np.sum(np.square(action)) * REWARD_WEIGHTS['energy']

    alive_bonus = REWARD_WEIGHTS['alive']

    reward = (reward_height + balance_penalty + vel_reward +
              lat_penalty + asym_penalty + energy_penalty + alive_bonus)

    reward_info = {
        'height': reward_height,
        'balance': balance_penalty,
        'velocity': vel_reward,
        'lateral': lat_penalty,
        'asymmetry': asym_penalty,
        'energy': energy_penalty,
        'alive': alive_bonus,
        'total': reward
    }
    return reward, reward_info
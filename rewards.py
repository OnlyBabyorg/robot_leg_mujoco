import numpy as np

REWARD_WEIGHTS = {
    'height': 15.0,
    'balance': -10.0,
    'velocity': 4.0,
    'lateral_velocity': -6.0,      # TĂNG để phạt đi ngang mạnh hơn
    'ang_vel_yaw': -2.0,           # MỚI: phạt xoay
    'asymmetry': -2.0,
    'posture': -0.5,
    'action_rate': -0.2,
    'energy': -0.005,
    'alive': 0.0,
    'foot_clearance': 5.0,
    'contact_schedule': 3.0,
}

def compute_reward(env):
    data = env.data
    qpos = data.qpos
    qvel = data.qvel
    action = env.last_action
    prev_action = env.prev_action

    right_roll = qpos[env.joint_qposadr['hip1r']]
    left_roll  = qpos[env.joint_qposadr['hip1l']]
    right_hip  = qpos[env.joint_qposadr['hip2r']]
    left_hip   = qpos[env.joint_qposadr['hip2l']]
    right_knee = qpos[env.joint_qposadr['hip3r']]
    left_knee  = qpos[env.joint_qposadr['hip3l']]

    # 1. Chiều cao
    torso_z = qpos[2]
    height_error = abs(torso_z - env.target_height)
    reward_height = np.exp(-10.0 * height_error) * REWARD_WEIGHTS['height']

    # 2. Thăng bằng
    quat = qpos[3:7]
    qw = quat[0]
    balance_penalty = max(0.0, 0.6 - abs(qw)) * REWARD_WEIGHTS['balance']

    # 3. Vận tốc dọc
    vx = qvel[0]
    target_vx = env.target_vx
    vel_reward = -abs(vx - target_vx) * REWARD_WEIGHTS['velocity']

    # 4. Vận tốc ngang (local)
    vy = qvel[1]
    lat_penalty = -abs(vy) * REWARD_WEIGHTS['lateral_velocity']

    # 5. Phạt xoay (yaw angular velocity)
    wz = qvel[5]  # vận tốc góc z (yaw)
    ang_yaw_penalty = -np.square(wz) * REWARD_WEIGHTS['ang_vel_yaw']

    # 6. Bất đối xứng
    asym_roll = abs(right_roll - left_roll)
    asym_thigh = abs(right_hip - left_hip)
    asym_penalty = (asym_roll + asym_thigh) * REWARD_WEIGHTS['asymmetry']

    # 7. Tư thế khớp
    joint_angles = np.array([right_roll, left_roll, right_hip, left_hip, right_knee, left_knee])
    posture_penalty = -np.sum(np.square(joint_angles)) * REWARD_WEIGHTS['posture']

    # 8. Mượt hành động
    action_rate_penalty = -np.sum(np.square(action - prev_action)) * REWARD_WEIGHTS['action_rate']
    # 9. Năng lượng
    energy_penalty = -np.sum(np.square(action)) * REWARD_WEIGHTS['energy']

    # 10. Chân nhấc cao
    foot_r_z = env.data.geom('contact_foot_r').xpos[2]
    foot_l_z = env.data.geom('contact_foot_l').xpos[2]
    foot_clearance = (max(0.0, foot_r_z - 0.02) + max(0.0, foot_l_z - 0.02)) * REWARD_WEIGHTS['foot_clearance']

    # 11. Lịch tiếp xúc
    contact_r = env._has_contact(env.foot_r_id)
    contact_l = env._has_contact(env.foot_l_id)
    single_contact = (contact_r != contact_l)
    contact_schedule_reward = single_contact * REWARD_WEIGHTS['contact_schedule']
    double_air = (not contact_r and not contact_l)
    contact_schedule_reward -= double_air * REWARD_WEIGHTS['contact_schedule'] * 0.5

    alive_bonus = REWARD_WEIGHTS['alive']

    reward = (reward_height + balance_penalty + vel_reward +
              lat_penalty + ang_yaw_penalty + asym_penalty + posture_penalty +
              action_rate_penalty + energy_penalty + foot_clearance +
              contact_schedule_reward + alive_bonus)

    reward_info = {
        'height': reward_height,
        'balance': balance_penalty,
        'velocity': vel_reward,
        'lateral': lat_penalty,
        'ang_yaw': ang_yaw_penalty,
        'asymmetry': asym_penalty,
        'posture': posture_penalty,
        'action_rate': action_rate_penalty,
        'energy': energy_penalty,
        'foot_clearance': foot_clearance,
        'contact_schedule': contact_schedule_reward,
        'alive': alive_bonus,
        'total': reward
    }
    return reward, reward_info
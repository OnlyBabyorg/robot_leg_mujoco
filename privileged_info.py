import numpy as np

def get_privileged_info(env):
    """
    Trả về vector privileged info từ môi trường.
    env phải có các thuộc tính lưu giá trị thực tế sau DR.
    """

    mass_noise = (env.body_mass / env.nominal_mass) - 1.0
    friction_noise = env.friction - 1.0
    damping_noises = [
        (env.model.dof_damping[j_id] / env.nominal_damping[i]) - 1.0
        for i, j_id in enumerate(env.joint_ids)
    ]
    return np.array([mass_noise, friction_noise] + damping_noises, dtype=np.float32)
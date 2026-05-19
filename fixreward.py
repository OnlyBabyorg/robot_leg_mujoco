def _compute_reward(self, action):
    z = self.data.qpos[2]  
    quat = self.data.qpos[3:7]
    qw, qx, qy, qz = quat
    

    thigh_r = self.data.qpos[8]
    knee_r = self.data.qpos[9]
    thigh_l = self.data.qpos[11]
    knee_l = self.data.qpos[12]
    
    
    vx = self.data.qvel[0]
    vy = self.data.qvel[1]
    
    min_height = 0.28        
    target_height = 0.38     
    if z < min_height:
        height_reward = -10.0  
    else:
        
        height_diff = abs(z - target_height)
        height_reward = 5.0 * np.exp(-8.0 * height_diff)
   
    if abs(qw) < 0.6:
        upright_reward = -15.0
    else:
        upright_reward = 8.0 * (abs(qw) - 0.6)  
    
  
    grav_x = 2 * (qx * qz - qw * qy)
    grav_y = 2 * (qy * qz + qw * qx)
    orientation_penalty = 10.0 * (grav_x**2 + grav_y**2)
    
    # ----- 3. Tư thế khớp (khom vừa phải) -----
    target_thigh = -0.25   # Đùi hơi nghiêng
    target_knee = 0.3      # Gối gập nhẹ
    
    thigh_diff_r = abs(thigh_r - target_thigh)
    thigh_diff_l = abs(thigh_l - target_thigh)
    knee_diff_r = abs(knee_r - target_knee)
    knee_diff_l = abs(knee_l - target_knee)
    
    # Thưởng nếu gần target, nhưng phạt nếu vượt quá xa
    def bounded_reward(diff, limit=0.5):
        if diff < limit:
            return 2.0 * np.exp(-5.0 * diff)
        else:
            return -1.0 * diff  # Phạt tuyến tính nếu lệch xa
    
    thigh_reward = bounded_reward(thigh_diff_r) + bounded_reward(thigh_diff_l)
    knee_reward = bounded_reward(knee_diff_r) + bounded_reward(knee_diff_l)
    
    # Phạt nếu gối gập quá sâu (knee > 0.8 rad ~ 45°)
    if knee_r > 0.8:
        knee_reward -= 2.0 * (knee_r - 0.8)
    if knee_l > 0.8:
        knee_reward -= 2.0 * (knee_l - 0.8)
    
    # ----- 4. Khuyến khích di chuyển về trước (thăng bằng động) -----
    # Robot point-foot cần bước liên tục, nên thưởng vận tốc tiến nhỏ
    target_vx = 0.25  # m/s, đi chậm
    if vx < 0:
        speed_reward = -2.0 * abs(vx)  # Phạt đi lùi
    elif vx < 0.05:
        # Đứng yên cũng phạt nhẹ (vì robot point-foot không thể đứng yên)
        speed_reward = -1.0
    else:
        speed_reward = 2.0 * np.exp(-8.0 * (vx - target_vx)**2)
    
    # Phạt vận tốc ngang (giữ thẳng hướng)
    lateral_penalty = 2.0 * vy**2
    
    # ----- 5. Phạt năng lượng và giật cục -----
    joint_vel = np.sum(np.square(self.data.qvel[6:12]))
    velocity_penalty = 0.002 * joint_vel
    
    action_change = np.sum(np.square(action - self.prev_action))
    smoothness_penalty = 0.3 * action_change
    
    ctrl_cost = 0.001 * np.sum(np.square(action))
    
    # ----- 6. Đối xứng chân -----
    sym_diff = abs(thigh_r - thigh_l) + abs(knee_r - knee_l)
    symmetry_reward = 1.0 * np.exp(-3.0 * sym_diff)
    
    # Tổng hợp
    reward = (height_reward
              + upright_reward
              - orientation_penalty
              + thigh_reward
              + knee_reward
              + speed_reward
              - lateral_penalty
              + symmetry_reward
              - velocity_penalty
              - smoothness_penalty
              - ctrl_cost
              + 0.5)  # Alive bonus nhỏ
    
    return reward
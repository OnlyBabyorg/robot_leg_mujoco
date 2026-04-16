import mujoco
import numpy as np
from scipy.optimize import minimize
import os

# 1. Khởi tạo mô hình từ file XML (Đã bật <freejoint>)
current_dir = os.path.dirname(os.path.abspath(__file__))
scene_path = os.path.join(current_dir, "scene.xml") 
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

# 2. Định vị chính xác tâm KHỐI CẦU bàn chân
foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "contact_foot_r")
foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "contact_foot_l")

def calc_balance(q, target_z):
    # q = [hip2_r, hip3_r, hip2_l, hip3_l]
    
    # Ép tọa độ gốc của base_link
    data.qpos[0:3] = [0.0, 0.0, target_z] # X, Y, Z
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] # Quaternion thẳng đứng (w, x, y, z)
    
    # Gán góc cho các khớp (Index 7->12 tương ứng với 6 khớp bản lề sau freejoint)
    data.qpos[7] = 0.0  # hip1r (Giữ Roll = 0)
    data.qpos[8] = q[0] # hip2r
    data.qpos[9] = q[1] # hip3r
    data.qpos[10] = 0.0 # hip1l
    data.qpos[11] = q[2]# hip2l
    data.qpos[12] = q[3]# hip3l

    # Tính toán Động học thuận (Không mô phỏng vật lý)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    # Lấy tọa độ khối tâm (CoM) toàn hệ thống
    com = data.subtree_com[0]
    
    # Lấy tọa độ tuyệt đối của 2 tâm khối cầu bàn chân
    pos_r = data.geom_xpos[foot_r_id]
    pos_l = data.geom_xpos[foot_l_id]
    center_feet = (pos_r + pos_l) / 2.0

    # Sai số CoM trên mặt phẳng XY
    error_xy = (com[0] - center_feet[0])**2 + (com[1] - center_feet[1])**2
    
    # Ràng buộc đối xứng động học
    symmetry_penalty = (abs(q[0]) - abs(q[2]))**2 + (abs(q[1]) - abs(q[3]))**2
    
    return error_xy + symmetry_penalty * 10

# Khảo sát ở độ cao Z = 0.45m để tạo tư thế khuỵu gối an toàn
target_z = 0.45 
initial_guess = np.array([0.5, -0.8, -0.5, 0.8]) 

# Giới hạn góc dựa trên XML
bounds = [(-3.14, 3.14), (-1.0, 1.2), (-3.14, 3.14), (-1.0, 1.2)]

result = minimize(lambda q: calc_balance(q, target_z), initial_guess, bounds=bounds, method='SLSQP')

if result.success:
    q = result.x
  
    print("-" * 50)
    print("class init_state(LeggedRobotCfg.init_state):")
    print("    pos = [0.0, 0.0, {:.3f}] ".format(target_z + 0.02241)) # Cộng bù bán kính để không lún sàn
    print("    default_joint_positions = {")
    print("        'hip1.*': 0.0,")
    print(f"        'hip2r': {q[0]:.4f},")
    print(f"        'hip3r': {q[1]:.4f},")
    print(f"        'hip2l': {q[2]:.4f},")
    print(f"        'hip3l': {q[3]:.4f}")
    print("    }")
    print("-" * 50)
else:
    print("adfasdasd")
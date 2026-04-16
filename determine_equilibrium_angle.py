import mujoco
import numpy as np
from scipy.optimize import minimize
import os


TARGET_Z = 0.3  
ROLL_R   = -0.08
ROLL_L   = 0.08     



current_dir = os.path.dirname(os.path.abspath(__file__))
scene_path = os.path.join(current_dir, "scene.xml") 
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)

foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "contact_foot_r")
foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "contact_foot_l")

def compute_balance(q):
  
    
   
    data.qpos[0:3] = [0.0, 0.0, TARGET_Z] 
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] 
    
   
    data.qpos[7]  = ROLL_R  
    data.qpos[10] = ROLL_L  
    
  
    data.qpos[8]  = q[0]    
    data.qpos[9]  = q[1]    
    data.qpos[11] = q[2]    
    data.qpos[12] = q[3]    

    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    com = data.subtree_com[0]
    pos_r = data.geom_xpos[foot_r_id]
    pos_l = data.geom_xpos[foot_l_id]
    center_feet = (pos_r + pos_l) / 2.0

   
    error_xy = (com[0] - center_feet[0])**2 + (com[1] - center_feet[1])**2
    symmetry_penalty = (abs(q[0]) - abs(q[2]))**2 + (abs(q[1]) - abs(q[3]))**2
    
    return error_xy + symmetry_penalty * 10


initial_guess = np.array([0.5, -0.8, -0.5, 0.8]) 
bounds = [(-3.14, 3.14), (-1.0, 1.2), (-3.14, 3.14), (-1.0, 1.2)]


result = minimize(compute_balance, initial_guess, bounds=bounds, method='SLSQP')

if result.success and result.fun < 1e-4:
    q = result.x
   
    print(f"syntax: Z = {TARGET_Z} | Roll_R = {ROLL_R} | Roll_L = {ROLL_L}")
    print(f"denta: {result.fun:.8f}\n")
    print(":")
    print("default_joint_positions = {")
    print(f"    'hip1r': {ROLL_R},")
    print(f"    'hip1l': {ROLL_L},")
    print(f"    'hip2r': {q[0]:.4f},")
    print(f"    'hip3r': {q[1]:.4f},")
    print(f"    'hip2l': {q[2]:.4f},")
    print(f"    'hip3l': {q[3]:.4f}")
    print("}")
else:
    print("cant deter")
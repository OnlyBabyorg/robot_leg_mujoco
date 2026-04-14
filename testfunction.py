import mujoco
import mujoco.viewer
import time
import os
import threading
import numpy as np

# q1: Roll 
# q2: Pitch đùi
# q3: Pitch gối 
shared_cmd = {
    "R": {"q1": 0.0, "q2": 0.0, "q3": 0.0},
    "L": {"q1": 0.0, "q2": 0.0, "q3": 0.0}
}

def terminal_controller():
    print("==CONTROL BROAD==")
    print("Syntax: [leg(R/L)] [Corner_Roll] [Corner_Hip] [Coner_Knee]")
    print(" Radian ")
    print("")   
    print("enter 'q' for exit.")
    
    while True:
        try:
            cmd = input("-> command: ").strip()
            if cmd.lower() == 'q':
                os._exit(0)
                
            parts = cmd.split()
            if len(parts) == 4:
                leg = parts[0].upper()
                if leg not in ["R", "L"]:
                    print("error.")
                    continue
                    
                q1 = float(parts[1])
                q2 = float(parts[2])
                q3 = float(parts[3])
                
                shared_cmd[leg]["q1"] = q1
                shared_cmd[leg]["q2"] = q2
                shared_cmd[leg]["q3"] = q3
                print(f"[Anhdomixi] {leg} -> Roll: {q1:.2f} | Hip: {q2:.2f} | Knee: {q3:.2f}")
            else:
                print("error element ")
        except ValueError:
            print("error , real corner only .")

# main
current_dir = os.path.dirname(os.path.abspath(__file__))
# Đảm bảo đường dẫn file XML trỏ đúng vào file bạn vừa sửa 
scene_path = os.path.join(current_dir, "scene.xml") 
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)
actuator_names = ["pos_hip1r", "pos_hip2r", "pos_hip3r", "pos_hip1l", "pos_hip2l", "pos_hip3l"]
actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names]

input_thread = threading.Thread(target=terminal_controller, daemon=True)
input_thread.start()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        # 1. Bỏ comment để PID nhận lệnh từ Terminal
        data.ctrl[actuator_ids[0]] = shared_cmd["R"]["q1"]
        data.ctrl[actuator_ids[1]] = shared_cmd["R"]["q2"]
        data.ctrl[actuator_ids[2]] = shared_cmd["R"]["q3"]
        data.ctrl[actuator_ids[3]] = shared_cmd["L"]["q1"]
        data.ctrl[actuator_ids[4]] = shared_cmd["L"]["q2"]
        data.ctrl[actuator_ids[5]] = shared_cmd["L"]["q3"]     
        
        # 2. HACK ĐỘNG HỌC: Treo base_link trên không trung
        # Thẻ <freejoint> sinh ra 7 biến vị trí ở đầu mảng qpos (X, Y, Z, qw, qx, qy, qz)
        data.qpos[0:3] = [0.0, 0.0, 0.6]  # Ép thân ở tọa độ X=0, Y=0, Z=0.6 mét
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Ép thân thẳng đứng (Quaternion: không xoay)
        
        mujoco.mj_step(model, data)
        viewer.sync()     
        
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
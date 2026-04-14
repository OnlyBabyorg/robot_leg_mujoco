import mujoco
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "urdf", "test9.urdf")
xml_path = os.path.join(current_dir, "scene.xml")
model = mujoco.MjModel.from_xml_path(urdf_path)
mujoco.mj_saveLastXML(xml_path, model)
print("xuat thanh cong !")
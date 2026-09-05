# -*- coding: utf-8 -*-
"""R21-fix: NTU120 joint-count adaptive prep + P2′ serial auto-chain.

P2′ fine-tune → extract → E7 → then P5-B pretext → arms. All serial on GPU.
"""
import io, subprocess, sys, os

# ---- 1) Fix P5-B prep() to auto-detect joint count ----
p19 = 'docs/../scripts/run_p19_ntu120_retention.py'
s = io.open(p19, encoding='utf-8').read()
old_prep = '''    def prep(r):
        kp = np.asarray(r["kp"], dtype=np.float32)
        T = kp.shape[0]
        if T < 30:
            idx = np.resize(np.arange(T), 30)
        else:
            idx = np.linspace(0, T - 1, 30, dtype=int)
        kp = kp[idx]
        conf = np.ones((30, 17, 1), dtype=np.float32)
        kp3 = np.concatenate([kp, conf], axis=2)  # (30,17,3)
        return center_keypoints(kp3)'''
new_prep = '''    def prep(r):
        kp = np.asarray(r["kp"], dtype=np.float32)
        V = kp.shape[-2]  # auto-detect joints (NTU120 3D=25, HRNet 2D=17)
        T = kp.shape[0]
        if T < 30:
            idx = np.resize(np.arange(T), 30)
        else:
            idx = np.linspace(0, T - 1, 30, dtype=int)
        kp = kp[idx]
        if kp.shape[-1] < 3:
            conf = np.ones((30, V, 1), dtype=np.float32)
            kp = np.concatenate([kp, conf], axis=2)
        return center_keypoints(kp)'''
assert old_prep in s, 'prep() not found in p19'
s = s.replace(old_prep, new_prep)
# also fix dump_features which has same hardcode
old_dump = 'kp = center_keypoints(np.concatenate([kp[idx], np.ones((30, 17, 1), np.float32)], axis=2))'
new_dump = 'V2 = kp.shape[-2]\n        kp = center_keypoints(np.concatenate([kp[idx], np.ones((30, V2, 1), np.float32)], axis=2)) if kp.shape[-1] < 3 else center_keypoints(kp[idx])'
assert old_dump in s, 'dump hardcode not found'
s = s.replace(old_dump, new_dump)
io.open(p19, 'w', encoding='utf-8', newline='\n').write(s)
print('p19 joints auto-detect OK')

# ---- 2) Kill all Python except self, then chain P2′ → P5-B ----
subprocess.Popen(['taskkill', '/F', '/IM', 'python.exe'],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import time; time.sleep(3)

# ---- 3) P2′ fine-tune (GPU solo) ----
print('=== Starting P2′ YOLO11x-pose fine-tune (solo GPU) ===')
PY = r'D:\Desktop\psd-framework\.venv\Scripts\python.exe'
os.environ['PYTHONUNBUFFERED'] = '1'
r1 = subprocess.call([PY, '-c', '''
from ultralytics import YOLO
model = YOLO("yolo11x-pose.pt")
model.train(data="D:/Desktop/datasets/dog-pose/dog-pose.yaml",
    epochs=100, imgsz=640, batch=4, device=0,
    project="runs/p18_yolox_finetune", name="x_pose_dog",
    patience=20, save=True, amp=True, lr0=0.001, lrf=0.01, optimizer="AdamW")
print("FINE-TUNE DONE")
'''], cwd=r'D:\Desktop\psd-framework')

if r1 != 0:
    print(f'P2′ fine-tune FAILED (exit {r1})'); sys.exit(1)
print('=== P2′ fine-tune DONE → extract + E7 ===')

# ---- 4) P2′ extraction + E7 rerun ----
r2 = subprocess.call([PY, 'scripts/run_p18_superanimal_extract.py'],
                     cwd=r'D:\Desktop\psd-framework')
print(f'P2′ extract+E7 exit={r2}')

# ---- 5) P5-B NTU120 retention ----
print('=== Starting P5-B NTU120 (solo GPU) ===')
r3 = subprocess.call([PY, 'scripts/run_p19_ntu120_retention.py', '--pretext', '--max', '12000'],
                     cwd=r'D:\Desktop\psd-framework')
print(f'P5-B NTU120 exit={r3}')

print('=== ALL SERIAL PIPELINE DONE ===')

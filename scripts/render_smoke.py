from pathlib import Path
import numpy as np
from PIL import Image
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite=benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(0)
env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
try:
 env.seed(7);obs=env.reset()
 for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
 out=Path(__file__).resolve().parents[1]/'logs'
 for k in ['agentview_image','robot0_eye_in_hand_image']:
  a=obs[k];assert a.shape==(256,256,3) and np.isfinite(a).all() and a.std()>1;Image.fromarray(a).save(out/(k+'.png'))
 print('RENDER_PASS',task.language)
finally:env.close()

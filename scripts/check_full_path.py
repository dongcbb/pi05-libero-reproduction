import torch
from openpi.models.pi0_config import Pi0Config
from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
c=Pi0Config(pi05=True,paligemma_variant='dummy',action_expert_variant='dummy',action_dim=8,action_horizon=4,dtype='float32',pytorch_compile_mode=None)
m=PI0Pytorch(c).cuda()
o=Observation(images={k:torch.rand(1,3,224,224,device='cuda')*2-1 for k in ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']},image_masks={k:torch.ones(1,dtype=torch.bool,device='cuda') for k in ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']},state=torch.zeros(1,8,device='cuda'),tokenized_prompt=torch.ones(1,8,dtype=torch.long,device='cuda'),tokenized_prompt_mask=torch.ones(1,8,dtype=torch.bool,device='cuda'))
m.train();loss=m(o,torch.randn(1,4,8,device='cuda')).mean();print('LOSS',float(loss.detach()),flush=True);loss.backward();assert torch.isfinite(loss)
m.eval()
with torch.no_grad():a=m.sample_actions('cuda',o,num_steps=2)
assert a.shape==(1,4,8) and torch.isfinite(a).all();print('FULL_PATH_PASS',a.shape,flush=True)

import torch
from openpi.models import gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
m=PaliGemmaWithExpertModel(gemma.get_config('dummy'),gemma.get_config('dummy'),use_adarms=[False,True],precision='float32').cuda().eval()
p=torch.randn(1,3,64,device='cuda');s=torch.randn(1,2,64,device='cuda');cond=torch.randn(1,64,device='cuda');mask=torch.zeros(1,1,5,5,device='cuda');mask[:,:,:3,3:]=torch.finfo(torch.float32).min
with torch.no_grad():
 out,_=m(inputs_embeds=[p,s],attention_mask=mask,position_ids=torch.arange(5,device='cuda')[None],adarms_cond=[None,cond])
 _,cache=m(inputs_embeds=[p,None],attention_mask=mask[:,:,:3,:3],position_ids=torch.arange(3,device='cuda')[None],use_cache=True)
 cached,_=m(inputs_embeds=[None,s],attention_mask=mask[:,:,3:,:],position_ids=torch.arange(3,5,device='cuda')[None],past_key_values=cache,use_cache=False,adarms_cond=[None,cond])
 print('joint/cache max error',float((out[1]-cached[1]).abs().max()))
 torch.testing.assert_close(out[1],cached[1],atol=1e-5,rtol=1e-4)
assert cache.get_seq_length()==3, 'Suffix inference mutated prefix cache'
m.train();out,_=m(inputs_embeds=[p,s],attention_mask=mask,position_ids=torch.arange(5,device='cuda')[None],adarms_cond=[None,cond]);loss=out[1].square().mean();loss.backward();assert m.gemma_expert.model.layers[0].input_layernorm.dense.weight.grad is not None
print('PASS joint forward, cached inference equivalence, adaptive norm backward')

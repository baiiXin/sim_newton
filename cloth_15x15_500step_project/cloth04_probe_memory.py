"""Probe CUDA memory for a proposed 15x15 model/training microbatch."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from cloth03_solvers_and_models import MLPOptimizer, ModelSpec
from cloth05_train_models import SampleSource, make_train_chunk, rollout_loss
from cloth_common import load_json, load_physical, save_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('cloth_15x15_500step_pipeline'))
    p.add_argument('--sample-source-root',type=Path,default=None); p.add_argument('--device',default='cuda:0')
    p.add_argument('--activation',default='relu'); p.add_argument('--depth',type=int,default=10); p.add_argument('--width',type=int,default=4096)
    p.add_argument('--use-bias',action='store_true'); p.add_argument('--sample-count',type=int,default=32); p.add_argument('--k',type=int,default=30)
    p.add_argument('--time-width',type=int,default=32); p.add_argument('--output',type=Path,default=None); a=p.parse_args()
    device=torch.device(a.device); root=a.root; source_root=a.sample_source_root or root/'data'/'samples'
    manifest=load_json(root/'data'/'datasets'/'train_manifest.json'); source=SampleSource(source_root,manifest)
    physical=load_physical(root); spec=ModelSpec(a.activation,a.depth,a.width,a.use_bias); model=MLPOptimizer(5e-2,spec).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    batch=make_train_chunk(source=source,reference=None,motions=[int(v) for v in manifest['train_motion_indices']],time_start=0,time_stop=a.time_width,sample_start=0,sample_stop=a.sample_count,device=device)
    optimizer.zero_grad(set_to_none=True); loss,_=rollout_loss(model,batch,physical,a.k,5e-2); loss.backward(); optimizer.step(); torch.cuda.synchronize(device)
    result={'model_spec':spec.__dict__,'parameter_count':model.parameter_count,'batch_points':int(batch['initial_y'].shape[0]),'k':a.k,'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30,'loss':float(loss.detach().cpu())}
    print(json.dumps(result,indent=2)); save_json(result,a.output or root/'memory_probe.json')
if __name__=='__main__': main()

"""Run a 500-frame continuous rollout on the hardest converged test motion."""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import torch
from cloth03_solvers_and_models import FIXED_VERTEX_INDICES, MLPOptimizer, ModelSpec, NUM_PARTICLES, SPATIAL_DIM, TORCH_DTYPE, apply_model_update, full_state_from_free_state, full_state_from_positions, make_q_free, physical_config_from_dict, project_fixed_vertices, stationarity_residual_norm_full
from cloth_common import load_json, save_json

def choose_motion(root:Path,excluded:set[int],candidates:list[int])->int:
    audit=load_json(root/'data'/'reference'/'residual_audit'/'reference_audit.json'); rows={int(r['motion_index']):r for r in audit['ranking_rows']}
    valid=[i for i in candidates if i not in excluded and rows[i]['num_nonfinite']==0]
    if not valid: raise RuntimeError('no finite candidate test motions')
    return max(valid,key=lambda i:float(rows[i]['residual_p95']))

def load_model(path:Path,device):
    c=torch.load(path,map_location=device); s=c['model_spec']; spec=ModelSpec(str(s['activation']),int(s['depth']),int(s['width']),bool(s['use_bias']))
    scale=float(c.get('config',{}).get('residual_length_scale',5e-2)); m=MLPOptimizer(scale,spec).to(device); m.load_state_dict(c['model_state_dict']); m.eval(); return m,spec

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('cloth_15x15_500step_pipeline')); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--motion-index',type=int,default=None); p.add_argument('--candidate-motions',type=int,nargs='*',default=list(range(20,32))); p.add_argument('--exclude-motion-indices',type=int,nargs='*',default=[])
    p.add_argument('--rollout-length',type=int,default=500); p.add_argument('--inner-steps',type=int,default=50); p.add_argument('--device',default='cuda:0'); p.add_argument('--output',type=Path,default=None); a=p.parse_args()
    excluded=set(a.exclude_motion_indices); exfile=a.root/'data'/'motion_exclusions.json'
    if exfile.exists(): excluded.update(load_json(exfile).get('excluded_motion_indices',[]))
    motion=a.motion_index if a.motion_index is not None else choose_motion(a.root,excluded,a.candidate_motions)
    device=torch.device(a.device); runtime=load_json(a.root/'data'/'reference'/'runtime_config.json'); physical=physical_config_from_dict(runtime['physical_config'])
    states=torch.load(a.root/'data'/'reference'/'reference_motion_states.pt',map_location='cpu'); ids=[int(v) for v in states['motion_index'].tolist()]; row=ids.index(motion)
    reference=states['positions'][row,:a.rollout_length+1]; refv=states['velocities'][row,:a.rollout_length+1]
    model,spec=load_model(a.checkpoint,device); masses=torch.tensor([physical.masses[i] for i in range(NUM_PARTICLES) if i not in set(FIXED_VERTEX_INDICES)],dtype=TORCH_DTYPE,device=device).reshape(1,-1)
    positions=[reference[0].clone()]; velocities=[refv[0].clone()]; curves=[]; errors=[]; elapsed=[]
    for frame in range(a.rollout_length):
        t=time.perf_counter(); pn=positions[-1].to(device); vn=velocities[-1].to(device); qf=make_q_free(pn,vn,physical).reshape(1,-1); q=project_fixed_vertices(full_state_from_free_state(qf,physical),physical); y=project_fixed_vertices(full_state_from_positions(pn).reshape(1,-1),physical)
        pr=torch.zeros_like(y); pu=torch.zeros_like(y); rc=[float(stationarity_residual_norm_full(y,q,masses,physical).item())]
        for _ in range(a.inner_steps): y,d,r=apply_model_update(model,y,q,masses,physical,previous_residual=pr,previous_update=pu); pr=r.detach(); pu=d.detach(); rc.append(float(stationarity_residual_norm_full(y,q,masses,physical).item()))
        if not torch.isfinite(y).all() or not all(math.isfinite(v) for v in rc): print(f'failed at frame {frame}'); break
        pnext=y.reshape(NUM_PARTICLES,SPATIAL_DIM); vnext=(pnext-pn)/physical.dt; vnext[list(FIXED_VERTEX_INDICES)]=0
        positions.append(pnext.cpu()); velocities.append(vnext.cpu()); curves.append(torch.tensor(rc,dtype=TORCH_DTYPE)); errors.append(float(torch.linalg.vector_norm(pnext-reference[frame+1].to(device)).item())); elapsed.append(time.perf_counter()-t)
        if frame==0 or (frame+1)%25==0: print(f'motion={motion} frame={frame+1}/{a.rollout_length} residual={rc[-1]:.3e} error={errors[-1]:.3e}')
    out=a.output or a.root/'rollouts'/f'motion_{motion:03d}'/spec.experiment_name/'curve.pt'; out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'motion_index':motion,'model_spec':spec.__dict__,'positions':torch.stack(positions),'velocities':torch.stack(velocities),'residual_by_frame_and_iteration':torch.stack(curves) if curves else torch.empty(0,a.inner_steps+1),'reference_error_by_frame':torch.tensor(errors),'elapsed_seconds_by_frame':torch.tensor(elapsed),'metadata':{'requested_rollout_length':a.rollout_length,'completed_frames':len(curves),'inner_steps':a.inner_steps,'selection':'highest reference residual_p95 among finite test motions'}},out)
    save_json({'motion_index':motion,'checkpoint':str(a.checkpoint),'output':str(out),'completed_frames':len(curves)},out.with_suffix('.json')); print(out)
if __name__=='__main__': main()

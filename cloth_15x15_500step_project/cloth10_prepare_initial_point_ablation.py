"""Prepare nested {1,8,32,128,512,1024} perturbation prefixes in window shards.

Only `initial_y` is duplicated across the perturbation axis. q, masses and exact_y
remain in the shared reference file. Shards are aligned with training time windows
so the 1024-point experiment can be memory-mapped without loading ~33 GiB at once.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from cloth03_solvers_and_models import FULL_STATE_DIM, TORCH_DTYPE, TRAIN_SOBOL_SEED, full_state_from_free_state, generate_sobol_points, physical_config_from_dict, project_fixed_vertices
from cloth_common import load_json, resolve_exclusions, save_json

DEFAULT_COUNTS=(1,8,32,128,512,1024)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('cloth_15x15_500step_pipeline'))
    p.add_argument('--output-root',type=Path,default=None); p.add_argument('--sample-counts',type=int,nargs='+',default=list(DEFAULT_COUNTS)); p.add_argument('--max-points',type=int,default=1024)
    p.add_argument('--train-time-stop',type=int,default=400); p.add_argument('--time-window',type=int,default=32); p.add_argument('--seed',type=int,default=TRAIN_SOBOL_SEED)
    p.add_argument('--exclude-motion-indices',type=int,nargs='*',default=[]); p.add_argument('--exclusion-file',type=Path,default=None); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
    counts=tuple(a.sample_counts)
    if tuple(sorted(set(counts)))!=counts or counts[-1]>a.max_points or counts[0]<=0: raise ValueError('sample-counts must be sorted unique positive prefixes <= max-points')
    runtime=load_json(a.root/'data'/'reference'/'runtime_config.json'); physical=physical_config_from_dict(runtime['physical_config'])
    reference=torch.load(a.root/'data'/'reference'/'reference_problems.pt',map_location='cpu'); exfile=a.exclusion_file or a.root/'data'/'motion_exclusions.json'; excluded=set(resolve_exclusions(a.exclude_motion_indices,exfile))
    motions=[i for i in range(16) if i not in excluded]; out=a.output_root or a.root/'data'/'initial_point_ablation'/'max_1024'; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for motion in motions:
        for start in range(0,a.train_time_stop,a.time_window):
            stop=min(start+a.time_window,a.train_time_stop); path=out/f'motion_{motion:03d}'/f'time_{start:03d}_{stop-1:03d}.pt'
            if path.exists() and not a.overwrite:
                records.append({'motion_index':motion,'time_start':start,'time_stop':stop,'path':str(path.resolve()),'reused':True}); continue
            mask=(reference['motion_index']==motion)&(reference['time_index']>=start)&(reference['time_index']<stop); rows=torch.nonzero(mask,as_tuple=False).flatten(); rows=rows[torch.argsort(reference['time_index'].index_select(0,rows))]
            initial=torch.empty((stop-start,a.max_points,FULL_STATE_DIM),dtype=TORCH_DTYPE)
            for local,row in enumerate(rows.tolist()):
                time_index=int(reference['time_index'][row]); points,_=generate_sobol_points(count=a.max_points,center=reference['exact_y_free'][row],radius=float(reference['sampling_radius'][row]),seed=a.seed+100_003*motion+1009*time_index,physical=physical,explicit_points=())
                initial[local]=project_fixed_vertices(full_state_from_free_state(points,physical),physical).cpu()
                print(f'motion={motion:03d} time={time_index:03d} points={a.max_points}')
            path.parent.mkdir(parents=True,exist_ok=True); torch.save({'initial_y':initial.contiguous(),'metadata':{'format':'window_shard_v1','motion_index':motion,'time_start':start,'time_stop':stop,'max_points':a.max_points,'physical_xn_included':False,'nested_prefixes':list(counts)}},path)
            records.append({'motion_index':motion,'time_start':start,'time_stop':stop,'path':str(path.resolve()),'reused':False})
    save_json({'format':'window_shards_v1','max_points':a.max_points,'points_per_problem':a.max_points,'sample_counts':list(counts),'motion_indices':motions,'train_time_range':[0,a.train_time_stop-1],'time_window':a.time_window,'physical_xn_included':False,'nested_prefixes':True,'approx_initial_y_storage_gib':len(motions)*a.train_time_stop*a.max_points*FULL_STATE_DIM*8/2**30,'records':records},out/'manifest.json')
    print(out/'manifest.json')
if __name__=='__main__': main()

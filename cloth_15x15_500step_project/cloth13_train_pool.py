"""Metamizer-style live training pool for the 15x15 project.

Pool evolution matches the 5x5 cloth13 implementation, while checkpoint selection
uses this project's offline one-step xn validation set. This keeps the training
method ablation separate from the validation protocol.
"""
from __future__ import annotations
import argparse, importlib.util, math, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from cloth02_dataset_catalog import load_dataset
from cloth03_solvers_and_models import DEFAULT_GRADIENT_CLIP_NORM, DEFAULT_RESIDUAL_LENGTH_SCALE, LEARNING_RATE, MLPOptimizer, ModelSpec, apply_model_update, physical_energy_scale, stationarity_residual_norm_full, variational_energy_full
from cloth_common import evaluate_one_step, load_json, load_physical, save_json, write_csv

_BASE_PATH=Path(__file__).resolve().parent.parent/'cloth_5x5_500step_project'/'cloth13_train_metamizer_pool_models.py'
_spec=importlib.util.spec_from_file_location('_cloth5_pool_shared',_BASE_PATH)
if _spec is None or _spec.loader is None: raise ImportError(_BASE_PATH)
_shared=importlib.util.module_from_spec(_spec); sys.modules[_spec.name]=_shared; _spec.loader.exec_module(_shared)
ClothPool=_shared.ClothPool

def save_ckpt(path,model,optimizer,epoch,updates,spec,best,config):
    torch.save({'epoch':epoch,'update_count':updates,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'model_spec':asdict(spec),'best_validation':best,'config':config},path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('cloth_15x15_500step_pipeline')); p.add_argument('--device',default='cuda:0')
    p.add_argument('--activation',required=True); p.add_argument('--depth',type=int,required=True); p.add_argument('--width',type=int,required=True); p.add_argument('--use-bias',action='store_true')
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--updates-per-epoch',type=int,default=1000); p.add_argument('--k-buckets',type=int,nargs='+',default=[1,3,5,10,30]); p.add_argument('--validation-interval',type=int,default=10); p.add_argument('--evaluation-batch-size',type=int,default=512)
    p.add_argument('--learning-rate',type=float,default=LEARNING_RATE); p.add_argument('--residual-length-scale',type=float,default=DEFAULT_RESIDUAL_LENGTH_SCALE); p.add_argument('--gradient-clip-norm',type=float,default=DEFAULT_GRADIENT_CLIP_NORM); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-energy',type=float,default=1e8); p.add_argument('--max-residual',type=float,default=1e8); p.add_argument('--max-abs-position',type=float,default=1e3); p.add_argument('--min-spring-length',type=float,default=1e-8); p.add_argument('--max-spring-length',type=float,default=1e3); p.add_argument('--max-lifetime-physical-steps',type=int,default=500)
    p.add_argument('--resume',action='store_true'); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
    device=torch.device(a.device); physical=load_physical(a.root); runtime=load_json(a.root/'data'/'reference'/'runtime_config.json'); motions=list(runtime['motions'])
    manifest=load_json(a.root/'data'/'datasets'/'dataset_manifest.json'); train_motions=[int(v) for v in manifest['splits']['train']]; validation=load_dataset('validation_xn',a.root)
    spec=ModelSpec(a.activation,a.depth,a.width,a.use_bias); out=a.root/'experiments'/'training_pool'/'samples_0000'/spec.experiment_name; out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); model=MLPOptimizer(a.residual_length_scale,spec).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=a.learning_rate)
    start,best,updates=1,math.inf,0; logs=[]; history=[]; latest=out/'latest_checkpoint.pt'
    config={'sample_count':0,'training_method':'Metamizer-style live pool','model_spec':asdict(spec),'parameter_count':model.parameter_count,'train_motions':train_motions,'k_buckets':a.k_buckets,'updates_per_epoch':a.updates_per_epoch,'epochs':a.epochs,'loss':'mean physical energy after one learned update / energy scale','validation':'all validation xn problems, exactly one update','checkpoint_metric':'p95(log10(r1/r0))','residual_length_scale':a.residual_length_scale}; save_json(config,out/'config.json')
    if a.resume and latest.exists() and not a.overwrite:
        c=torch.load(latest,map_location=device); model.load_state_dict(c['model_state_dict']); optimizer.load_state_dict(c['optimizer_state_dict']); start=int(c['epoch'])+1; best=float(c.get('best_validation',math.inf)); updates=int(c.get('update_count',0))
    pool=ClothPool(motions=motions,motion_indices=train_motions,k_buckets=a.k_buckets,physical=physical,device=device,args=a); save_json(pool.manifest(),out/'pool_manifest.json'); scale=physical_energy_scale(pool.masses.detach(),physical,a.residual_length_scale)
    for epoch in range(start,a.epochs+1):
        t=time.perf_counter(); losses=[]; residuals=[]; reset_totals={k:0 for k in ['resets_total','resets_nonfinite','resets_energy','resets_residual','resets_position','resets_spring','resets_lifetime']}
        for step in range(a.updates_per_epoch):
            b=pool.ask(); optimizer.zero_grad(set_to_none=True); y,d,r=apply_model_update(model,b['y'],b['q'],b['masses'],physical,previous_residual=b['prev_residual'],previous_update=b['prev_update']); energy=variational_energy_full(y,b['q'],b['masses'],physical); loss=energy.mean()/max(float(scale),1e-30); loss.backward()
            if a.gradient_clip_norm>0: torch.nn.utils.clip_grad_norm_(model.parameters(),a.gradient_clip_norm)
            optimizer.step(); rn=stationarity_residual_norm_full(y,b['q'],b['masses'],physical); stats=pool.tell(y_next=y,delta=d,current_residual=r,energy=energy,residual_norm=rn)
            for k in reset_totals: reset_totals[k]+=int(stats.get(k,0))
            losses.append(float(loss.detach().cpu())); residuals.append(float(rn.mean().detach().cpu())); updates+=1
        row={'epoch':epoch,'update_count':updates,'loss_mean':float(np.mean(losses)),'residual_mean':float(np.mean(residuals)),'elapsed_seconds':time.perf_counter()-t,**reset_totals}; logs.append(row); write_csv(logs,out/'train_log.csv'); save_ckpt(latest,model,optimizer,epoch,updates,spec,best,config)
        if epoch==1 or epoch%a.validation_interval==0 or epoch==a.epochs:
            val=evaluate_one_step(model=model,dataset=validation,physical=physical,device=device,batch_size=a.evaluation_batch_size); record={'epoch':epoch,'update_count':updates,**val['summary']}; history.append(record); save_json({'history':history},out/'validation_metrics.json'); score=float(record['selection_metric'])
            best_path=out/'best_validation_model.pt'
            if (not best_path.exists()) or score<best: best=score; save_ckpt(best_path,model,optimizer,epoch,updates,spec,best,config)
            print(f'pool epoch={epoch}/{a.epochs} loss={row["loss_mean"]:.3e} val_p95_log_ratio={score:.3e} resets={row["resets_total"]}')
    best_ckpt=torch.load(out/'best_validation_model.pt',map_location=device); model.load_state_dict(best_ckpt['model_state_dict']); model.eval()
    test_metrics={}; test_curves={}
    for name in ('test_id_xn','test_ood_xn','test_all_xn'):
        result=evaluate_one_step(model=model,dataset=load_dataset(name,a.root),physical=physical,device=device,batch_size=a.evaluation_batch_size)
        test_metrics[name]=result['summary']; test_curves[name]={'r0':result['residual_before'],'r1':result['residual_after']}
    save_json(test_metrics,out/'test_metrics.json'); torch.save(test_curves,out/'test_curves.pt')
    save_json({'completed':True,'best_validation':best},out/'completed.json')
if __name__=='__main__': main()

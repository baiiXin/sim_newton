"""Thin launcher for the initial-perturbation-count ablation."""
from __future__ import annotations
import argparse
from pathlib import Path
from cloth05_train_models import main as train_main

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('cloth_15x15_500step_pipeline')); p.add_argument('--sample-source-root',type=Path,default=None)
    p.add_argument('--sample-counts',type=int,nargs='+',default=[1,8,32,128,512,1024]); p.add_argument('--activation',required=True); p.add_argument('--depth',type=int,required=True); p.add_argument('--width',type=int,required=True)
    p.add_argument('--bias-mode',choices=('no-bias','with-bias'),required=True); p.add_argument('--sample-chunk-size',type=int,default=0); p.add_argument('--device',default='cuda:0'); p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--validation-interval',type=int,default=50); p.add_argument('--resume',action='store_true'); p.add_argument('--skip-completed',action='store_true'); a=p.parse_args()
    source=a.sample_source_root or a.root/'data'/'initial_point_ablation'/'max_1024'
    for count in a.sample_counts:
        argv=['--root',str(a.root),'--stage','initial_points','--sample-source-root',str(source),'--sample-count',str(count),'--sample-chunk-size',str(a.sample_chunk_size),'--activations',a.activation,'--depths',str(a.depth),'--widths',str(a.width),'--bias-mode',a.bias_mode,'--device',a.device,'--epochs',str(a.epochs),'--validation-interval',str(a.validation_interval)]
        if a.resume: argv.append('--resume')
        if a.skip_completed: argv.append('--skip-completed')
        train_main(argv)
if __name__=='__main__': main()

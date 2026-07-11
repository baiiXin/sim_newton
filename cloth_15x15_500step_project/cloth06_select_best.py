"""Select the best completed configuration in one experiment stage."""
from __future__ import annotations
import argparse
from pathlib import Path
from cloth_common import load_json, save_json, write_csv

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('cloth_15x15_500step_pipeline'))
    p.add_argument('--stage',required=True); p.add_argument('--metric',default='selection_metric'); p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--output',type=Path,default=None); a=p.parse_args()
    rows=[]
    stage_root=a.root/'experiments'/a.stage
    for path in stage_root.glob('samples_*/activation_*/validation_metrics.json'):
        data=load_json(path); history=data.get('history',[])
        if not history: continue
        best=min(history,key=lambda r: float(r.get(a.metric,float('inf'))))
        config=load_json(path.parent/'config.json')
        rows.append({'experiment_dir':str(path.parent),'sample_count':config['sample_count'],**config['model_spec'],'parameter_count':config['parameter_count'],'best_epoch':best['epoch'],a.metric:best[a.metric]})
    if not rows: raise RuntimeError(f'no completed validation histories under {stage_root}')
    rows.sort(key=lambda r:(float(r[a.metric]),int(r['parameter_count'])))
    selection={'stage':a.stage,'metric':a.metric,'best':rows[0],'top':rows[:a.top_k]}
    out=a.output or stage_root/'selection.json'; save_json(selection,out); write_csv(rows,stage_root/'ranking.csv')
    print(f"best: {rows[0]['experiment_dir']} {a.metric}={rows[0][a.metric]:.6e}")
    print(f"width={rows[0]['width']} depth={rows[0]['depth']} activation={rows[0]['activation']} bias={rows[0]['use_bias']}")
if __name__=='__main__': main()

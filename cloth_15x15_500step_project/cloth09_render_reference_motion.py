"""Render one stored 15x15 reference motion selected by motion index."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import torch
from cloth_common import load_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('cloth_15x15_500step_pipeline'))
    p.add_argument('--motion-index',type=int,required=True); p.add_argument('--stride',type=int,default=1); p.add_argument('--fps',type=int,default=30)
    p.add_argument('--format',choices=('mp4','gif'),default='mp4'); p.add_argument('--output',type=Path,default=None); a=p.parse_args()
    runtime=load_json(a.root/'data'/'reference'/'runtime_config.json'); states=torch.load(a.root/'data'/'reference'/'reference_motion_states.pt',map_location='cpu')
    ids=[int(v) for v in states['motion_index'].tolist()]; row=ids.index(a.motion_index)
    pos=states['positions'][row,::a.stride].numpy(); residual=states['exact_residual'][row,::a.stride].numpy()
    edges=[tuple(e) for e in runtime['spring_edges']]; fixed=runtime['fixed_vertex_indices']
    mins=pos.reshape(-1,3).min(0); maxs=pos.reshape(-1,3).max(0); center=(mins+maxs)/2; radius=max(float((maxs-mins).max())*.58,1e-3)
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d')
    lines=[ax.plot([],[],[],linewidth=.35)[0] for _ in edges]; points=ax.scatter([],[],[],s=5); pins=ax.scatter([],[],[],s=45,marker='s')
    ax.set_xlim(center[0]-radius,center[0]+radius); ax.set_ylim(center[1]-radius,center[1]+radius); ax.set_zlim(center[2]-radius,center[2]+radius)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z'); ax.view_init(elev=22,azim=-60)
    def update(frame):
        x=pos[frame]
        for line,(i,j) in zip(lines,edges): line.set_data_3d([x[i,0],x[j,0]],[x[i,1],x[j,1]],[x[i,2],x[j,2]])
        points._offsets3d=(x[:,0],x[:,1],x[:,2]); pins._offsets3d=(x[fixed,0],x[fixed,1],x[fixed,2])
        r=residual[min(frame,len(residual)-1)]; ax.set_title(f'motion {a.motion_index:03d} frame {frame*a.stride:03d} reference residual={r:.3e}')
        return [*lines,points,pins]
    anim=FuncAnimation(fig,update,frames=len(pos),interval=1000/a.fps,blit=False)
    out=a.output or a.root/'renders'/'reference'/f'motion_{a.motion_index:03d}.{a.format}'; out.parent.mkdir(parents=True,exist_ok=True)
    writer=FFMpegWriter(fps=a.fps) if a.format=='mp4' else PillowWriter(fps=a.fps); anim.save(out,writer=writer,dpi=140); plt.close(fig); print(out)
if __name__=='__main__': main()

import subprocess, sys
U='/root/.claude/uploads/8a3fc039-b934-5eda-b2a3-ab541c6142cc/'
SRC={'A':U+'9bc1edea-VID-20261029-WA0001.mp4','B':U+'b9076259-VID-20261029-WA0000.mp4'}
G={'A':"eq=contrast=1.10:brightness=0.01:saturation=1.0:gamma=0.97,colorbalance=rs=-0.02:bs=0.03:rh=0.02:bh=-0.02,unsharp=5:5:0.6",
   'B':"eq=contrast=1.12:brightness=-0.015:saturation=1.08:gamma=0.95,colorbalance=bs=0.04:gs=0.01,unsharp=5:5:0.6"}
def run(c): subprocess.run(['ffmpeg','-nostdin','-v','error','-y']+c,check=True)
only=sys.argv[1:]
for line in open('segs.txt'):
    n,s,st,d,sp=line.split(); st,d,sp=float(st),float(d),float(sp)
    if only and n not in only: continue
    f=SRC[s]; ss=max(0,st-1); off=st-ss; D=d+2
    run(['-ss',str(ss),'-t',str(D),'-i',f,'-vf',f'vidstabdetect=shakiness=6:accuracy=12:result=seg/{n}.trf','-f','null','-'])
    sl='fps=30' if sp==1 else f'minterpolate=fps=50:mi_mode=mci:mc_mode=aobmc:vsbmc=1,setpts=PTS/{sp},fps=30'
    run(['-ss',str(ss),'-t',str(D),'-i',f,'-vf',f'vidstabtransform=input=seg/{n}.trf:smoothing=20:zoom=4:optzoom=0,{G[s]},trim=start={off}:duration={d},setpts=PTS-STARTPTS,{sl}','-an','-c:v','libx264','-crf','10','-preset','fast',f'seg/{n}.mp4'])
    run(['-ss',str(st),'-t',str(d),'-i',f,'-vn','-ac','2','-ar','48000',f'seg/{n}.wav'])
    r=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',f'seg/{n}.mp4'],capture_output=True,text=True)
    print(n,r.stdout.strip(),flush=True)

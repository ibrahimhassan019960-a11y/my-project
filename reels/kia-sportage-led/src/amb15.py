import numpy as np, wave
SR=48000; N=SR*15; out=np.zeros((N,2))
plan=[('s1',0,45,.10),('s3',45,60,.16),('s4',105,45,.16),('s6b',150,75,.10),('s7',225,60,.10),('s11',345,105,.10)]
_=[('s1',0,60,.10),('s2',60,60,.10),('s3',120,90,.16),('s4',210,60,.16),('s5',270,60,.16),('s6b',375,90,.10),('s7',465,90,.10),('s9',675,90,.12),('s11',765,135,.10)]
for n,f0,nf,g in plan:
    w=wave.open(f'seg/{n}.wav'); a=np.frombuffer(w.readframes(w.getnframes()),np.int16).reshape(-1,2)/32768
    L=min(len(a),int(nf/30*SR)); i=int(f0/30*SR)
    e=np.ones(L); r=int(0.04*SR); e[:r]=np.linspace(0,1,r); e[-r:]=np.linspace(1,0,r)
    out[i:i+L]+=a[:L]*e[:,None]*g
w=wave.open('amb15.wav','wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(out,-1,1)*32767).astype(np.int16).tobytes())

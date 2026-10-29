import cv2, numpy as np, json, sys
def detect(f,ymin=0,hmin=18):
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    v=hsv[...,2]; s=hsv[...,1]; h=hsv[...,0]
    m=((v>150)&((s<90)|((h>80)&(h<115)))).astype(np.uint8)*255
    m[:int(f.shape[0]*0.35)]=0
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,15),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((9,9),np.uint8))
    n,l,st,_=cv2.connectedComponentsWithStats(m)
    best=None
    for i in range(1,n):
        x,y,w,hh,a=st[i]
        if y<ymin: continue
        if w<50 or hh<hmin or w/hh<1.4 or w/hh>5.5: continue
        if a/(w*hh)<0.45: continue
        if best is None or a>best[4]: best=(x,y,w,hh,a)
    return best
out={}
for n in sys.argv[1:]:
    c=cv2.VideoCapture(f'seg/{n}.mp4'); boxes=[]
    while True:
        ok,f=c.read()
        if not ok: break
        boxes.append(detect(f,330,4) if n=='s8' else detect(f))
    # fill gaps & smooth
    idx=[i for i,b in enumerate(boxes) if b]
    print(n,len(boxes),'detected',len(idx))
    arr=np.array([[*b[:4]] if b else [np.nan]*4 for b in boxes],float)
    for k in range(4):
        col=arr[:,k]; good=~np.isnan(col)
        if good.any(): arr[:,k]=np.interp(np.arange(len(col)),np.where(good)[0],col[good])
    ker=np.ones(5)/5
    sm=np.array([np.convolve(np.pad(arr[:,k],2,mode='edge'),ker,'valid') for k in range(4)]).T
    out[n]=sm.tolist()
old=json.load(open('plates.json')); old.update(out); json.dump(old,open('plates.json','w'))

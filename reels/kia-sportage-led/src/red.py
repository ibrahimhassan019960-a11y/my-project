import cv2, numpy as np, json, sys
out={}
for n in ['s1','s2','s6b','s7','s8','s9','s11']:
    c=cv2.VideoCapture(f'seg/{n}.mp4'); r=[]
    while True:
        ok,f=c.read()
        if not ok: break
        hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
        m=((hsv[...,0]<8)|(hsv[...,0]>170))&(hsv[...,1]>120)&(hsv[...,2]>120)
        ys,xs=np.nonzero(m)
        if len(xs)<50: r.append(None); continue
        r.append([int(np.percentile(xs,1)),int(np.percentile(xs,99)),int(np.median(ys)),int(np.percentile(ys,5)),int(np.percentile(ys,95))])
    out[n]=r
    print(n,[r[i] for i in (0,len(r)//2,len(r)-1)])
json.dump(out,open('red.json','w'))

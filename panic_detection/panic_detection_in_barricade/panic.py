import cv2
import numpy as np
from ultralytics import YOLO
from alarm import beep,fast_beep,siren,stop_alarm

detector=YOLO("yolov8n.pt")
# detector=YOLO("yolov8s_1_finetuned.pt")
# detector=YOLO("yolov8n.onnx")

#video=cv2.VideoCapture("test4.mp4")
# video=cv2.VideoCapture("new_test.mp4")
# video=cv2.VideoCapture("last_test.mp4")
video=cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok,fr=video.read()
if not ok:print("Error");exit()
prev=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
H,W=fr.shape[:2]

FLOW={'pyr_scale':0.6,'levels':5,'winsize':10,'iterations':5,'poly_n':5,'poly_sigma':1.1,'flags':0}
STEP=8;MIN=0.12;REST=0.20
MID=W//2
EXP={'LEFT':180,'RIGHT':0}
TOL=30
SM=0.05
flt=100.0

def match(a,t):
    d=abs(a-t)
    d=min(d,360-d)
    return d<=TOL

cv2.namedWindow("Crowd",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Crowd",1200,700)
while True:
    ok,fr=video.read()
    if not ok:break
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    flow=cv2.calcOpticalFlowFarneback(prev,g,None,**FLOW)
    r=detector(fr,classes=[0],conf=0.20,verbose=False)

    boxes=[]
    for rr in r:
        for b in rr.boxes:
            x1,y1,x2,y2=map(int,b.xyxy[0])
            if(x2-x1)>=12 and(y2-y1)>=18:boxes.append((x1,y1,x2,y2))

    tot=len(boxes);cor=0

    for x1,y1,x2,y2 in boxes:
        cx=(x1+x2)//2;cy=(y1+y2)//2
        side="RIGHT"if cx>MID else"LEFT"
        tgt=EXP[side]
        dx=dy=0.0;c=0

        for yy in range(y1,y2,STEP):
            for xx in range(x1,x2,STEP):
                fx,fy=flow[yy,xx]
                if np.hypot(fx,fy)>=MIN:
                    dx+=fx;dy+=fy;c+=1

        if c==0:
            cor+=1;cv2.rectangle(fr,(x1,y1),(x2,y2),(0,255,0),2);continue

        ax=dx/c;ay=dy/c
        m=np.hypot(ax,ay)

        if m<REST:
            cor+=1;cv2.rectangle(fr,(x1,y1),(x2,y2),(0,255,0),2);continue

        ang = np.degrees(np.arctan2(ax,-ay))
        ang = (ang + 360) % 360
        good=match(ang,tgt)
        clr=(255,255,255)if good else(0,0,255)
        if good:cor+=1

        cv2.rectangle(fr,(x1,y1),(x2,y2),clr,2)

        ex=int(cx+ax*12)
        ey=int(cy+ay*12)
        cv2.arrowedLine(fr,(cx,cy),(ex,ey),clr,2,tipLength=0.4)

    pct=(cor/tot*100)if tot>0 else 100
    flt=SM*pct+(1-SM)*flt

    if flt>75:
        txt=f"Normal:{flt:.1f}%";col=(255,255,255)
        stop_alarm()
    elif flt>60:
        txt=f"Warning:{flt:.1f}%";col=(0,255,255)
        fast_beep()   
    else:
        txt=f"PANIC:{flt:.1f}%";col=(0,0,255)
        siren()

    cv2.putText(fr,txt,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,3)
    cv2.line(fr,(MID,0),(MID,H),(255,255,255),2)
    cv2.imshow("Crowd",fr)

    if cv2.waitKey(1)&0xFF==ord('q'):break
    prev=g.copy()

video.release()
cv2.destroyAllWindows()
stop_alarm()

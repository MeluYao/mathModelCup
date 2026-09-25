# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent;TAB=ROOT/'q3_results'/'tables';FIG=ROOT/'q3_results'/'figures';FIG.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(TAB/'transport_plan.csv');r=pd.read_csv(TAB/'relay_plan.csv');c=pd.read_csv(TAB/'communication_summary.csv');obj=pd.read_csv(TAB/'objective_comparison.csv');nodes=pd.read_csv(ROOT/'preprocessing'/'tables'/'nodes.csv').set_index('node_id')
F=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);COL={'A':'#2878B5','B':'#F28E2B','C':'#59A14F'}
def canvas(title,w=1800,h=1100):im=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(im);d.text((55,30),title,font=F(36),fill='#202124');return im,d
def axes(d,x0,y0,x1,y1,xlab='',ylab=''):d.line((x0,y1,x1,y1),fill='#444',width=3);d.line((x0,y0,x0,y1),fill='#444',width=3);d.text((x1-210,y1+20),xlab,font=F(21),fill='#444');d.text((x0+5,y0-32),ylab,font=F(21),fill='#444')

# 联合空间方案
im,d=canvas('问题三运输路线与中继悬停点',1800,1350);x0,y0,x1,y1=120,120,1690,1230;axes(d,x0,y0,x1,y1,'东向距离 / km','北向距离 / km')
allx=list(nodes.x_m)+list(r.x_m);ally=list(nodes.y_m)+list(r.y_m);xmin,xmax=min(allx)/1000,max(allx)/1000;ymin,ymax=min(ally)/1000,max(ally)/1000
px=lambda x:x0+(x-xmin)/(xmax-xmin)*(x1-x0);py=lambda y:y1-(y-ymin)/(ymax-ymin)*(y1-y0)
for z in p.itertuples():
    seq=['O01']+z.visit_order.split('-')+['O01'];pts=[(px(nodes.loc[s,'x_m']/1000),py(nodes.loc[s,'y_m']/1000)) for s in seq];d.line(pts,fill=COL[z.type_id],width=3)
for z in r.drop_duplicates('hover_site_id').itertuples():
    a=(px(0),py(0));b=(px(z.x_m/1000),py(z.y_m/1000));d.line((a,b),fill='#9467BD',width=4);x,y=b;d.polygon([(x,y-13),(x-12,y+10),(x+12,y+10)],fill='#9467BD');d.text((x+13,y-16),f'{z.hover_site_id}\n{z.hover_agl_m:.0f}m AGL',font=F(17),fill='#6A3D9A')
for s,z in nodes.iterrows():
    x,y=px(z.x_m/1000),py(z.y_m/1000);rad=10 if s=='O01' else 6;d.ellipse((x-rad,y-rad,x+rad,y+rad),fill='#D62728' if s=='O01' else '#222');d.text((x+8,y-10),s,font=F(16),fill='#222')
im.save(FIG/'01_联合空间方案.png')

# 两类无人机甘特图
im,d=canvas('运输与中继无人机联合甘特图',1900,1350);x0,y0,x1,y1=240,120,1800,1230;axes(d,x0,y0,x1,y1,'任务开始后的分钟','')
ids=[f'U{i:02d}' for i in range(1,9)]+['R01','R02'];tmax=max(p.return_s.max(),r.return_s.max())/60
for i,a in enumerate(ids):
    y=y0+(i+.5)*(y1-y0)/len(ids);d.text((80,y-14),a,font=F(23),fill='#222')
    if a.startswith('U'):
        for z in p[p.aircraft_id.eq(a)].itertuples():
            xa=x0+z.start_s/60/tmax*(x1-x0);xb=x0+z.return_s/60/tmax*(x1-x0);d.rounded_rectangle((xa,y-20,xb,y+20),7,fill=COL[z.type_id]);d.text((xa+3,y-12),z.flight_id[-3:],font=F(14),fill='white')
    else:
        for z in r[r.relay_id.eq(a)].itertuples():
            xa=x0+z.start_s/60/tmax*(x1-x0);xb=x0+z.return_s/60/tmax*(x1-x0);xs=x0+z.service_start_s/60/tmax*(x1-x0);xe=x0+z.service_end_s/60/tmax*(x1-x0)
            d.rectangle((xa,y-18,xb,y+18),fill='#C7B5D9');d.rectangle((xs,y-18,xe,y+18),fill='#7B52AB');d.text((xs+3,y-12),z.relay_flight_id[-3:],font=F(14),fill='white')
for k in range(8):x=x0+k*(x1-x0)/7;d.line((x,y0,x,y1),fill='#E6E6E6');d.text((x-15,y1+15),f'{tmax*k/7:.0f}',font=F(17),fill='#555')
im.save(FIG/'02_联合甘特图.png')

# 通信构成
im,d=canvas('各运输架次通信方式构成',1800,1100);x0,y0,x1,y1=230,110,1700,980;axes(d,x0,y0,x1,y1,'通信采样比例','')
for i,z in c.iterrows():
    y=y0+(i+.5)*(y1-y0)/len(c);d.text((55,y-10),z.transport_flight_id,font=F(15),fill='#333');w=x1-x0;direct=z.direct_fraction
    d.rectangle((x0,y-12,x0+w*direct,y+12),fill='#4E79A7');d.rectangle((x0+w*direct,y-12,x1,y+12),fill='#E15759')
d.rectangle((1250,55,1275,80),fill='#4E79A7');d.text((1282,50),'直连',font=F(18),fill='#333');d.rectangle((1380,55,1405,80),fill='#E15759');d.text((1412,50),'中继',font=F(18),fill='#333')
im.save(FIG/'03_通信构成.png')

# 可行候选权衡
v=obj[(obj.hard_violations==0)&(~obj.policy.astype(str).str.startswith('balanced_refined'))].copy();v=pd.concat([v,obj[obj.policy.astype(str).str.startswith('balanced_refined')].tail(1)],ignore_index=True)
im,d=canvas('硬时限可行联合方案权衡',1800,950);x0,y0,x1,y1=180,150,1700,810;axes(d,x0,y0,x1,y1,'中继架次数','联合完成时间 / h')
xmin,xmax=v.relay_trips.min()-0.5,v.relay_trips.max()+0.5;ymin,ymax=v.joint_makespan_s.min()/3600-.05,v.joint_makespan_s.max()/3600+.05
for z in v.itertuples():
    x=x0+(z.relay_trips-xmin)/(xmax-xmin)*(x1-x0);y=y1-(z.joint_makespan_s/3600-ymin)/(ymax-ymin)*(y1-y0);rad=12+int((z.total_energy_kwh-v.total_energy_kwh.min())*8);d.ellipse((x-rad,y-rad,x+rad,y+rad),fill='#2A9D8F');d.text((x+15,y-12),f'{z.policy}/{z.merge_gap_s:.0f}s\n{z.total_energy_kwh:.2f}kWh',font=F(17),fill='#333')
im.save(FIG/'04_目标权衡.png')
print('generated',len(list(FIG.glob('*.png'))),'figures')

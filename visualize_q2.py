"""用 Pillow 生成问题二路线、资源、交付和目标权衡图。"""
from pathlib import Path
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent; TAB=ROOT/'q2_results'/'tables'; FIG=ROOT/'q2_results'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
trips=pd.read_csv(TAB/'plan_balanced.csv'); deliveries=pd.read_csv(TAB/'deliveries_balanced.csv')
nodes=pd.read_csv(ROOT/'preprocessing'/'tables'/'nodes.csv').set_index('node_id'); comp=pd.read_csv(TAB/'objective_comparison.csv')
font_path='C:/Windows/Fonts/msyh.ttc'
F=lambda n:ImageFont.truetype(font_path,n)
COL={'A':'#2878B5','B':'#F28E2B','C':'#59A14F'}

def canvas(title,w=1800,h=1100):
    im=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(im);d.text((60,35),title,font=F(36),fill='#202124');return im,d
def axes(d,x0,y0,x1,y1,xlab='',ylab=''):
    d.line((x0,y1,x1,y1),fill='#444',width=3);d.line((x0,y0,x0,y1),fill='#444',width=3)
    d.text((x1-180,y1+25),xlab,font=F(22),fill='#444');d.text((x0+10,y0-35),ylab,font=F(22),fill='#444')

im,d=canvas('问题二主方案运输路线（22架次）',1800,1350);x0,y0,x1,y1=130,120,1680,1230;axes(d,x0,y0,x1,y1,'东向距离/km','北向距离/km')
xmin,xmax=nodes.x_m.min()/1000,nodes.x_m.max()/1000;ymin,ymax=nodes.y_m.min()/1000,nodes.y_m.max()/1000
px=lambda x:x0+(x-xmin)/(xmax-xmin)*(x1-x0);py=lambda y:y1-(y-ymin)/(ymax-ymin)*(y1-y0)
for r in trips.itertuples():
    seq=['O01']+r.visit_order.split('-')+['O01']; pts=[(px(nodes.loc[s,'x_m']/1000),py(nodes.loc[s,'y_m']/1000)) for s in seq]
    d.line(pts,fill=COL[r.type_id],width=3)
for s,r in nodes.iterrows():
    x,y=px(r.x_m/1000),py(r.y_m/1000);rad=11 if s=='O01' else 7;d.ellipse((x-rad,y-rad,x+rad,y+rad),fill='#D62728' if s=='O01' else '#222');d.text((x+10,y-12),s,font=F(18),fill='#222')
for i,g in enumerate('ABC'):d.line((130+i*180,95,175+i*180,95),fill=COL[g],width=6);d.text((185+i*180,78),f'{g}型',font=F(20),fill='#333')
im.save(FIG/'01_主方案路线.png')

im,d=canvas('实体无人机占用甘特图',1800,1000);x0,y0,x1,y1=220,120,1700,880;axes(d,x0,y0,x1,y1,'任务开始后的分钟','')
air=sorted(trips.aircraft_id.unique()); tmax=trips.return_s.max()/60
for i,a in enumerate(air):
    y=y0+(i+.5)*(y1-y0)/len(air);d.text((80,y-15),a,font=F(24),fill='#222')
    for r in trips[trips.aircraft_id.eq(a)].itertuples():
        xa=x0+r.start_s/60/tmax*(x1-x0);xb=x0+r.return_s/60/tmax*(x1-x0);d.rounded_rectangle((xa,y-23,xb,y+23),8,fill=COL[r.type_id]);d.text((xa+5,y-14),r.flight_id[-3:],font=F(16),fill='white')
for k in range(7):
    x=x0+k*(x1-x0)/6;d.line((x,y0,x,y1),fill='#E5E5E5',width=1);d.text((x-20,y1+18),f'{tmax*k/6:.0f}',font=F(18),fill='#555')
im.save(FIG/'02_无人机甘特图.png')

im,d=canvas('电池飞行占用与充电安排（彩色：飞行，灰色：充电）',1800,1350);x0,y0,x1,y1=250,120,1700,1230;axes(d,x0,y0,x1,y1,'任务开始后的分钟','')
bats=sorted(trips.battery_id.unique()); tmax=trips.battery_recharged_s.max()/60
for i,b in enumerate(bats):
    y=y0+(i+.5)*(y1-y0)/len(bats);d.text((45,y-12),b,font=F(18),fill='#222')
    for r in trips[trips.battery_id.eq(b)].itertuples():
        xa=x0+r.start_s/60/tmax*(x1-x0);xb=x0+r.return_s/60/tmax*(x1-x0);xc=x0+r.battery_recharged_s/60/tmax*(x1-x0)
        d.rectangle((xa,y-16,xb,y+16),fill=COL[r.type_id]);d.rectangle((xb,y-16,xc,y+16),fill='#BDBDBD')
im.save(FIG/'03_电池周转.png')

deli=deliveries.sort_values(['expected_s','delivery_s']).reset_index(drop=True); im,d=canvas('逐箱送达时刻',1800,950);x0,y0,x1,y1=150,120,1700,820;axes(d,x0,y0,x1,y1,'按期望时间排序的货箱','分钟')
tmax=max(deli.delivery_s.max(),deli.expected_s.max())/60
for i,r in deli.iterrows():
    x=x0+i/(len(deli)-1)*(x1-x0); yy=y1-r.delivery_s/60/tmax*(y1-y0); ye=y1-r.expected_s/60/tmax*(y1-y0)
    d.line((x-5,ye,x+5,ye),fill='#222',width=2);d.ellipse((x-4,yy-4,x+4,yy+4),fill='#D62728' if r.delivery_s>r.expected_s+1e-7 else '#2CA02C')
on=(deli.delivery_s<=deli.expected_s+1e-7).sum();d.text((1250,75),f'按期 {on} 箱，超期 {len(deli)-on} 箱',font=F(24),fill='#333');im.save(FIG/'04_逐箱交付.png')

metrics=['weighted_tardiness','weighted_delivery_ratio','makespan_s','energy_kwh','trips'];labels=['加权迟到','交付比','最晚返航','总能耗','架次数']
u=comp.drop_duplicates(subset=metrics).reset_index(drop=True); lo=u[metrics].min(); hi=u[metrics].max(); norm=(u[metrics]-lo)/(hi-lo).replace(0,1)
im,d=canvas('代表方案目标权衡（候选范围归一化，越低越好）',1800,950);x0,y0,x1,y1=170,140,1700,800;axes(d,x0,y0,x1,y1,'','归一化值')
palette=['#4E79A7','#E15759','#59A14F','#F28E2B']; group=(x1-x0)/len(metrics)
for j,m in enumerate(metrics):
    for i,r in u.iterrows():
        bw=group/(len(u)+1);x=x0+j*group+(i+.5)*bw;h=float(norm.loc[i,m])*(y1-y0);d.rectangle((x,y1-h,x+bw*.8,y1),fill=palette[i%len(palette)])
    d.text((x0+j*group+15,y1+18),labels[j],font=F(22),fill='#333')
for i,r in u.iterrows(): d.rectangle((1150+i*150,85,1175+i*150,110),fill=palette[i%len(palette)]);d.text((1180+i*150,80),r.policy,font=F(18),fill='#333')
im.save(FIG/'05_目标权衡.png')
print(f'generated {len(list(FIG.glob("*.png")))} figures')

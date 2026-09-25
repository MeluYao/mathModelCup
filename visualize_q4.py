# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from PIL import Image,ImageDraw,ImageFont

R=Path(__file__).resolve().parent;O=R/'q4_results'/'figures';O.mkdir(parents=True,exist_ok=True)
FONT='C:/Windows/Fonts/msyh.ttc';BOLD='C:/Windows/Fonts/msyhbd.ttc'
def f(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
COL=['#2F6BFF','#F29E38','#19A974'];DARK='#1B2533';GRID='#DDE4EE';RED='#D9485F'

def partition_map():
    a=pd.read_csv(R/'q4_results/tables/partition_assignments.csv');n=pd.read_csv(R/'preprocessing/tables/nodes.csv').set_index('node_id')
    im=Image.new('RGB',(1800,950),'white');d=ImageDraw.Draw(im);d.text((70,30),'问题四：2组与3组任务分区',font=f(40,True),fill=DARK)
    for p,(sc,x0) in enumerate([('2组',60),('3组',930)]):
        x1=x0+810;y0=120;y1=850;z=a[a.scenario.eq(sc)];pts=n.loc[z.service_id]
        xmin,xmax=pts.x_m.min(),pts.x_m.max();ymin,ymax=pts.y_m.min(),pts.y_m.max();pad=.08
        sx=lambda x:x0+60+(x-xmin)/(xmax-xmin)*(690)
        sy=lambda y:y1-60-(y-ymin)/(ymax-ymin)*(610)
        d.rounded_rectangle((x0,y0,x1,y1),radius=22,outline=GRID,width=3,fill='#FAFCFF');d.text((x0+25,y0+20),sc,font=f(30,True),fill=DARK)
        # O01仅作方位参照
        ox,oy=n.loc['O01',['x_m','y_m']];px=max(x0+55,min(x1-55,sx(ox)));py=max(y0+75,min(y1-50,sy(oy)));d.regular_polygon((px,py,15),4,rotation=45,fill=DARK);d.text((px+18,py-15),'O01',font=f(20),fill=DARK)
        # 不可拆原多点架次
        q=n.loc[['S006','S007']];d.line((sx(q.iloc[0].x_m),sy(q.iloc[0].y_m),sx(q.iloc[1].x_m),sy(q.iloc[1].y_m)),fill='#7A8798',width=5)
        for row in z.itertuples():
            gid=int(row.group_id[1:]);q=n.loc[row.service_id];cx,cy=sx(q.x_m),sy(q.y_m)
            d.ellipse((cx-17,cy-17,cx+17,cy+17),fill=COL[gid-1],outline='white',width=3);d.text((cx+20,cy-14),row.service_id,font=f(18,True),fill=DARK)
        for gid in sorted(z.group_id.unique()):
            ss='、'.join(z[z.group_id.eq(gid)].service_id);d.text((x0+30,y1-45-28*(int(gid[1:])-1)),f'{gid}: {ss}',font=f(15),fill=COL[int(gid[1:])-1])
    im.save(O/'01_任务分区.png')

def resources():
    r=pd.read_csv(R/'q4_results/tables/resource_comparison.csv');names=list(r[r.scenario.eq('2组')].resource);im=Image.new('RGB',(1800,1000),'white');d=ImageDraw.Draw(im);d.text((70,30),'资源规模、库存与缺口',font=f(40,True),fill=DARK)
    base=r[r.scenario.eq('2组')].centralized_required.to_list();stock=r[r.scenario.eq('2组')].existing_stock.to_list();v2=r[r.scenario.eq('2组')].partition_total_required.to_list();v3=r[r.scenario.eq('3组')].partition_total_required.to_list();mx=max(stock+v2+v3)+1
    left,top,bottom=110,150,850;scale=(bottom-top)/mx
    for y in range(mx+1):yy=bottom-y*scale;d.line((left,yy,1740,yy),fill=GRID,width=1);d.text((70,yy-12),str(y),font=f(18),fill='#687386')
    bw=38;step=195
    for i,name in enumerate(names):
        cx=left+75+i*step
        for j,(val,c) in enumerate(zip([base[i],v2[i],v3[i]],['#93A4B8',COL[0],COL[1]])):
            x=cx+(j-1)*45;d.rectangle((x-bw//2,bottom-val*scale,x+bw//2,bottom),fill=c);d.text((x-8,bottom-val*scale-28),str(val),font=f(18,True),fill=c)
        sy=bottom-stock[i]*scale;d.line((cx-72,sy,cx+72,sy),fill=RED,width=4);d.text((cx-55,bottom+20),name,font=f(17),fill=DARK)
    d.text((110,910),'灰：集中执行下界    蓝：2组    橙：3组    红线：现有库存',font=f(24),fill=DARK)
    im.save(O/'02_资源规模与缺口.png')

def workload():
    g=pd.read_csv(R/'q4_results/tables/group_summary.csv');metrics=[('boxes','货箱数'),('transport_trips','运输架次'),('transport_energy_kwh','运输能耗'),('transport_busy_s','运输忙时'),('relay_task_copies','中继任务副本'),('relay_service_s','中继服务时长')]
    im=Image.new('RGB',(1800,1000),'white');d=ImageDraw.Draw(im);d.text((70,30),'组间工作量（相对本方案组均值）',font=f(40,True),fill=DARK)
    for pi,(sc,x0) in enumerate([('2组',80),('3组',930)]):
        z=g[g.scenario.eq(sc)].reset_index(drop=True);d.text((x0,110),sc,font=f(30,True),fill=DARK);basey=860
        for m,(col,label) in enumerate(metrics):
            x=x0+35+m*125;d.text((x,880),label,font=f(16),fill=DARK)
            mean=z[col].mean()
            for j,row in z.iterrows():
                val=row[col]/mean;h=min(val,1.7)*390;xx=x+j*30
                d.rectangle((xx,basey-h,xx+24,basey),fill=COL[j]);d.text((xx-3,basey-h-25),f'{val:.2f}',font=f(14),fill=COL[j])
        for yv in [0.5,1,1.5]:
            yy=basey-yv*390;d.line((x0,yy,x0+780,yy),fill=GRID,width=2);d.text((x0-45,yy-12),f'{yv:.1f}',font=f(16),fill='#687386')
        legend='  '.join(f'G{i+1}' for i in range(len(z)));d.text((x0,950),legend,font=f(22),fill=DARK)
    im.save(O/'03_工作量均衡.png')

if __name__=='__main__':partition_map();resources();workload();print('created 3 figures')

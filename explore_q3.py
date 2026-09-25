from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; SRC=ROOT/'preprocessing'/'tables'
nodes=pd.read_csv(SRC/'nodes.csv').set_index('node_id'); arcs=pd.read_csv(SRC/'arcs.csv').set_index(['from_id','to_id'])
types=pd.read_csv(SRC/'transport_types.csv').set_index('type_id')
z=np.load(ROOT/'preprocessing'/'spatial'/'dem.npz'); dem=z['elevation_m']; lons=z['lon_deg']; lats=z['lat_deg']
proj=__import__('json').load(open(ROOT/'preprocessing'/'spatial'/'projection.json',encoding='utf-8'))
lon0=proj['lon0'];lat0=proj['lat0'];ex=proj['east_m_per_degree'];ny=proj['north_m_per_degree']
f=2400.;obs=10.

def elev(x,y):
    lon=lon0+x/ex;lat=lat0+y/ny
    c=int(np.clip(np.searchsorted(lons,lon),1,len(lons)-1));
    # latitude array is descending
    r=int(np.clip(np.searchsorted(-lats,-lat),1,len(lats)-1))
    c0=c-1 if abs(lons[c-1]-lon)<abs(lons[c]-lon) else c
    r0=r-1 if abs(lats[r-1]-lat)<abs(lats[r]-lat) else r
    return float(dem[r0,c0])
def link(a,b,limit):
    x1,y1,h1=a;x2,y2,h2=b;d3=math.sqrt((x2-x1)**2+(y2-y1)**2+(h2-h1)**2)/1000
    n=max(3,int(math.hypot(x2-x1,y2-y1)/120)+2);tt=np.linspace(0,1,n)
    blocked=False
    for u in tt[1:-1]:
        x=x1+(x2-x1)*u;y=y1+(y2-y1)*u;h=h1+(h2-h1)*u
        if elev(x,y)>h-1e-6:blocked=True;break
    loss=32.45+20*math.log10(f)+20*math.log10(max(d3,1e-6))+(obs if blocked else 0)
    return loss<=limit+1e-9,limit-loss,blocked
gateway=(0.,0.,float(nodes.loc['O01','ground_m'])+20)

def points(planrow):
    order=planrow.visit_order.split('-'); seq=['O01']+order+['O01']; pts=[]
    for i,j in zip(seq[:-1],seq[1:]):
        a=arcs.loc[(i,j)]; ni=nodes.loc[i];nj=nodes.loc[j]
        oi=float(ni.ground_m+(0 if i=='O01' else 30));oj=float(nj.ground_m+(0 if j=='O01' else 30));cr=float(a.cruise_alt_m)
        pts.extend([(ni.x_m,ni.y_m,oi),(ni.x_m,ni.y_m,cr)])
        n=max(3,int(a.distance_m/150)+2)
        for u in np.linspace(0,1,n):pts.append((ni.x_m+(nj.x_m-ni.x_m)*u,ni.y_m+(nj.y_m-ni.y_m)*u,cr))
        pts.extend([(nj.x_m,nj.y_m,cr),(nj.x_m,nj.y_m,oj)])
    return pts

raw=[]
for s,r in nodes.iterrows():
    if s=='O01':continue
    for frac in [.55,.8,1.0]:
        x=float(r.x_m*frac);y=float(r.y_m*frac);raw.append((f'{s}-f{frac:.2f}',x,y,elev(x,y)+300))
# add central ring/grid-like means
for a,b in [('S002','S004'),('S003','S015'),('S012','S014'),('S006','S007'),('S008','S005')]:
    x=float((nodes.loc[a,'x_m']+nodes.loc[b,'x_m'])/2);y=float((nodes.loc[a,'y_m']+nodes.loc[b,'y_m'])/2);raw.append((f'{a}-{b}',x,y,elev(x,y)+300))
candidates=[]
for name,x,y,h in raw:
    ok,margin,_=link((x,y,h),gateway,126)
    if ok:candidates.append((name,x,y,h,margin))
print('candidates',len(candidates))
for file in ['plan_balanced.csv','plan_timely.csv']:
    plan=pd.read_csv(ROOT/'q2_results'/'tables'/file)
    cover=[]
    for p in plan.itertuples():
        need=[q for q in points(p) if not link(q,gateway,122)[0]]
        good=[]
        for ci,c in enumerate(candidates):
            if all(link(q,c[1:4],116)[0] for q in need):good.append(ci)
        cover.append(good)
        print(file,p.flight_id,p.visit_order,'need',len(need),'cands',len(good))
    # greedy set cover
    remain=set(i for i,g in enumerate(cover) if g);selected=[]
    while remain:
        best=max(range(len(candidates)),key=lambda c:sum(c in cover[i] for i in remain))
        hit={i for i in remain if best in cover[i]};
        if not hit:break
        selected.append(best);remain-=hit
    print('selected',[(candidates[i][0],sum(i in g for g in cover)) for i in selected],'uncovered routes',[i for i,g in enumerate(cover) if not g], 'remain',remain)

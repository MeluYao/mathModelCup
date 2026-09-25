# -*- coding: utf-8 -*-
"""问题四：在问题三固定联合任务上精确枚举2/3组分区并核算独立资源。"""
from pathlib import Path
from collections import defaultdict
import heapq,json,math,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;Q3=ROOT/'q3_results'/'tables';SRC=ROOT/'preprocessing'/'tables';OUT=ROOT/'q4_results';TAB=OUT/'tables';FIG=OUT/'figures';TAB.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
RES=['运输机A','运输机B','运输机C','运输电池A','运输电池B','运输电池C','中继无人机','中继能源组件']
STOCK=np.array([4,2,2,6,4,4,2,6],int)
def save(df,name):df.to_csv(TAB/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.12g')
def dump(x,p):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def peak(intervals):
    events=[]
    for s,e,*_ in intervals:events.extend([(float(s),1),(float(e),-1)])
    cur=mx=0
    for _,v in sorted(events,key=lambda x:(x[0],x[1])):cur+=v;mx=max(mx,cur)
    return mx
def color(intervals,prefix):
    """最优区间图着色；返回原任务到组内资源编号。"""
    active=[];free=[];next_id=1;ans={}
    for s,e,task in sorted(intervals,key=lambda z:(z[0],z[1],z[2])):
        while active and active[0][0]<=s+1e-9:
            _,rid=heapq.heappop(active);heapq.heappush(free,rid)
        if free:rid=heapq.heappop(free)
        else:rid=next_id;next_id+=1
        ans[task]=f'{prefix}{rid:02d}';heapq.heappush(active,(e,rid))
    return ans,next_id-1

class PartitionModel:
    def __init__(self):
        self.plan=pd.read_csv(Q3/'transport_plan.csv');self.deliv=pd.read_csv(Q3/'box_deliveries.csv');self.relay=pd.read_csv(Q3/'relay_plan.csv');self.nodes=pd.read_csv(SRC/'nodes.csv').set_index('node_id');self.dem=pd.read_csv(SRC/'service_demand_summary.csv').set_index('service_id')
        self.services=sorted(self.deliv.service_id.unique());self.make_units();self.n=len(self.units);self.full=(1<<self.n)-1
        self.service_unit={s:i for i,u in enumerate(self.units) for s in u}
        self.flight_services={r.flight_id:set(r.visit_order.split('-')) for r in self.plan.itertuples()}
        self.relay_flights={r.relay_flight_id:set(str(r.covered_transport_flights).split(';')) for r in self.relay.itertuples()}
        self.sub=[];self.precompute();self.baseline=self.sub[self.full]['resources']
        # 热循环使用普通 tuple，避免数十万次枚举中反复创建 NumPy 小数组。
        self.rtab=[tuple(int(v) for v in x['resources']) for x in self.sub]
        self.wtab=[tuple(float(v) for v in x['work']) for x in self.sub]
        self.stab=[float(x['sse']) for x in self.sub]

    def make_units(self):
        parent={s:s for s in self.services}
        def find(x):
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b:parent[b]=a
        for r in self.plan.itertuples():
            ss=r.visit_order.split('-')
            for s in ss[1:]:union(ss[0],s)
        g=defaultdict(list)
        for s in self.services:g[find(s)].append(s)
        self.units=sorted((tuple(sorted(v)) for v in g.values()),key=lambda u:u[0])

    def group_data(self,services):
        flights=set(r.flight_id for r in self.plan.itertuples() if self.flight_services[r.flight_id]&services)
        relays=set(rid for rid,fs in self.relay_flights.items() if fs&flights)
        tp=self.plan[self.plan.flight_id.isin(flights)];rp=self.relay[self.relay.relay_flight_id.isin(relays)]
        rv=[]
        for g in 'ABC':
            z=tp[tp.type_id.eq(g)];rv.append(peak([(x.start_s,x.return_s,x.flight_id) for x in z.itertuples()]))
        for g in 'ABC':
            z=tp[tp.type_id.eq(g)];rv.append(peak([(x.start_s,x.battery_recharged_s,x.flight_id) for x in z.itertuples()]))
        rv.append(peak([(x.start_s,x.return_s+300,x.relay_flight_id) for x in rp.itertuples()]))
        rv.append(peak([(x.start_s,x.available_again_s,x.relay_flight_id) for x in rp.itertuples()]))
        dd=self.deliv[self.deliv.service_id.isin(services)];work=np.array([len(dd),len(tp),tp.energy_kwh.sum(),(tp.return_s-tp.start_s).sum(),len(rp),(rp.service_end_s-rp.service_start_s).sum()],float)
        xy=self.nodes.loc[sorted(services),['x_m','y_m']].to_numpy(float);w=self.deliv.groupby('service_id').size().loc[sorted(services)].to_numpy(float);cent=np.average(xy,axis=0,weights=w);sse=float(((xy-cent)**2*w[:,None]).sum())
        return dict(services=tuple(sorted(services)),flights=tuple(sorted(flights)),relays=tuple(sorted(relays)),resources=np.array(rv,int),work=work,sse=sse,centroid=cent)

    def precompute(self):
        self.sub=[None]*(self.full+1);self.sub[0]=dict(resources=np.zeros(8,int),work=np.zeros(6),sse=0.,services=(),flights=(),relays=(),centroid=np.zeros(2))
        for mask in range(1,self.full+1):
            services=set()
            for i,u in enumerate(self.units):
                if mask>>i&1:services.update(u)
            self.sub[mask]=self.group_data(services)
        allxy=self.nodes.loc[self.services,['x_m','y_m']].to_numpy(float);w=self.deliv.groupby('service_id').size().loc[self.services].to_numpy(float);cent=np.average(allxy,axis=0,weights=w);self.total_sse=float(((allxy-cent)**2*w[:,None]).sum())

    def partitions(self,k):
        if k==2:
            for a in range(1,self.full):
                if not(a&1):continue
                b=self.full^a
                if b:yield (a,b)
        else:
            for a in range(1,self.full):
                if not(a&1):continue
                rem=self.full^a
                if rem==0:continue
                anchor=rem&-rem;b=rem
                while b:
                    if b&anchor:
                        c=rem^b
                        if c:yield (a,b,c)
                    b=(b-1)&rem

    def measures(self,parts):
        k=len(parts);req=[sum(self.rtab[p][j] for p in parts) for j in range(8)]
        deficit=[max(req[j]-int(STOCK[j]),0) for j in range(8)]
        defscore=sum(deficit[j]/float(STOCK[j]) for j in range(8))
        red=sum(max(req[j]-int(self.baseline[j]),0)/float(STOCK[j]) for j in range(8))
        weights=(.20,.15,.20,.15,.10,.20);balance=0.0
        for j,wt in enumerate(weights):
            vals=[self.wtab[p][j] for p in parts];mean=sum(vals)/k
            if mean>0:balance+=wt*math.sqrt(sum((v-mean)**2 for v in vals)/k)/mean
        spatial=sum(self.stab[p] for p in parts)/self.total_sse
        return float(defscore),float(red),float(balance),float(spatial),np.array(req,int),np.array(deficit,int)

    def choose(self,k):
        # ε-约束先排除名义分组：箱量和运输架次必须落在平均值附近。
        # 两组采用±25%；三组考虑整数架次，采用箱量±35%、架次约±36%。
        total=self.wtab[self.full];box_tol=.25 if k==2 else .35;trip_tol=.25 if k==2 else .36
        box_lo,box_hi=total[0]/k*(1-box_tol),total[0]/k*(1+box_tol)
        trip_lo,trip_hi=total[1]/k*(1-trip_tol),total[1]/k*(1+trip_tol)
        min_def=math.inf;global_min_def=math.inf;count=0;balanced_count=0;eligible_rows=[]
        for parts in self.partitions(k):
            count+=1
            req=[sum(self.rtab[p][j] for p in parts) for j in range(8)]
            gd=sum(max(req[j]-int(STOCK[j]),0)/float(STOCK[j]) for j in range(8));global_min_def=min(global_min_def,gd)
            if not all(box_lo-1e-9<=self.wtab[p][0]<=box_hi+1e-9 and trip_lo-1e-9<=self.wtab[p][1]<=trip_hi+1e-9 for p in parts):continue
            balanced_count+=1;d,red,bal,spa,req,deficit=self.measures(parts)
            if d<min_def-1e-12:min_def=d;eligible_rows=[]
            if abs(d-min_def)<=1e-12:eligible_rows.append((parts,red,bal,spa,req,deficit))
        vals=np.array([[x[1],x[2],x[3]] for x in eligible_rows]);lo=vals.min(0);hi=vals.max(0);eligible=len(eligible_rows)
        best=None
        for parts,red,bal,spa,req,deficit in eligible_rows:
            v=np.array([red,bal,spa]);norm=(v-lo)/np.where(hi>lo,hi-lo,1);dist=float(math.sqrt(.40*norm[0]**2+.45*norm[1]**2+.15*norm[2]**2))
            key=(dist,red,bal,spa)
            if best is None or key<best['key']:best=dict(parts=parts,key=key,defscore=min_def,red=red,balance=bal,spatial=spa,req=req,deficit=deficit,ideal_distance=dist)
        best.update(enumerated=count,balanced_candidates=balanced_count,eligible=eligible,global_min_deficit_lower_bound=global_min_def,balance_bounds={'boxes':[box_lo,box_hi],'transport_trips':[trip_lo,trip_hi]},lo=lo.tolist(),hi=hi.tolist());return best

    def relabel(self,parts):
        # 西到东排序，便于解释与地图阅读。
        return sorted(parts,key=lambda x:(self.sub[x]['centroid'][0],self.sub[x]['centroid'][1]))

    def allocations(self,scenario,parts):
        rows=[]
        for gi,mask in enumerate(parts,1):
            gd=self.sub[mask];tp=self.plan[self.plan.flight_id.isin(gd['flights'])];rp=self.relay[self.relay.relay_flight_id.isin(gd['relays'])]
            for typ in 'ABC':
                z=tp[tp.type_id.eq(typ)];ints=[(x.start_s,x.return_s,x.flight_id) for x in z.itertuples()];mp,_=color(ints,f'{scenario}-G{gi}-U{typ}-')
                bins=[(x.start_s,x.battery_recharged_s,x.flight_id) for x in z.itertuples()];bp,_=color(bins,f'{scenario}-G{gi}-BAT{typ}-')
                for x in z.itertuples():
                    rows.extend([dict(scenario=scenario,group_id=f'G{gi}',task_class='运输架次',task_id=x.flight_id,resource_class=f'运输机{typ}',local_resource_id=mp[x.flight_id],occupied_start_s=x.start_s,occupied_end_s=x.return_s,available_again_s=x.return_s),dict(scenario=scenario,group_id=f'G{gi}',task_class='运输架次',task_id=x.flight_id,resource_class=f'运输电池{typ}',local_resource_id=bp[x.flight_id],occupied_start_s=x.start_s,occupied_end_s=x.return_s,available_again_s=x.battery_recharged_s)])
            ri=[(x.start_s,x.return_s+300,x.relay_flight_id) for x in rp.itertuples()];rm,_=color(ri,f'{scenario}-G{gi}-R-')
            ei=[(x.start_s,x.available_again_s,x.relay_flight_id) for x in rp.itertuples()];em,_=color(ei,f'{scenario}-G{gi}-RE-')
            for x in rp.itertuples():
                rows.extend([dict(scenario=scenario,group_id=f'G{gi}',task_class='中继架次',task_id=x.relay_flight_id,resource_class='中继无人机',local_resource_id=rm[x.relay_flight_id],occupied_start_s=x.start_s,occupied_end_s=x.return_s,available_again_s=x.return_s+300),dict(scenario=scenario,group_id=f'G{gi}',task_class='中继架次',task_id=x.relay_flight_id,resource_class='中继能源组件',local_resource_id=em[x.relay_flight_id],occupied_start_s=x.start_s,occupied_end_s=x.return_s,available_again_s=x.available_again_s)])
        return rows

def run():
    started=time.time();m=PartitionModel();solutions={2:m.choose(2),3:m.choose(3)};assign=[];groups=[];rescomp=[];alloc=[];dups=[]
    for k,sol in solutions.items():
        scenario=f'{k}组';parts=m.relabel(sol['parts']);sol['parts_ordered']=parts
        total_req=np.sum([m.sub[x]['resources'] for x in parts],axis=0);deficit=np.maximum(total_req-STOCK,0)
        for gi,mask in enumerate(parts,1):
            gd=m.sub[mask];dd=m.deliv[m.deliv.service_id.isin(gd['services'])]
            for s in gd['services']:assign.append(dict(scenario=scenario,group_id=f'G{gi}',service_id=s,atomic_unit=';'.join(m.units[m.service_unit[s]])))
            row=dict(scenario=scenario,group_id=f'G{gi}',services=';'.join(gd['services']),service_count=len(gd['services']),boxes=int(gd['work'][0]),mass_kg=float(m.dem.loc[list(gd['services']),'total_mass_kg'].sum()),transport_trips=int(gd['work'][1]),transport_energy_kwh=float(gd['work'][2]),transport_busy_s=float(gd['work'][3]),relay_task_copies=int(gd['work'][4]),relay_service_s=float(gd['work'][5]),joint_task_count=int(gd['work'][1]+gd['work'][4]))
            row.update({RES[j]:int(gd['resources'][j]) for j in range(8)});groups.append(row)
        for j,name in enumerate(RES):rescomp.append(dict(scenario=scenario,resource=name,centralized_required=int(m.baseline[j]),partition_total_required=int(total_req[j]),existing_stock=int(STOCK[j]),redundancy=int(total_req[j]-m.baseline[j]),spare=int(max(STOCK[j]-total_req[j],0)),deficit=int(deficit[j])))
        alloc.extend(m.allocations(scenario,parts))
        for rid,fs in m.relay_flights.items():
            gs=[f'G{i+1}' for i,x in enumerate(parts) if set(fs)&set(m.sub[x]['flights'])]
            dups.append(dict(scenario=scenario,relay_flight_id=rid,assigned_groups=';'.join(gs),copy_count=len(gs),additional_copies=max(0,len(gs)-1)))
    save(pd.DataFrame(assign),'partition_assignments');save(pd.DataFrame(groups),'group_summary');save(pd.DataFrame(rescomp),'resource_comparison');save(pd.DataFrame(alloc),'task_resource_allocation');save(pd.DataFrame(dups),'relay_duplication')
    compare=[]
    for k,s in solutions.items():
        compare.append(dict(scenario=f'{k}组',enumerated_partitions=s['enumerated'],min_relative_deficit=s['defscore'],resource_redundancy_index=s['red'],workload_balance_cv=s['balance'],spatial_dispersion_ratio=s['spatial'],ideal_distance=s['ideal_distance'],total_deficit_units=int(s['deficit'].sum())))
    save(pd.DataFrame(compare),'partition_comparison')
    summary=dict(atomic_units=[list(x) for x in m.units],atomic_unit_count=m.n,centralized_required={RES[i]:int(m.baseline[i]) for i in range(8)},stock={RES[i]:int(STOCK[i]) for i in range(8)},solutions={str(k):{kk:(vv.tolist() if isinstance(vv,np.ndarray) else vv) for kk,vv in s.items() if kk not in ('parts','parts_ordered','key')}|{'groups':[list(m.sub[x]['services']) for x in s['parts_ordered']]} for k,s in solutions.items()},elapsed_s=time.time()-started,selection_rule='minimum relative inventory deficit, then normalized ideal distance: 40% redundancy, 45% workload balance, 15% spatial dispersion')
    dump(summary,OUT/'solver_summary.json')
    wb={}
    for n,f in [('分区方案','partition_assignments'),('分组汇总','group_summary'),('资源比较','resource_comparison'),('任务资源配置','task_resource_allocation'),('中继复制','relay_duplication'),('方案比较','partition_comparison')]:
        z=pd.read_csv(TAB/f'{f}.csv');wb[n]={'columns':list(z.columns),'rows':[[None if pd.isna(v) else v for v in row] for row in z.itertuples(index=False,name=None)]}
    dump(wb,OUT/'workbook_data.json');print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':run()

# -*- coding: utf-8 -*-
"""问题三：通信约束下的运输—中继联合调度启发式求解。"""
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
import hashlib,json,math,random,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; SRC=ROOT/'preprocessing'/'tables'; Q2=ROOT/'q2_results'/'tables'
OUT=ROOT/'q3_results'; TAB=OUT/'tables'; FIG=OUT/'figures'; TAB.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
G0=9.80665; DT=10.; LINK_MARGIN=.20
def save(df,name):df.to_csv(TAB/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.12g')
def dump(x,p):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

@dataclass(frozen=True)
class Site:
    sid:str;x:float;y:float;ground:float;agl:float;alt:float;backhaul_margin:float
    out_s:float;back_s:float;flight_energy:float;lead_s:float;max_service_s:float

class JointModel:
    def __init__(self):
        self.nodes=pd.read_csv(SRC/'nodes.csv').set_index('node_id');self.arcs=pd.read_csv(SRC/'arcs.csv').set_index(['from_id','to_id'])
        self.tt=pd.read_csv(SRC/'transport_types.csv').set_index('type_id');self.boxes=pd.read_csv(SRC/'boxes.csv').set_index('box_id')
        self.rtype=pd.read_csv(SRC/'relay_types.csv').iloc[0];self.rinv=pd.read_csv(SRC/'relay_energy_inventory.csv').iloc[0]
        self.demz=np.load(ROOT/'preprocessing'/'spatial'/'dem.npz');self.dem=self.demz['elevation_m'];self.lons=self.demz['lon_deg'];self.lats=self.demz['lat_deg']
        self.proj=json.loads((ROOT/'preprocessing'/'spatial'/'projection.json').read_text(encoding='utf-8'))
        self.gateway=(0.,0.,float(self.nodes.loc['O01','ground_m'])+20.)
        self.plan_cache={}
        self.sites=self.make_sites();print('通信可行候选悬停状态',len(self.sites),flush=True)

    def elev(self,x,y):
        lon=self.proj['lon0']+x/self.proj['east_m_per_degree'];lat=self.proj['lat0']+y/self.proj['north_m_per_degree']
        c=int(np.clip(np.searchsorted(self.lons,lon),1,len(self.lons)-1));r=int(np.clip(np.searchsorted(-self.lats,-lat),1,len(self.lats)-1))
        c=c-1 if abs(self.lons[c-1]-lon)<abs(self.lons[c]-lon) else c;r=r-1 if abs(self.lats[r-1]-lat)<abs(self.lats[r]-lat) else r
        return float(self.dem[r,c])

    @lru_cache(maxsize=600000)
    def link_cached(self,a,b,limit,step=90):
        x1,y1,h1=a;x2,y2,h2=b;horizontal=math.hypot(x2-x1,y2-y1);d3=math.sqrt(horizontal**2+(h2-h1)**2)/1000
        n=max(3,int(horizontal/step)+2);blocked=False
        for u in np.linspace(0,1,n)[1:-1]:
            x=x1+(x2-x1)*u;y=y1+(y2-y1)*u;h=h1+(h2-h1)*u
            if self.elev(x,y)>h-1e-7:blocked=True;break
        loss=32.45+20*math.log10(2400)+20*math.log10(max(d3,1e-6))+(10 if blocked else 0)
        return loss<=limit-LINK_MARGIN,limit-loss,blocked
    def link(self,a,b,limit,step=90):
        aa=tuple(round(float(x),2) for x in a);bb=tuple(round(float(x),2) for x in b)
        return self.link_cached(aa,bb,float(limit),int(step))

    def relay_flight(self,x,y,alt):
        rt=self.rtype;dist=math.hypot(x,y);n=max(3,int(dist/30)+2);mx=max(self.elev(x*u,y*u) for u in np.linspace(0,1,n));cr=max(mx+50,alt)
        oalt=float(self.nodes.loc['O01','ground_m']);
        out_cl=max(0.,cr-oalt);out_de=max(0.,cr-alt);back_cl=max(0.,cr-alt);back_de=max(0.,cr-oalt)
        horiz=dist/rt.cruise_mps
        out=out_cl/rt.up_mps+horiz+out_de/rt.down_mps;back=back_cl/rt.up_mps+horiz+back_de/rt.down_mps
        ehor=2*rt.cruise_power_kw*horiz/3600
        eup=rt.takeoff_mass_kg*G0*(out_cl+back_cl)/(rt.up_efficiency*3.6e6)
        energy=float(ehor+eup);usable=(1-rt.reserve_fraction)*rt.energy_kwh
        max_service=(usable-energy)/(rt.hover_power_kw+rt.comm_power_kw)*3600
        return float(out),float(back),energy,float(max_service)

    def make_sites(self):
        raw=[]
        for s,r in self.nodes.iterrows():
            if s=='O01':continue
            for frac in [.50,.65,.80,1.00]:
                x=float(r.x_m*frac);y=float(r.y_m*frac);raw.append((f'{s}-F{int(frac*100):02d}',x,y))
        pairs=[('S002','S004'),('S003','S015'),('S012','S014'),('S006','S007'),('S008','S005'),('S001','S009'),('S010','S013')]
        for a,b in pairs:raw.append((f'{a}-{b}',float((self.nodes.loc[a,'x_m']+self.nodes.loc[b,'x_m'])/2),float((self.nodes.loc[a,'y_m']+self.nodes.loc[b,'y_m'])/2)))
        sites=[];seen=set()
        for name,x,y in raw:
            key=(round(x,1),round(y,1))
            if key in seen:continue
            seen.add(key);ground=self.elev(x,y)
            for agl in [150.,225.,300.]:
                alt=ground+agl;ok,margin,_=self.link((x,y,alt),self.gateway,126)
                if not ok:continue
                out,back,ef,maxsvc=self.relay_flight(x,y,alt)
                if maxsvc<=60:continue
                sid=f'H{len(sites)+1:03d}'
                lead=float(self.rtype.prepare_s+out+self.rtype.link_setup_s)
                sites.append(Site(sid,x,y,ground,agl,alt,margin,out,back,ef,lead,maxsvc))
        return sites

    def trajectory(self,p):
        """返回相对任务开始的分段：(t0,t1,位置函数类型及参数)。"""
        t=self.tt.loc[p.type_id];ids=p.boxes.split(';');b=self.boxes.loc[ids]
        clock=float(t.prepare_s+len(ids)*t.load_per_box_s);order=p.visit_order.split('-');seq=['O01']+order+['O01'];ph=[]
        for leg,(i,j) in enumerate(zip(seq[:-1],seq[1:])):
            a=self.arcs.loc[(i,j)];ni=self.nodes.loc[i];nj=self.nodes.loc[j];oi=float(ni.ground_m+(0 if i=='O01' else 30));oj=float(nj.ground_m+(0 if j=='O01' else 30));cr=float(a.cruise_alt_m)
            dur=float(a.climb_m/t.up_mps);ph.append((clock,clock+dur,'vertical',(float(ni.x_m),float(ni.y_m),oi,cr)));clock+=dur
            dur=float(a.distance_m/t.cruise_mps);ph.append((clock,clock+dur,'linear',(float(ni.x_m),float(ni.y_m),float(nj.x_m),float(nj.y_m),cr)));clock+=dur
            dur=float(a.descent_m/t.down_mps);ph.append((clock,clock+dur,'vertical',(float(nj.x_m),float(nj.y_m),cr,oj)));clock+=dur
            if j!='O01':
                here=b[b.service_id.eq(j)];dur=float(t.handover_base_s+len(here)*t.handover_per_box_s);ph.append((clock,clock+dur,'fixed',(float(nj.x_m),float(nj.y_m),oj)));clock+=dur
        return ph
    def pos(self,phase,t):
        a,b,k,v=phase;u=0 if b<=a else min(1,max(0,(t-a)/(b-a)))
        if k=='fixed':return v
        if k=='vertical':return (v[0],v[1],v[2]+u*(v[3]-v[2]))
        return (v[0]+u*(v[2]-v[0]),v[1]+u*(v[3]-v[1]),v[4])
    def route_profile(self,p):
        ph=self.trajectory(p);times=set()
        for a,b,*_ in ph:
            times.add(a);times.add(b)
            for x in np.arange(math.ceil(a/DT)*DT,b,DT):times.add(float(x))
        rows=[]
        for tm in sorted(times):
            phase=next((x for x in ph if x[0]-1e-7<=tm<=x[1]+1e-7),ph[-1]);q=self.pos(phase,tm);direct,margin,blocked=self.link(q,self.gateway,122)
            rows.append((tm,q,direct,margin,blocked))
        need=[r for r in rows if not r[2]]
        return rows,need

    def site_covers(self,site,need):return all(self.link(q,(site.x,site.y,site.alt),116)[0] for _,q,*_ in need)

    def charge(self,soc):
        full=float(self.rinv.full_charge_s)
        return full*(.65*(.9-soc)/.9+.35) if soc<.9 else full*.35*(1-soc)/.1

    def blocks(self,plan,profiles,assignment,merge_gap):
        intervals=defaultdict(list)
        for i,p in enumerate(plan.itertuples()):
            need=profiles[i][1]
            if not need:continue
            intervals[assignment[i]].append((p.start_s+need[0][0],p.start_s+need[-1][0],i))
        result=[]
        for si,ints in intervals.items():
            ints.sort();s,e=ints[0][0],ints[0][1];routes={ints[0][2]};site=self.sites[si]
            merged=[]
            for a,b,i in ints[1:]:
                if a<=e+merge_gap:e=max(e,b);routes.add(i)
                else:merged.append((s,e,set(routes)));s,e,routes=a,b,{i}
            merged.append((s,e,set(routes)))
            for s,e,routes in merged:
                cur=s
                while e-cur>site.max_service_s:
                    result.append(dict(site_idx=si,service_start=cur,service_end=cur+site.max_service_s,routes=set(routes)));cur+=site.max_service_s
                result.append(dict(site_idx=si,service_start=cur,service_end=e,routes=set(routes)))
        return result

    def relay_schedule(self,blocks):
        # 统一平移运输计划，使最早的中继架次可从t=0开始准备。
        delta=max([0.]+[self.sites[b['site_idx']].lead_s-b['service_start'] for b in blocks])
        tasks=[]
        for b in blocks:
            site=self.sites[b['site_idx']];ss=b['service_start']+delta;se=b['service_end']+delta
            start=ss-site.lead_s;ret=se+site.back_s;dur=se-ss
            energy=site.flight_energy+(self.rtype.hover_power_kw+self.rtype.comm_power_kw)*dur/3600;soc=1-energy/self.rtype.energy_kwh
            tasks.append(dict(site_idx=b['site_idx'],service_start_s=ss,service_end_s=se,start_s=start,return_s=ret,energy_kwh=energy,return_soc=soc,routes=b['routes']))
        tasks.sort(key=lambda x:x['start_s']);rav={'R01':0.,'R02':0.};eav={f'RE-R-{i:02d}':0. for i in range(1,int(self.rinv.resource_count)+1)}
        conflicts=0
        for x in tasks:
            drones=[r for r,a in rav.items() if a<=x['start_s']+1e-7]
            comps=[r for r,a in eav.items() if a<=x['start_s']+1e-7]
            if not drones:
                conflicts+=1;drone=min(rav,key=rav.get);x['resource_conflict']=True;x['required_start_s']=rav[drone]
            else:drone=max(drones,key=lambda r:rav[r])
            if not comps:
                conflicts+=1;comp=min(eav,key=eav.get)
            else:comp=max(comps,key=lambda r:eav[r])
            x['relay_id']=drone;x['energy_id']=comp
            rav[drone]=x['return_s']+self.rtype.turnaround_s
            eav[comp]=x['return_s']+self.charge(x['return_soc']);x['available_again_s']=eav[comp]
        return tasks,delta,conflicts

    def propagate_transport(self,base,requested):
        """保持问题二同机/同电池先后关系，向后传播新增延迟。"""
        work=base.copy();dur=(base.return_s-base.start_s).to_dict();rech=(base.battery_recharged_s-base.start_s).to_dict()
        work['start_s']=[max(float(base.loc[i,'start_s']),float(requested.get(i,base.loc[i,'start_s']))) for i in work.index]
        air={};bat={}
        for i in sorted(work.index,key=lambda j:(base.loc[j,'start_s'],j)):
            s=float(work.loc[i,'start_s']);s=max(s,air.get(base.loc[i,'aircraft_id'],0.),bat.get(base.loc[i,'battery_id'],0.))
            work.loc[i,'start_s']=s;work.loc[i,'return_s']=s+dur[i];work.loc[i,'battery_recharged_s']=s+rech[i]
            air[base.loc[i,'aircraft_id']]=work.loc[i,'return_s'];bat[base.loc[i,'battery_id']]=work.loc[i,'battery_recharged_s']
        return work

    def repair_schedule(self,base,profiles,assignment,merge_gap):
        requested={i:float(base.loc[i,'start_s']) for i in base.index};work=self.propagate_transport(base,requested)
        for _ in range(24):
            blocks=self.blocks(work,profiles,assignment,merge_gap);tasks,delta,conflicts=self.relay_schedule(blocks)
            if delta>1e-4:
                # 只推迟造成最早负开始时刻的服务块及其运输架次。
                first=min(blocks,key=lambda b:b['service_start']-self.sites[b['site_idx']].lead_s)
                for i in first['routes']:requested[i]=max(requested[i],float(work.loc[i,'start_s'])+delta+1.)
                work=self.propagate_transport(base,requested);continue
            if conflicts==0:return work,tasks
            bad=next(x for x in tasks if x.get('resource_conflict'))
            shift=max(1.,bad['required_start_s']-bad['start_s']+1.)
            for i in bad['routes']:requested[i]=max(requested[i],float(work.loc[i,'start_s'])+shift)
            work=self.propagate_transport(base,requested)
        return None,None

    def assignment_search(self,plan,profiles,options,selected,merge_gap,seed):
        req=[i for i,p in enumerate(profiles) if p[1]];rng=random.Random(seed)
        base_deliv=pd.read_csv(Q2/f"deliveries_{plan.attrs['policy']}.csv")
        hard_base=base_deliv[base_deliv.hard_deadline_s.notna()]
        choices={i:[s for s in options[i] if s in selected] for i in req}
        if any(not choices[i] for i in req):return None
        def evaluate(a):
            blocks=self.blocks(plan,profiles,a,merge_gap);tasks,delta,conflicts=self.relay_schedule(blocks)
            viol=int(((hard_base.delivery_s+delta)>hard_base.hard_deadline_s+1e-7).sum());late=float(np.maximum(hard_base.delivery_s+delta-hard_base.hard_deadline_s,0).max() if len(hard_base) else 0)
            energy=sum(x['energy_kwh'] for x in tasks);makespan=max(float(plan.return_s.max()+delta),max((x['return_s'] for x in tasks),default=0))
            return (viol,late,conflicts,len(tasks),energy,makespan,delta),tasks
        cur={i:min(choices[i],key=lambda s:(self.sites[s].flight_energy,-self.sites[s].backhaul_margin)) for i in req};best_key,best_tasks=evaluate(cur);best=cur.copy()
        cur_key=best_key
        for _ in range(320):
            cand=best.copy() if rng.random()<.75 else cur.copy();i=rng.choice(req);cand[i]=rng.choice(choices[i]);key,tasks=evaluate(cand)
            if key<best_key:best_key,best_tasks,best=key,tasks,cand.copy()
            if key<=cur_key or rng.random()<.02:cur,cur_key=cand,key
        if best_tasks is None:return None
        return dict(key=best_key,tasks=best_tasks,assignment=best)

    def candidate_sets(self,options,profiles,seed):
        req=[i for i,p in enumerate(profiles) if p[1]];sets=[];rng=random.Random(seed)
        for rep in range(14):
            remain=set(req);chosen=[]
            while remain:
                scores=[]
                for s in range(len(self.sites)):
                    hit=sum(s in options[i] for i in remain)
                    if hit:scores.append((hit*(.85+.3*rng.random())/(1+.08*self.sites[s].flight_energy),s))
                if not scores:break
                s=max(scores)[1];chosen.append(s);remain={i for i in remain if s not in options[i]}
            if not remain:
                key=tuple(sorted(chosen))
                if key not in sets:sets.append(key)
        return sets

    def solve_plan(self,policy,merge_gap):
        if policy in self.plan_cache:
            plan,profiles,options=self.plan_cache[policy]
        else:
            plan=pd.read_csv(Q2/f'plan_{policy}.csv');plan.attrs['policy']=policy
            profiles=[];options={}
            for i,p in enumerate(plan.itertuples()):
                prof=self.route_profile(p);profiles.append(prof)
                options[i]=[s for s,site in enumerate(self.sites) if self.site_covers(site,prof[1])] if prof[1] else []
                if prof[1] and not options[i]:raise RuntimeError(f'{policy} {p.flight_id}无单点完整中继覆盖')
            self.plan_cache[policy]=(plan,profiles,options)
        best=None
        for k,sites in enumerate(self.candidate_sets(options,profiles,177+int(merge_gap))):
            x=self.assignment_search(plan,profiles,options,sites,merge_gap,1000+k)
            if x and (best is None or x['key']<best['key']):best=x
        if best is None:return None
        if best is None:return None
        p,tasks=self.repair_schedule(plan,profiles,best['assignment'],merge_gap)
        if p is None:return None
        # 按路线索引把运输架次延迟传递到逐箱送达时刻。
        delay_by_route={int(p.loc[i,'route_index']):float(p.loc[i,'start_s']-plan.loc[i,'start_s']) for i in p.index}
        deliv=pd.read_csv(Q2/f'deliveries_{policy}.csv');d=deliv.copy();d['delivery_s']=[r.delivery_s+delay_by_route[int(r.route_index)] for r in d.itertuples()]
        delta=float(max(delay_by_route.values(),default=0.))
        hard=d[d.hard_deadline_s.notna()];soft=np.maximum(d.delivery_s-d.expected_s,0)
        relay_energy=sum(x['energy_kwh'] for x in tasks);metrics=dict(policy=policy,merge_gap_s=merge_gap,hard_violations=int((hard.delivery_s>hard.hard_deadline_s+1e-7).sum()),max_hard_lateness_s=float(np.maximum(hard.delivery_s-hard.hard_deadline_s,0).max()),weighted_tardiness=float((d.priority*soft/d.expected_s).sum()),weighted_delivery_ratio=float((d.priority*d.delivery_s/d.expected_s).sum()/d.priority.sum()),on_time_boxes=int((d.delivery_s<=d.expected_s+1e-7).sum()),joint_makespan_s=float(max(p.return_s.max(),max(x['return_s'] for x in tasks))),transport_energy_kwh=float(p.energy_kwh.sum()),relay_energy_kwh=float(relay_energy),total_energy_kwh=float(p.energy_kwh.sum()+relay_energy),transport_trips=len(p),relay_trips=len(tasks),total_trips=len(p)+len(tasks),transport_shift_s=float(delta),hover_sites=len(set(x['site_idx'] for x in tasks)))
        return dict(plan=p,deliveries=d,profiles=profiles,options=options,assignment=best['assignment'],relay=tasks,metrics=metrics)

def nondominated(xs):
    keys=['weighted_tardiness','weighted_delivery_ratio','joint_makespan_s','total_energy_kwh','transport_trips','relay_trips'];out=[]
    for i,x in enumerate(xs):
        a=x['metrics'];dom=False
        for j,y in enumerate(xs):
            if i==j:continue
            b=y['metrics']
            if all(b[k]<=a[k]+1e-9 for k in keys) and any(b[k]<a[k]-1e-9 for k in keys):dom=True;break
        if not dom:out.append(x)
    return out

def run():
    started=time.time();m=JointModel();solutions=[]
    for pol in ['balanced','timely']:
        for gap in [0.,180.,360.,600.]:
            s=m.solve_plan(pol,gap)
            if s:solutions.append(s);print(pol,gap,s['metrics'],flush=True)
    feasible=[s for s in solutions if s['metrics']['hard_violations']==0]
    if not feasible:raise RuntimeError('无硬时限可行联合方案')
    pareto=nondominated(feasible);keys=['weighted_delivery_ratio','joint_makespan_s','total_energy_kwh','transport_trips','relay_trips']
    a=np.array([[x['metrics'][k] for k in keys] for x in pareto]);lo=a.min(0);hi=a.max(0);span=np.where(hi>lo,hi-lo,1);dist=np.sqrt((((a-lo)/span)**2).mean(1));main=pareto[int(np.argmin(dist))]
    # 标记并保存运输与交付
    p=main['plan'].copy();p['flight_id']=p.flight_id.str.replace('Q2-','Q3-T-',regex=False);d=main['deliveries'].copy();d['flight_id']=d.flight_id.str.replace('Q2-','Q3-T-',regex=False)
    save(p,'transport_plan');save(d,'box_deliveries')
    # 中继表
    rr=[]
    for k,x in enumerate(sorted(main['relay'],key=lambda z:z['start_s']),1):
        site=m.sites[x['site_idx']];rr.append(dict(relay_flight_id=f'Q3-R-{k:03d}',relay_id=x['relay_id'],energy_id=x['energy_id'],hover_site_id=site.sid,x_m=site.x,y_m=site.y,ground_m=site.ground,hover_agl_m=site.agl,flight_alt_m=site.alt,start_s=x['start_s'],service_start_s=x['service_start_s'],service_end_s=x['service_end_s'],return_s=x['return_s'],energy_kwh=x['energy_kwh'],return_soc=x['return_soc'],available_again_s=x['available_again_s'],covered_transport_flights=';'.join(sorted(p.iloc[list(x['routes'])].flight_id))))
    relay=pd.DataFrame(rr);save(relay,'relay_plan')
    # 逐运输架次通信摘要及10秒明细
    summaries=[];timeline=[]
    delta=main['metrics']['transport_shift_s']
    for i,row in enumerate(p.itertuples()):
        prof=main['profiles'][i];need=prof[1];si=main['assignment'].get(i);site=m.sites[si] if si is not None else None
        direct_n=sum(r[2] for r in prof[0]);relay_n=len(prof[0])-direct_n;min_direct=min(r[3] for r in prof[0]);access=[]
        for tm,q,direct,dmarg,blocked in prof[0]:
            mode='direct' if direct else 'relay';amarg=np.nan
            if not direct:
                ok,amarg,_=m.link(q,(site.x,site.y,site.alt),116);access.append(amarg)
            timeline.append(dict(transport_flight_id=row.flight_id,time_s=tm+row.start_s,mode=mode,hover_site_id='' if direct else site.sid,direct_margin_db=dmarg,access_margin_db=amarg,communication_ok=True))
        summaries.append(dict(transport_flight_id=row.flight_id,visit_order=row.visit_order,communication_samples=len(prof[0]),direct_samples=direct_n,relay_samples=relay_n,direct_fraction=direct_n/len(prof[0]),relay_required=bool(need),hover_site_id='' if site is None else site.sid,min_direct_margin_db=min_direct,min_access_margin_db=min(access) if access else np.nan,first_relay_need_s=(row.start_s+need[0][0]) if need else np.nan,last_relay_need_s=(row.start_s+need[-1][0]) if need else np.nan))
    save(pd.DataFrame(summaries),'communication_summary');save(pd.DataFrame(timeline),'communication_timeline_10s')
    # 两类资源占用
    resources=[]
    for r in p.itertuples():
        resources.extend([dict(resource_class='transport_aircraft',resource_id=r.aircraft_id,task_id=r.flight_id,occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.return_s,start_soc=1,end_soc=r.return_soc),dict(resource_class='transport_battery',resource_id=r.battery_id,task_id=r.flight_id,occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.battery_recharged_s,start_soc=1,end_soc=r.return_soc)])
    for r in relay.itertuples():
        resources.extend([dict(resource_class='relay_aircraft',resource_id=r.relay_id,task_id=r.relay_flight_id,occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.return_s+m.rtype.turnaround_s,start_soc=1,end_soc=r.return_soc),dict(resource_class='relay_energy',resource_id=r.energy_id,task_id=r.relay_flight_id,occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.available_again_s,start_soc=1,end_soc=r.return_soc)])
    save(pd.DataFrame(resources),'resource_usage')
    comp=pd.DataFrame([x['metrics'] for x in solutions]);save(comp,'objective_comparison');save(pd.DataFrame([{**x['metrics'],'ideal_distance':dist[i]} for i,x in enumerate(pareto)]),'pareto_candidates')
    sites_used=relay[['hover_site_id','x_m','y_m','ground_m','hover_agl_m','flight_alt_m']].drop_duplicates();save(sites_used,'hover_sites')
    summary=dict(main_metrics=main['metrics'],candidate_count=len(solutions),pareto_count=len(pareto),ideal={k:float(lo[i]) for i,k in enumerate(keys)},nadir={k:float(hi[i]) for i,k in enumerate(keys)},elapsed_s=time.time()-started,communication_time_step_s=DT,link_safety_margin_db=LINK_MARGIN,method='transport candidate comparison + DEM link screening + relay set cover/assignment search')
    dump(summary,OUT/'solver_summary.json')
    # workbook JSON
    wb={}
    for name,file in [('运输架次','transport_plan'),('逐箱交付','box_deliveries'),('中继架次','relay_plan'),('通信汇总','communication_summary'),('资源使用','resource_usage'),('目标对照','objective_comparison')]:
        z=pd.read_csv(TAB/f'{file}.csv');wb[name]={'columns':list(z.columns),'rows':z.where(pd.notna(z),None).values.tolist()}
    dump(wb,OUT/'workbook_data.json')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':run()

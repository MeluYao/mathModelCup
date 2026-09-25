"""问题二：多点路线合并、异构机队与共享电池离散事件调度。"""
from pathlib import Path
from dataclasses import dataclass
from itertools import permutations
from functools import lru_cache
from collections import defaultdict
import hashlib,json,math,random,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'q2_results'; TABLE=OUT/'tables'; TABLE.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'preprocessing'/'tables'; G0=9.80665; RHO=.20
POLICIES=['balanced','timely','makespan','energy','trips']

def save(df,name): df.to_csv(TABLE/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.12g')
def dump(x,p): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

@dataclass(frozen=True)
class Variant:
    type_id:str; order:tuple; energy:float; duration:float; delivery_offsets:tuple
    mass:float; volume:float; box_count:int; min_soc:float

class Q2:
    def __init__(self):
        self.boxes=pd.read_csv(SRC/'boxes.csv').set_index('box_id')
        self.types=pd.read_csv(SRC/'transport_types.csv').set_index('type_id')
        self.arcs=pd.read_csv(SRC/'arcs.csv').set_index(['from_id','to_id'])
        self.fleet=pd.read_csv(SRC/'transport_fleet.csv')
        self.inv=pd.read_csv(SRC/'transport_battery_inventory.csv').set_index('type_id')
        self.batteries={g:[f'BAT-{g}-{i:02d}' for i in range(1,int(r.resource_count)+1)] for g,r in self.inv.iterrows()}
        self.seed=self.priority_seed()
        assert len(set().union(*self.seed))==80

    def priority_seed(self):
        """将3600秒硬时限箱单独放入首波候选，其余逐区按C型可行性首次适配。"""
        routes=[]
        for service in sorted(self.boxes.service_id.unique()):
            b=self.boxes[self.boxes.service_id.eq(service)].copy()
            urgent=list(b.index[b.hard_deadline_s.eq(3600)])
            if urgent:routes.append(frozenset(urgent))
            remaining=[x for x in b.sort_values(['expected_s','priority'],ascending=[True,False]).index if x not in urgent]
            bins=[]
            for box in remaining:
                placed=False
                for i,bin_ in enumerate(bins):
                    trial=frozenset(set(bin_)|{box})
                    if any(v.type_id=='C' for v in self.variants(trial)):
                        bins[i]=trial;placed=True;break
                if not placed:bins.append(frozenset([box]))
            routes.extend(bins)
        return tuple(routes)

    def segment(self,g,i,j,q):
        t=self.types.loc[g]; a=self.arcs.loc[(i,j)]
        L=t.range_empty_m-(t.range_empty_m-t.range_full_m)*(q/t.max_payload_kg)**1.5
        e=t.energy_kwh*a.distance_m/L+(t.empty_mass_kg+q)*G0*a.climb_m/(t.up_efficiency*3.6e6)
        ft=a.climb_m/t.up_mps+a.distance_m/t.cruise_mps+a.descent_m/t.down_mps
        return float(e),float(ft)

    @lru_cache(maxsize=None)
    def variants(self,route):
        route=frozenset(route); b=self.boxes.loc[list(route)]
        mass=float(b.mass_kg.sum()); vol=float(b.volume_m3.sum()); n=len(b)
        stops=tuple(sorted(b.service_id.unique()))
        if len(stops)>3:return ()
        result=[]
        # 同期限站优先；所有排列仍完整评估。
        for g,t in self.types.iterrows():
            if mass>t.max_payload_kg+1e-9 or vol>t.capacity_m3+1e-12:continue
            for order in permutations(stops):
                remain=mass; prev='O01'; energy=0.; clock=float(t.prepare_s+n*t.load_per_box_s); offsets={}
                for s in order:
                    e,ft=self.segment(g,prev,s,remain); energy+=e; clock+=ft
                    delivered=b[b.service_id.eq(s)]
                    clock+=float(t.handover_base_s+len(delivered)*t.handover_per_box_s)
                    offsets[s]=clock; remain-=float(delivered.mass_kg.sum()); prev=s
                e,ft=self.segment(g,prev,'O01',0.);energy+=e;clock+=ft
                if energy<=(1-RHO)*t.energy_kwh+1e-9:
                    result.append(Variant(g,order,energy,clock,tuple((s,offsets[s]) for s in order),mass,vol,n,1-energy/t.energy_kwh))
        return tuple(result)

    def charge(self,g,soc):
        full=float(self.inv.loc[g,'full_charge_s'])
        return full*(.65*(.9-soc)/.9+.35) if soc<.9 else full*.35*(1-soc)/.1

    def route_priority(self,route):
        b=self.boxes.loc[list(route)]
        hard=b.hard_deadline_s.dropna()
        return (float(hard.min()) if len(hard) else math.inf,
                float(np.average(b.expected_s,weights=b.priority)), -float(b.priority.sum()))

    def schedule_order(self,routes,order):
        dav={r.aircraft_id:0. for r in self.fleet.itertuples()}
        bav={bid:0. for ids in self.batteries.values() for bid in ids}
        rows=[]; deliveries=[]
        for ridx in order:
            route=routes[ridx]; b=self.boxes.loc[list(route)]; options=[]
            for v in self.variants(route):
                drones=self.fleet[self.fleet.type_id.eq(v.type_id)].aircraft_id
                for drone in drones:
                    for bat in self.batteries[v.type_id]:
                        start=max(dav[drone],bav[bat]); ret=start+v.duration
                        completion={s:start+off for s,off in v.delivery_offsets}
                        hard_late=[]; wt=0.; wratio=0.; weighted=0.
                        for box,r in b.iterrows():
                            c=completion[r.service_id]; weighted+=r.priority
                            wratio+=r.priority*c/r.expected_s
                            wt+=r.priority*max(0.,c-r.expected_s)/r.expected_s
                            if pd.notna(r.hard_deadline_s):hard_late.append(max(0.,c-r.hard_deadline_s))
                        key=(sum(x>1e-7 for x in hard_late),max(hard_late or [0.]),wt,wratio/weighted,ret,v.energy,start)
                        options.append((key,v,drone,bat,start,ret,completion))
            if not options:return None
            _,v,drone,bat,start,ret,completion=min(options,key=lambda x:x[0])
            dav[drone]=ret; bav[bat]=ret+self.charge(v.type_id,v.min_soc)
            rows.append(dict(route_index=ridx,type_id=v.type_id,aircraft_id=drone,battery_id=bat,start_s=start,
                return_s=ret,energy_kwh=v.energy,return_soc=v.min_soc,mass_kg=v.mass,volume_m3=v.volume,
                box_count=v.box_count,visit_order='-'.join(v.order),duration_s=v.duration,
                battery_recharged_s=bav[bat],boxes=';'.join(sorted(route))))
            for box,r in b.iterrows():
                deliveries.append(dict(box_id=box,service_id=r.service_id,delivery_s=completion[r.service_id],
                    expected_s=r.expected_s,hard_deadline_s=None if pd.isna(r.hard_deadline_s) else r.hard_deadline_s,
                    priority=r.priority,type_id=v.type_id,aircraft_id=drone,battery_id=bat,route_index=ridx))
        return self.metrics(pd.DataFrame(rows),pd.DataFrame(deliveries))

    def metrics(self,trips,deliv):
        hard=deliv[deliv.hard_deadline_s.notna()]
        soft_late=np.maximum(deliv.delivery_s-deliv.expected_s,0.)
        weighted_tard=float((deliv.priority*soft_late/deliv.expected_s).sum())
        weighted_ratio=float((deliv.priority*deliv.delivery_s/deliv.expected_s).sum()/deliv.priority.sum())
        metrics=dict(hard_violations=int((hard.delivery_s>hard.hard_deadline_s+1e-7).sum()),
            max_hard_lateness_s=float(np.maximum(hard.delivery_s-hard.hard_deadline_s,0).max() if len(hard) else 0),
            weighted_tardiness=weighted_tard,weighted_delivery_ratio=weighted_ratio,
            on_time_boxes=int((deliv.delivery_s<=deliv.expected_s+1e-7).sum()),
            makespan_s=float(trips.return_s.max()),energy_kwh=float(trips.energy_kwh.sum()),trips=len(trips))
        return dict(trips=trips,deliveries=deliv,metrics=metrics)

    def score(self,m,policy):
        prefix=(m['hard_violations'],m['max_hard_lateness_s'])
        if policy=='timely':tail=(m['weighted_tardiness'],m['weighted_delivery_ratio'],m['makespan_s'],m['energy_kwh'],m['trips'])
        elif policy=='makespan':tail=(m['makespan_s'],m['weighted_tardiness'],m['weighted_delivery_ratio'],m['energy_kwh'],m['trips'])
        elif policy=='energy':tail=(m['energy_kwh'],m['weighted_tardiness'],m['makespan_s'],m['trips'])
        elif policy=='trips':tail=(m['trips'],m['weighted_tardiness'],m['makespan_s'],m['energy_kwh'])
        else:tail=(m['weighted_tardiness'],m['weighted_delivery_ratio'],m['makespan_s'],m['energy_kwh'],m['trips'])
        return prefix+tail

    def optimize_order(self,routes,policy,iterations=400,seed=0):
        n=len(routes); base=sorted(range(n),key=lambda i:self.route_priority(routes[i]))
        best=self.schedule_order(routes,base); best_order=base[:]
        rng=random.Random(seed+97*n+sum(len(r) for r in routes))
        # 随机键扰动 + 交换/插入邻域；所有结果确定性复现。
        candidates=[]
        for k in range(min(80,iterations//3)):
            noisy=sorted(range(n),key=lambda i:(self.route_priority(routes[i])[0],self.route_priority(routes[i])[1]*(.75+.5*rng.random()),rng.random()))
            candidates.append(noisy)
        current=best_order[:]
        for _ in range(iterations):
            cand=current[:]
            if rng.random()<.55:
                i,j=rng.sample(range(n),2);cand[i],cand[j]=cand[j],cand[i]
            else:
                i,j=rng.sample(range(n),2);x=cand.pop(i);cand.insert(j,x)
            candidates.append(cand)
            if rng.random()<.08:current=cand
        for order in candidates:
            sol=self.schedule_order(routes,order)
            if sol and self.score(sol['metrics'],policy)<self.score(best['metrics'],policy):best,best_order=sol,order[:];current=order[:]
        best['order']=best_order;return best

    def canonical(self,routes):return tuple(sorted((tuple(sorted(r)) for r in routes),key=lambda r:(len(r),r)))

    def merge_children(self,routes):
        result=[]
        for i in range(len(routes)):
            for j in range(i+1,len(routes)):
                merged=routes[i]|routes[j]
                if len(set(self.boxes.loc[list(merged)].service_id))>3 or not self.variants(merged):continue
                child=[r for k,r in enumerate(routes) if k not in (i,j)]+[merged]
                result.append(tuple(child))
        return result

    def search(self,beam_width=14,min_trips=11):
        level=[self.seed]; seen={self.canonical(self.seed)}; records=[]
        for depth in range(len(self.seed)-min_trips+1):
            scored=[]
            for ix,routes in enumerate(level):
                sol=self.optimize_order(routes,'balanced',0,depth*1000+ix)
                if sol:
                    records.append(dict(routes=tuple(routes),solution=sol))
                    m=sol['metrics']; key=(m['hard_violations'],m['max_hard_lateness_s'],m['weighted_tardiness'],m['weighted_delivery_ratio'],m['makespan_s']/10000+m['energy_kwh']/100+len(routes)/20)
                    scored.append((key,routes))
            if not scored:break
            print('搜索层',depth,'架次',len(level[0]),'状态',len(level),'最佳',min(x[0] for x in scored),flush=True)
            children=[]
            for _,routes in sorted(scored,key=lambda x:x[0])[:beam_width]:
                children.extend(self.merge_children(routes))
            next_level=[]
            for r in children:
                key=self.canonical(r)
                if key not in seen:seen.add(key);next_level.append(r)
            if not next_level:break
            # 快速静态指标保留多样性，精调在下一层完成。
            def quick(rs):
                e=sum(min(v.energy for v in self.variants(r)) for r in rs)
                return (len(rs),e,sum(min(v.duration for v in self.variants(r)) for r in rs))
            next_level.sort(key=quick); level=next_level[:beam_width]
        # 对硬时限可行的非重复路线集合加强顺序搜索，收集候选。
        feasible=[]; used=set()
        # 每个可行架次数层保留一个代表，确保综合比较同时覆盖“最少架次”边界。
        best_by_trips={}
        for rec in records:
            if rec['solution']['metrics']['hard_violations']!=0:continue
            n=len(rec['routes'])
            if n not in best_by_trips or self.score(rec['solution']['metrics'],'balanced') < self.score(best_by_trips[n]['solution']['metrics'],'balanced'):
                best_by_trips[n]=rec
        target_counts=sorted(best_by_trips)
        if len(target_counts)>6:
            picks={target_counts[0],target_counts[-1]}
            picks.update(target_counts[round(i*(len(target_counts)-1)/5)] for i in range(1,5))
            target_counts=sorted(picks)
        chosen=[best_by_trips[n] for n in target_counts]
        for rec in chosen:
            key=self.canonical(rec['routes'])
            if key in used:continue
            used.add(key)
            for policy in POLICIES[:-1]:
                feasible.append(dict(routes=rec['routes'],policy=policy,solution=self.optimize_order(rec['routes'],policy,15,len(used)*31)))
        return feasible

def nondominated(items):
    keys=['weighted_tardiness','weighted_delivery_ratio','makespan_s','energy_kwh','trips']; out=[]
    for i,x in enumerate(items):
        a=x['solution']['metrics']
        dominated=False
        for j,y in enumerate(items):
            if i==j:continue
            b=y['solution']['metrics']
            if all(b[k]<=a[k]+1e-9 for k in keys) and any(b[k]<a[k]-1e-9 for k in keys):dominated=True;break
        if not dominated:out.append(x)
    return out

def run():
    started=time.time(); model=Q2(); candidates=model.search()
    feasible=[x for x in candidates if x['solution']['metrics']['hard_violations']==0]
    if not feasible:raise RuntimeError('没有找到硬时限可行方案')
    pareto=nondominated(feasible)
    keys=['weighted_delivery_ratio','makespan_s','energy_kwh','trips']
    vals=np.array([[x['solution']['metrics'][k] for k in keys] for x in pareto],float)
    ideal=vals.min(0);nadir=vals.max(0);span=np.where(nadir>ideal,nadir-ideal,1.)
    dist=np.sqrt((((vals-ideal)/span)**2).mean(1))
    main=pareto[int(np.argmin(dist))]
    selected={'balanced':main}
    for pol in ['timely','makespan','energy','trips']:
        selected[pol]=min(feasible,key=lambda x:model.score(x['solution']['metrics'],pol))
    all_rows=[]
    for i,x in enumerate(pareto):
        m=x['solution']['metrics'];all_rows.append(dict(candidate_id=f'P{i+1:03d}',ideal_distance=dist[i],**m))
    save(pd.DataFrame(all_rows),'pareto_candidates')
    comparison=[]
    for pol,x in selected.items():
        m=x['solution']['metrics'];comparison.append(dict(policy=pol,**m))
        trips=x['solution']['trips'].copy().sort_values(['start_s','aircraft_id']).reset_index(drop=True)
        trips['flight_id']=[f'Q2-{i:03d}' for i in range(1,len(trips)+1)]
        mapping=dict(zip(trips.route_index,trips.flight_id))
        deliveries=x['solution']['deliveries'].copy();deliveries['flight_id']=deliveries.route_index.map(mapping)
        deliveries=deliveries.sort_values(['delivery_s','box_id']).reset_index(drop=True)
        save(trips,'plan_'+pol);save(deliveries,'deliveries_'+pol)
    comp=pd.DataFrame(comparison);save(comp,'objective_comparison')
    # 主方案专用资源时间线和提交模板。
    trips=pd.read_csv(TABLE/'plan_balanced.csv');deliveries=pd.read_csv(TABLE/'deliveries_balanced.csv')
    q2trips=trips[['flight_id','aircraft_id','type_id','battery_id','start_s','visit_order','return_s','energy_kwh']].copy()
    q2trips.columns=['架次编号','无人机编号','机型编号','电池编号','开始时刻（s）','访问服务区顺序','返回O01时刻（s）','架次能耗（kWh）']
    save(q2trips,'Q2_运输架次')
    q2boxes=deliveries[['box_id','flight_id','service_id','delivery_s']].copy();q2boxes.columns=['货箱编号','架次编号','服务区编号','交付完成时刻（s）'];save(q2boxes,'Q2_逐箱交付')
    usage=[]
    for r in trips.itertuples():
        usage.append(dict(resource_type='aircraft',resource_id=r.aircraft_id,type_id=r.type_id,flight_id=r.flight_id,
            occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.return_s,start_soc=1.,end_soc=r.return_soc))
        usage.append(dict(resource_type='battery',resource_id=r.battery_id,type_id=r.type_id,flight_id=r.flight_id,
            occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.battery_recharged_s,start_soc=1.,end_soc=r.return_soc))
    save(pd.DataFrame(usage),'resource_usage')
    legs=[]
    for r in trips.itertuples():
        b=model.boxes.loc[r.boxes.split(';')];remain=float(b.mass_kg.sum());prev='O01';clock=r.start_s+float(model.types.loc[r.type_id,'prepare_s']+len(b)*model.types.loc[r.type_id,'load_per_box_s'])
        for seq,s in enumerate(r.visit_order.split('-')+['O01'],1):
            e,ft=model.segment(r.type_id,prev,s,remain);arrive=clock+ft
            delivered=b[b.service_id.eq(s)] if s!='O01' else b.iloc[0:0]
            hand=0. if s=='O01' else float(model.types.loc[r.type_id,'handover_base_s']+len(delivered)*model.types.loc[r.type_id,'handover_per_box_s'])
            depart=arrive+hand
            legs.append(dict(flight_id=r.flight_id,leg_sequence=seq,from_id=prev,to_id=s,load_kg=remain,flight_s=ft,energy_kwh=e,arrival_s=arrive,handover_s=hand,departure_s=depart,delivered_boxes=';'.join(delivered.index)))
            remain-=float(delivered.mass_kg.sum());clock=depart;prev=s
    save(pd.DataFrame(legs),'route_legs')
    stats=dict(main_metrics=selected['balanced']['solution']['metrics'],pareto_count=len(pareto),candidate_count=len(feasible),
        ideal=dict(zip(keys,ideal.tolist())),nadir=dict(zip(keys,nadir.tolist())),elapsed_s=time.time()-started,
        method='beam route merging + deterministic neighborhood resource scheduling',seed_routes=len(model.seed))
    dump(stats,OUT/'solver_summary.json')
    inputs=[SRC/'boxes.csv',SRC/'transport_types.csv',SRC/'transport_fleet.csv',SRC/'transport_battery_inventory.csv',SRC/'arcs.csv',ROOT/'q1_results/tables/plan_N_E_T.csv']
    dump([dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],OUT/'input_manifest.json')
    build_report(model,trips,deliveries,comp,stats)
    workbook={name:dict(columns=list(df.columns),rows=json.loads(df.to_json(orient='values',force_ascii=False))) for name,df in [
        ('Q2_运输架次',q2trips),('Q2_逐箱交付',q2boxes),('资源使用',pd.DataFrame(usage)),('路线分段',pd.DataFrame(legs)),('目标对照',comp)]}
    dump(workbook,OUT/'workbook_data.json')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

def mdtable(df):
    cols=list(df.columns); lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for row in df.itertuples(index=False,name=None):
        vals=[]
        for x in row:
            if pd.isna(x):v=''
            elif isinstance(x,(float,np.floating)):v=f'{x:.6f}'
            else:v=str(x)
            vals.append(v.replace('|','/'))
        lines.append('|'+'|'.join(vals)+'|')
    return '\n'.join(lines)

def build_report(model,trips,deliveries,comp,stats):
    m=stats['main_metrics']; hard=deliveries[deliveries.hard_deadline_s.notna()]
    text=f'''# 问题二：异构无人机多点多架次运输调度

## 1. 建模范围与计算口径

本问允许一个架次访问一个或多个服务区，联合决定货箱组批、访问顺序、机型、具体无人机、共享电池和开始时刻。全部货箱不可拆分且只能交付一次。医疗物资期望时间、首批保障截止时间作为硬约束；其他物资的期望时间用于评价及时性。

沿用问题一的航段地形、时间与能量模型。多点路径p=(O01,i1,...,ik,O01)中，q_l为第l段起飞时的剩余载荷：

\[
q_1=\sum_{{b\in B_p}}w_b,\qquad q_{{l+1}}=q_l-\sum_{{b\text{{在}}i_l\text{{交付}}}}w_b.
\]

每段能耗为

\[
E_{{gij}}(q)=E_gd_{{ij}}/L_g(q)+(m_g+q)g_0h^+_{{ij}}/(3.6\times10^6\eta_g),
\]

总能耗为各段之和，并满足E_p≤(1−ρ_g)E_g。每次交付后重新从服务区离地30m作业高度爬升。下降耗时计入，下降附加能耗取0。水平耗能和势能爬升式是与问题一一致的推导约定。

任务开始时刻从准备开始计。起飞前完成固定准备和全部货箱装载；服务区的交付完成时刻为到达后完成基础交接及逐箱交接的时刻；返航时刻为最后一段下降至O01后的时刻。

## 2. 资源约束模型

若架次p分配给实体飞机u和电池r，开始、返航分别为s_p、f_p，则同一飞机的任务区间[s_p,f_p)不得重叠。同一电池也不得重叠，并且任务后SOC为1−E_p/E_g，再次使用前必须充至100%。充电完成时刻为

\[
a_r=f_p+t_{{chg}}(1-E_p/E_g).
\]

同型电池可跨同型飞机共享，不同机型不可混用。库存A/B/C型电池分别为6/4/4组，其中包含初始装机电池。题目未给充电桩数量上限，因此不同电池并行充电。

## 3. 多目标函数

对货箱b，交付时刻C_b、期望时间d_b、优先系数w_b。硬时限H_b存在时要求C_b≤H_b。及时性同时报告

\[
J_{{late}}=\sum_b w_b\max(0,C_b-d_b)/d_b,
\quad
J_{{ratio}}=\frac{{\sum_bw_bC_b/d_b}}{{\sum_bw_b}}.
\]

J_late强调超期程度，J_ratio在全部按期时仍能区分更早交付。全部任务完成时间为所有运输无人机最后返航时刻T_max=max_p f_p，另计总能耗和架次数。

求解中先强制硬时限，然后收集(J_late,J_ratio,T_max,E,N)非支配解。主方案从非支配集中，按J_ratio、T_max、E、N分别以候选集理想值和最差值归一化，选取四项等权平方距离最小的折中点。该规则避免把秒、kWh和架次直接相加。另给出及时性、完工时间、能耗及架次数优先方案。

## 4. 求解方法

以问题一18个已核验单点箱组作为初始可行模式，保证80个原始货箱ID不重不漏。对任意两条路线尝试合并，枚举合并后不超过3个服务区的所有访问顺序和A/B/C机型，逐段按剩余载荷检查质量、体积、能量及返航SOC。使用有限宽度束搜索逐层减少架次。

每个路线集合进入离散事件调度器。调度器记录8架飞机和14组电池的下一可用时刻，为候选任务选择可用的同型飞机和满电电池，并更新返航与充电完成时刻。对任务优先序进行固定随机种子的交换、插入和期限扰动搜索。全部结果可复现，但该组合问题采用启发式全局搜索，因此“非支配”与“理想点”均限于已搜索候选集；报告不把它误称为全局最优证明。

## 5. 主方案结果

主方案共{m['trips']}架次，全部80箱交付；硬时限违反{m['hard_violations']}箱；期望时间内交付{m['on_time_boxes']}箱。加权迟到指标{m['weighted_tardiness']:.6f}，加权交付比例{m['weighted_delivery_ratio']:.6f}。最后返航时刻为{m['makespan_s']:.6f}s（{m['makespan_s']/3600:.6f}h），总运输能耗{m['energy_kwh']:.6f}kWh。

最晚的硬时限交付裕量为{float((hard.hard_deadline_s-hard.delivery_s).min()):.6f}s；最低返航SOC为{trips.return_soc.min()*100:.6f}%。使用实体飞机{trips.aircraft_id.nunique()}架，使用不同电池{trips.battery_id.nunique()}组。

### 5.1 运输路线与架次安排

'''
    route=trips[['flight_id','aircraft_id','type_id','battery_id','start_s','visit_order','return_s','energy_kwh','return_soc','boxes']].copy()
    route.columns=['架次','无人机','机型','电池','开始/s','访问顺序','返航/s','能耗/kWh','返航SOC','货箱编号']
    text+=mdtable(route)+'\n\n### 5.2 逐箱交付时刻\n\n完整80行见tables/Q2_逐箱交付.csv和结果工作簿。按服务区汇总如下：\n\n'
    ds=deliveries.groupby('service_id').agg(boxes=('box_id','size'),first_delivery_s=('delivery_s','min'),last_delivery_s=('delivery_s','max'),expected_late_boxes=('delivery_s',lambda x:0)).reset_index()
    late=deliveries.assign(late=deliveries.delivery_s>deliveries.expected_s+1e-7).groupby('service_id').late.sum()
    ds['expected_late_boxes']=ds.service_id.map(late).astype(int)
    text+=mdtable(ds.rename(columns={'service_id':'服务区','boxes':'箱数','first_delivery_s':'最早交付/s','last_delivery_s':'最晚交付/s','expected_late_boxes':'超过期望箱数'}))
    text+='\n\n## 6. 目标权衡\n\n'+mdtable(comp.rename(columns={'policy':'方案','weighted_tardiness':'加权迟到','weighted_delivery_ratio':'加权交付比例','makespan_s':'最后返航/s','energy_kwh':'能耗/kWh','trips':'架次','on_time_boxes':'期望内箱数'}))
    text+='''

各单指标方案展示实际取舍：及时性优先通常把高优先系数和较早期望时间的箱子放入更早波次；完工时间优先倾向均衡8架实体飞机的工作量；能耗优先更愿意合并顺路服务区，但可能拉长单条路线和延后部分交付；架次数优先可能使用载荷更大的C型和更长的多点路径。主方案取候选非支配集的归一化折中点。若若干方案数值相同，说明搜索到的同一调度同时达到这些单项最好值，不人为制造差异。

## 7. 资源可行性

`resource_usage.csv`逐次记录飞机和电池占用区间、返航SOC及重新可用时刻。验证程序将按原始航段和箱ID独立重算：

1. 80个货箱恰出现一次，送达服务区正确；
2. 每架次初始质量和体积不超限，各航段剩余载荷递减；
3. 总能耗不超过80%可用电量，返航SOC不少于20%；
4. 同一实体飞机任务区间不重叠；
5. 同一电池占用不重叠，重复使用前已按两阶段模型充满；
6. 医疗期望时间和首批截止时间全部满足；
7. 最晚返航时刻与题目完工时间定义一致。

## 8. 输出文件与适用范围

- `tables/Q2_运输架次.csv`与`Q2_逐箱交付.csv`对应提交模板；
- `route_legs.csv`给出每段起终点、剩余载荷、到达、交接、离开和能耗；
- `resource_usage.csv`给出飞机与电池占用及充电完成时刻；
- `objective_comparison.csv`和`pareto_candidates.csv`说明多目标权衡；
- `问题二结果.xlsx`汇总可提交结果与核验说明。

本方案不考虑通信保障，符合问题二范围。风、临时禁飞、装卸工位数量和充电桩数量在附件中没有给定，未自行添加。若正式论文采用不同的能耗推导或任务开始口径，必须重新求解，不能只修改文字。
'''
    (OUT/'问题二_模型建立与求解.md').write_text(text,encoding='utf-8')

if __name__=='__main__':run()

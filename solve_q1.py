"""问题一：完整模式枚举 + 精确数量状态DP。只读preprocessing输入。"""
from pathlib import Path
from dataclasses import dataclass, asdict
from itertools import product
import hashlib
import json
import math
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'q1_results'
TABLE=OUT/'tables'
MATERIALS=['医疗物资','饮用水','应急食品','生活卫生用品']
COUNT_FIELDS=['medical_count','water_count','food_count','hygiene_count']
G0=9.80665
TOL_E=1e-9
TOL_T=1e-7
RHO=0.20
POLICIES={'N_E_T':(0,1,2),'E_N_T':(1,0,2),'T_N_E':(2,0,1),'N_T_E':(0,2,1)}
POLICY_CN={'N_E_T':'架次→能耗→时间（主方案）','E_N_T':'能耗→架次→时间','T_N_E':'时间→架次→能耗','N_T_E':'架次→时间→能耗'}

def write_csv(df,name):
    TABLE.mkdir(parents=True,exist_ok=True)
    df.to_csv(TABLE/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.12g')

def dump(obj,path):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

@dataclass(frozen=True)
class Pattern:
    pattern_id:str
    service_id:str
    type_id:str
    counts:tuple
    box_count:int
    mass_kg:float
    volume_m3:float
    energy_kwh:float
    flight_s:float
    operation_s:float
    return_soc:float
    rho_max:float

class Problem:
    def __init__(self):
        source=ROOT/'preprocessing'/'tables'
        self.boxes=pd.read_csv(source/'boxes.csv')
        self.types=pd.read_csv(source/'transport_types.csv').set_index('type_id')
        self.arcs=pd.read_csv(source/'arcs.csv').set_index(['from_id','to_id'])
        self.services=sorted(self.boxes.service_id.unique())
        self.physics={}
        self.patterns={}; self.by_id={}; self.states={}; self.candidates={}; self.demands={}
        for service in self.services:
            outbound=self.arcs.loc[('O01',service)]; inbound=self.arcs.loc[(service,'O01')]
            for typ,g in self.types.iterrows():
                flight=(outbound.climb_m+inbound.climb_m)/g.up_mps+(outbound.distance_m+inbound.distance_m)/g.cruise_mps+(outbound.descent_m+inbound.descent_m)/g.down_mps
                self.physics[(service,typ)]=dict(distance_out_m=float(outbound.distance_m),distance_back_m=float(inbound.distance_m),
                    climb_out_m=float(outbound.climb_m),climb_back_m=float(inbound.climb_m),
                    descent_out_m=float(outbound.descent_m),descent_back_m=float(inbound.descent_m),
                    cruise_alt_m=float(outbound.cruise_alt_m),flight_s=float(flight))
            b=self.boxes[self.boxes.service_id.eq(service)]
            demand=tuple(int(b.material.eq(m).sum()) for m in MATERIALS)
            self.demands[service]=demand
            states=sorted(product(*(range(n+1) for n in demand)),key=lambda s:(sum(s),s))
            self.states[service]=states
            masses=[]; volumes=[]
            for mat in MATERIALS:
                bm=b[b.material.eq(mat)]
                if len(bm):
                    assert bm.mass_kg.nunique()==bm.volume_m3.nunique()==1
                    masses.append(float(bm.mass_kg.iloc[0])); volumes.append(float(bm.volume_m3.iloc[0]))
                else: masses.append(0.); volumes.append(0.)
            patterns=[]
            for ix,c in enumerate(states[1:],1):
                q=sum(x*y for x,y in zip(c,masses)); v=sum(x*y for x,y in zip(c,volumes)); n=sum(c)
                for typ,g in self.types.iterrows():
                    if q>g.max_payload_kg+1e-10 or v>g.capacity_m3+1e-12: continue
                    e=self.energy(service,typ,q)
                    crit=1-e/g.energy_kwh
                    if crit < -1e-12: continue
                    f=self.physics[(service,typ)]['flight_s']
                    op=g.prepare_s+n*g.load_per_box_s+f+g.handover_base_s+n*g.handover_per_box_s
                    p=Pattern(f'{service}-{typ}-{ix:03d}',service,typ,c,n,q,v,float(e),f,float(op),float(crit),float(crit))
                    patterns.append(p); self.by_id[p.pattern_id]=p
            self.patterns[service]=patterns
            self.candidates[service]={state:[p for p in patterns if all(p.counts[k]<=state[k] for k in range(4))] for state in states}

    def energy_parts(self,service,typ,q):
        g=self.types.loc[typ]; h=self.physics[(service,typ)]
        length=g.range_empty_m-(g.range_empty_m-g.range_full_m)*(q/g.max_payload_kg)**1.5
        return dict(out_horizontal=g.energy_kwh*h['distance_out_m']/length,
                    out_climb=(g.empty_mass_kg+q)*G0*h['climb_out_m']/(g.up_efficiency*3.6e6),
                    back_horizontal=g.energy_kwh*h['distance_back_m']/g.range_empty_m,
                    back_climb=g.empty_mass_kg*G0*h['climb_back_m']/(g.up_efficiency*3.6e6))

    def energy(self,service,typ,q):
        return sum(self.energy_parts(service,typ,q).values())

    def capacity(self,service,typ,rho):
        g=self.types.loc[typ]; budget=(1-rho)*g.energy_kwh
        e0=self.energy(service,typ,0.); ef=self.energy(service,typ,g.max_payload_kg)
        if e0>budget+1e-12: q=None; status='empty_roundtrip_infeasible'
        elif ef<=budget+1e-12: q=float(g.max_payload_kg); status='rated_payload_limited'
        else:
            lo=0.; hi=float(g.max_payload_kg)
            for _ in range(70):
                mid=(lo+hi)/2
                if self.energy(service,typ,mid)<=budget: lo=mid
                else: hi=mid
            q=lo; status='energy_limited'
        return dict(service_id=service,type_id=typ,reserve_fraction=float(rho),max_safe_payload_kg=q,status=status,
            energy_budget_kwh=float(budget),energy_empty_kwh=float(e0),energy_at_rated_payload_kwh=float(ef),
            energy_at_safe_payload_kwh=None if q is None else float(self.energy(service,typ,q)),
            rated_payload_kg=float(g.max_payload_kg),**self.physics[(service,typ)])

    @staticmethod
    def better(a,b,order):
        if b is None: return True
        tolerances=(0,TOL_E,TOL_T)
        for k in order:
            if a[k]<b[k]-tolerances[k]: return True
            if a[k]>b[k]+tolerances[k]: return False
        return False

    def solve_site(self,service,rho,policy='N_E_T'):
        zero=(0,0,0,0); costs={zero:(0,0.,0.)}; paths={zero:()}
        for s in self.states[service][1:]:
            best=None; path=None
            for p in self.candidates[service][s]:
                if p.rho_max+1e-12<rho: continue
                remain=tuple(s[k]-p.counts[k] for k in range(4))
                prior=costs.get(remain)
                if prior is None: continue
                trial=(prior[0]+1,prior[1]+p.energy_kwh,prior[2]+p.operation_s)
                if self.better(trial,best,POLICIES[policy]):
                    best=trial; path=paths[remain]+(p.pattern_id,)
            if best is not None: costs[s]=best; paths[s]=path
        d=self.demands[service]
        if d not in costs: return None
        return dict(metrics=costs[d],patterns=paths[d])

    def solve(self,rho,policy='N_E_T'):
        selected=[]; missing=[]
        for service in self.services:
            solution=self.solve_site(service,rho,policy)
            if solution is None: missing.append(service)
            else: selected.extend(solution['patterns'])
        if missing: return dict(feasible=False,unserved_services=missing,patterns=(),metrics=None)
        metrics=(len(selected),sum(self.by_id[p].energy_kwh for p in selected),sum(self.by_id[p].operation_s for p in selected))
        return dict(feasible=True,unserved_services=[],patterns=tuple(selected),metrics=metrics)

    def materialize(self,ids,rho,scenario):
        """同物性箱可交换；确定性还原为原始箱ID，不声称这就是时间调度。"""
        pools={(s,m):sorted(self.boxes.loc[self.boxes.service_id.eq(s)&self.boxes.material.eq(m),'box_id']) for s in self.services for m in MATERIALS}
        trips=[]; assignments=[]
        for j,pid in enumerate(sorted(ids,key=lambda pid:(self.by_id[pid].service_id,self.by_id[pid].type_id,pid)),1):
            p=self.by_id[pid]; g=self.types.loc[p.type_id]
            assigned=[]
            for mat,n in zip(MATERIALS,p.counts):
                taken=pools[(p.service_id,mat)][:n]; pools[(p.service_id,mat)]=pools[(p.service_id,mat)][n:]
                assigned+=taken
            flight_id=f'Q1-{j:03d}'
            cap=self.capacity(p.service_id,p.type_id,rho)['max_safe_payload_kg']
            parts=self.energy_parts(p.service_id,p.type_id,p.mass_kg)
            trips.append(dict(scenario=scenario,flight_id=flight_id,service_id=p.service_id,type_id=p.type_id,
                 pattern_id=pid,box_ids=';'.join(assigned),box_count=p.box_count,mass_kg=p.mass_kg,volume_m3=p.volume_m3,
                 flight_s=p.flight_s,prepare_s=float(g.prepare_s),loading_s=p.box_count*float(g.load_per_box_s),
                 handover_s=float(g.handover_base_s+p.box_count*g.handover_per_box_s),operation_s=p.operation_s,
                 energy_kwh=p.energy_kwh,return_soc=p.return_soc,reserve_fraction=rho,max_safe_payload_kg=cap,
                 rated_mass_utilization=p.mass_kg/g.max_payload_kg,safe_mass_utilization=p.mass_kg/cap,
                 volume_utilization=p.volume_m3/g.capacity_m3,energy_budget_utilization=p.energy_kwh/((1-rho)*g.energy_kwh),
                 **{k+'_kwh':float(v) for k,v in parts.items()},**dict(zip(COUNT_FIELDS,p.counts))))
            assignments.extend([dict(scenario=scenario,box_id=bid,flight_id=flight_id,service_id=p.service_id,type_id=p.type_id) for bid in assigned])
        assert all(not pool for pool in pools.values())
        return pd.DataFrame(trips),pd.DataFrame(assignments)

    def site_pareto(self,service,rho):
        """固定每个数量状态的最少架次，保留E/T全部非支配标签。"""
        zero=(0,0,0,0); counts={zero:0}; fronts={zero:[(0.,0.,())]}
        for s in self.states[service][1:]:
            options=[]; min_n=10**6
            for p in self.candidates[service][s]:
                if p.rho_max+1e-12<rho: continue
                rem=tuple(s[k]-p.counts[k] for k in range(4))
                if rem not in counts: continue
                n=counts[rem]+1
                if n<min_n: min_n=n; options=[]
                if n==min_n:
                    options.extend((e+p.energy_kwh,t+p.operation_s,ids+(p.pattern_id,)) for e,t,ids in fronts[rem])
            if options: counts[s]=min_n; fronts[s]=pareto_prune(options)
        return fronts[self.demands[service]]

    def bottleneck_thresholds(self,service):
        """B(state,k)=max min(模式可承受余量,前序余量)，给出k架次可承受最大rho。"""
        zero=(0,0,0,0); values={zero:{0:1.}}
        for s in self.states[service][1:]:
            best={}
            for p in self.candidates[service][s]:
                rem=tuple(s[k]-p.counts[k] for k in range(4))
                for n,prior in values.get(rem,{}).items():
                    critical=min(prior,p.rho_max)
                    if critical>best.get(n+1,-math.inf): best[n+1]=critical
            values[s]=best
        return values[self.demands[service]]

def pareto_prune(labels):
    # 1e-10kWh量级只合并浮点求和误差；时间保留至1e-7s容差。
    labels=sorted(labels,key=lambda x:(round(x[0],10),x[1],x[2]))
    kept=[]; best_time=math.inf
    for label in labels:
        if label[1]<best_time-TOL_T:
            kept.append(label); best_time=label[1]
    return kept

def markdown_table(df, formats=None):
    cols=list(df.columns); lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for row in df.itertuples(index=False,name=None):
        cells=[]
        for col,value in zip(cols,row):
            if pd.isna(value): text='不可行'
            elif formats and col in formats: text=formats[col].format(value)
            elif isinstance(value,(float,np.floating)): text=f'{value:.6g}'
            else: text=str(value)
            cells.append(text.replace('|','/').replace('\n',' '))
        lines.append('|'+'|'.join(cells)+'|')
    return '\n'.join(lines)

def run():
    started=time.time(); TABLE.mkdir(parents=True,exist_ok=True)
    problem=Problem()
    print('枚举完成：', {s:len(p) for s,p in problem.patterns.items()},flush=True)
    capacities=pd.DataFrame([problem.capacity(s,g,RHO) for s in problem.services for g in problem.types.index])
    write_csv(capacities,'maximum_safe_payload')
    pattern_rows=[]
    for p in problem.by_id.values():
        r=asdict(p); r.pop('counts'); r.update(dict(zip(COUNT_FIELDS,p.counts))); pattern_rows.append(r)
    write_csv(pd.DataFrame(pattern_rows),'all_physical_patterns')
    comparisons=[]; policy_details={}; main_trips=None; main_assignments=None
    for policy in POLICIES:
        solution=problem.solve(RHO,policy); assert solution['feasible']
        n,e,t=solution['metrics']; trips,assignments=problem.materialize(solution['patterns'],RHO,policy)
        comparisons.append(dict(policy=policy,priority=POLICY_CN[policy],trips=n,energy_kwh=e,operation_s=t,
            operation_h=t/3600,flight_s=float(trips.flight_s.sum()),A_trips=int(trips.type_id.eq('A').sum()),
            B_trips=int(trips.type_id.eq('B').sum()),C_trips=int(trips.type_id.eq('C').sum())))
        write_csv(trips,'plan_'+policy); write_csv(assignments,'box_assignment_'+policy)
        policy_details[policy]=solution
        if policy=='N_E_T': main_trips=trips; main_assignments=assignments
    comparison=pd.DataFrame(comparisons); write_csv(comparison,'objective_comparison')
    main_summary=main_trips.groupby('service_id').agg(trips=('flight_id','count'),energy_kwh=('energy_kwh','sum'),
        operation_s=('operation_s','sum'),mass_kg=('mass_kg','sum'),volume_m3=('volume_m3','sum'),box_count=('box_count','sum'),
        min_return_soc=('return_soc','min')).reset_index()
    main_summary['types']=[','.join(main_trips.loc[main_trips.service_id.eq(s),'type_id']) for s in main_summary.service_id]
    write_csv(main_summary,'main_service_summary')
    template=main_trips[['flight_id','service_id','type_id','box_ids','mass_kg','volume_m3','operation_s','energy_kwh','return_soc']].copy()
    template['return_soc']*=100
    template.columns=['架次编号','服务区编号','机型编号','货箱编号列表','总质量（kg）','总体积（m³）','往返时间（s）','架次能耗（kWh）','返航SOC（%）']
    write_csv(template,'Q1_单点组批')
    print('主方案及三种对照已求解',flush=True)

    # 固定全局最少架次时的完整能耗—累计时间前沿，服务区之间做Minkowski加和后支配剪枝。
    frontier=[(0.,0.,())]; site_front_rows=[]
    for s in problem.services:
        sf=problem.site_pareto(s,RHO)
        site_front_rows += [dict(service_id=s,energy_kwh=e,operation_s=t,patterns=';'.join(ids)) for e,t,ids in sf]
        frontier=pareto_prune([(a[0]+b[0],a[1]+b[1],a[2]+b[2]) for a in frontier for b in sf])
    write_csv(pd.DataFrame(site_front_rows),'site_pareto_at_min_trips')
    pareto=pd.DataFrame([dict(point_id=f'P{i:03d}',trips=policy_details['N_E_T']['metrics'][0],energy_kwh=e,
        operation_s=t,operation_h=t/3600,patterns=';'.join(ids)) for i,(e,t,ids) in enumerate(frontier,1)])
    write_csv(pareto,'pareto_at_min_trips')
    print('最少架次Pareto点数：',len(frontier),flush=True)

    thresholds={s:problem.bottleneck_thresholds(s) for s in problem.services}
    threshold_rows=[dict(service_id=s,trips=k,max_reserve_fraction=v) for s,tab in thresholds.items() for k,v in sorted(tab.items())]
    write_csv(pd.DataFrame(threshold_rows),'exact_site_trip_thresholds')
    critical=min(max(tab.values()) for tab in thresholds.values())
    critical_services=[s for s,tab in thresholds.items() if abs(max(tab.values())-critical)<1e-10]
    breaks=sorted(set([0.,critical]+[v for tab in thresholds.values() for v in tab.values() if 0<v<critical]))
    plateaus=[]
    for lower,upper in zip(breaks[:-1],breaks[1:]):
        rho=(lower+upper)/2
        counts={s:min(k for k,v in tab.items() if v>=rho) for s,tab in thresholds.items()}
        total=sum(counts.values())
        if plateaus and plateaus[-1]['minimum_trips']==total:
            plateaus[-1]['upper_inclusive']=upper
        else: plateaus.append(dict(lower=lower,lower_inclusive=lower==0,upper_inclusive=upper,
                                   minimum_trips=total,counts=json.dumps(counts,ensure_ascii=False)))
    step=pd.DataFrame(plateaus); write_csv(step,'exact_global_trip_steps')

    sample_rhos=[0.,.1,.15,.2,.225,.25,.275,.3,.325,.35,.375,.4,.45,.5]
    sensitivity=[]; sens_trips=[]; sens_caps=[]; sens_groups=[]
    for rho in sample_rhos:
        sol=problem.solve(rho)
        n,e,t=sol['metrics'] if sol['feasible'] else (None,None,None)
        rec=dict(reserve_fraction=rho,feasible=sol['feasible'],trips=n,energy_kwh=e,operation_s=t,
                 operation_h=None if t is None else t/3600,unserved_services=';'.join(sol['unserved_services']))
        if sol['feasible']:
            detail,_=problem.materialize(sol['patterns'],rho,f'rho_{rho:.3f}'); sens_trips.append(detail)
            rec.update({g+'_trips':int(detail.type_id.eq(g).sum()) for g in problem.types.index})
            for s,df in detail.groupby('service_id'):
                sens_groups.append(dict(reserve_fraction=rho,service_id=s,trips=len(df),types=','.join(df.type_id),
                    masses_kg=';'.join(f'{x:g}' for x in df.mass_kg),energy_kwh=float(df.energy_kwh.sum()),
                    operation_s=float(df.operation_s.sum()),patterns=';'.join(df.pattern_id)))
        sensitivity.append(rec)
        sens_caps.extend(problem.capacity(s,g,rho) for s in problem.services for g in problem.types.index)
    sensitivity=pd.DataFrame(sensitivity); write_csv(sensitivity,'reserve_sensitivity')
    write_csv(pd.concat(sens_trips,ignore_index=True),'reserve_sensitivity_all_trips')
    write_csv(pd.DataFrame(sens_groups),'reserve_sensitivity_by_service')
    write_csv(pd.DataFrame(sens_caps),'reserve_sensitivity_capacities')
    dense_caps=pd.DataFrame([problem.capacity(s,g,float(rho)) for rho in np.linspace(0,.5,101) for s in problem.services for g in problem.types.index])
    write_csv(dense_caps,'reserve_capacity_curves')
    print('余量敏感性与精确阈值已求解',flush=True)
    stats=dict(baseline_reserve=RHO,main_metrics=list(policy_details['N_E_T']['metrics']),
        number_of_physical_patterns=len(problem.by_id),number_of_baseline_patterns=sum(p.rho_max>=RHO-1e-12 for p in problem.by_id.values()),
        state_counts={s:len(v) for s,v in problem.states.items()},pareto_points=len(frontier),
        largest_feasible_reserve=critical,limiting_services=critical_services,elapsed_seconds=time.time()-started,
        numerical_tolerances=dict(energy_kwh=TOL_E,time_s=TOL_T,feasibility_reserve_fraction=1e-12))
    dump(stats,OUT/'solver_summary.json')
    inputs=[ROOT/'preprocessing'/'tables'/name for name in ['boxes.csv','transport_types.csv','arcs.csv']]
    dump([dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],OUT/'input_manifest.json')
    write_report(problem,capacities,main_trips,main_summary,comparison,sensitivity,step,stats,pareto)
    # 供工作簿构建器使用，保持原生数字类型和空值语义。
    workbook={name:dict(columns=list(df.columns),rows=json.loads(df.to_json(orient='values',force_ascii=False))) for name,df in [
        ('Q1_单点组批',template),('最大安全载荷',capacities[['service_id','type_id','max_safe_payload_kg','status','energy_budget_kwh','energy_at_safe_payload_kwh']]),
        ('目标对照',comparison),('安全余量敏感性',sensitivity),('逐箱归属',main_assignments),('逐区汇总',main_summary)]}
    dump(workbook,OUT/'workbook_data.json')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


def write_report(problem,capacities,trips,summary,comparison,sensitivity,step,stats,pareto):
    main=comparison.iloc[0]; energy=comparison[comparison.policy.eq('E_N_T')].iloc[0]; speed=comparison[comparison.policy.eq('T_N_E')].iloc[0]
    cap=capacities.pivot(index='service_id',columns='type_id',values='max_safe_payload_kg').reset_index()
    cap.columns=['服务区','A型/kg','B型/kg','C型/kg']
    cap_text=markdown_table(cap,{c:'{:.4f}' for c in cap.columns[1:]})
    reduced=capacities[capacities.status.eq('energy_limited')]
    lines=[r'''# 问题一：单点往返运输能力与货箱组批优化

## 1. 问题范围与建模思路

本问每个架次只能执行O01→一个服务区→O01，80个货箱不可拆分、不可重复或漏运。15个服务区之间不允许混装。不限制实体无人机和共享电池的数量、并行关系或充电周转，也不在本问施加第二问的逐箱配送时限。

先由DEM和给定机型建立单点往返的时间与能量模型，计算每个“机型—服务区”组合的最大安全载荷；再完整枚举满足质量、体积和能量约束的箱组，采用有限状态动态规划精确求解组批问题。主方案采用**架次数→总运输能耗→累计作业时间**的字典序优先关系，并以其他目标顺序和Pareto前沿展示权衡。

### 1.1 数据与假设

1. 采用已预处理的节点、货箱、三类运输机型和DEM航段数据。货箱总数80，总质量758 kg，总体积2.011 m³。返航安全余量基准值为20%。
2. 给定地面海拔用于起降/作业高度，DEM用于沿线最高地形。两者不强制改为相同。航线采用以O01为原点的局部米制直线，与WGS84测地距离的最大相对差为0.05155%。
3. 固定速度、机型参数及题目航程—载荷公式，不另加入题目未给定的风雨折减、机体故障或充电资源约束。
4. 去程携带整批货物，交付后返程有效载荷为0。能耗采用题目“水平巡航＋爬升附加”的口径，下降附加能耗为0，但下降时间必须计入。
5. **明确的能耗推导约定**：由标准航程定义采用水平单位距离能耗Euse/L(q)，爬升采用势能除以效率。现有Markdown题面未完整展开两项表达式，以下两式属于沿用预处理的物理推导约定。数值结论以此口径为条件。
6. 未给出运输机交接悬停功率，且题面将架次运输能耗定义为各航段能耗之和，因此不额外虚构交接悬停能耗。累计作业时间仍完整包含交接。

### 1.2 符号

|符号|含义|
|---|---|
|i、g|服务区i，机型g∈{A,B,C}|
|B_i|服务区i的货箱集合|
|Q_g、V_g|标称最大载货质量与可用体积|
|m_g、E_g|含电池空载总质量与单组电池可用能量|
|L_g^0、L_g^F|空载与满载标准航程|
|ρ|返航安全余量比例|
|d_i、H_i|O01与服务区水平距离、沿线最高DEM高程加50 m|
|z_0、z_i|O01地面海拔、服务区给定地面海拔加30 m|
|q、v、n|某架次货箱总质量、总体积、箱数|
|E、T、N|总运输能耗、累计作业时间、往返架次数|

## 2. 单点往返时间与能量模型

### 2.1 航段地形与飞行时间

沿直线穿越的所有DEM闭像元取最大高程，边界接触像元也计入，防止漏掉山脊：

\[
H_i=\max_{u\in\mathcal C_i}Z_u+50,\qquad
a_i=H_i-z_0,\qquad b_i=H_i-z_i.
\]

去程爬升a_i、下降b_i；返程爬升b_i、下降a_i。于是往返纯飞行时间为

\[
\tau_{gi}=\frac{a_i+b_i}{v_g^{\uparrow}}+
\frac{2d_i}{v_g^c}+\frac{a_i+b_i}{v_g^{\downarrow}}.
\]

不同服务区地形不同，即使水平距离相同，也可能有不同的爬升和往返时间。

### 2.2 去程载货、返程空载的能耗

\[
L_g(q)=L_g^0-(L_g^0-L_g^F)(q/Q_g)^{3/2},\quad 0\le q\le Q_g.
\]

以kWh为能量单位，g_0=9.80665 m/s²，η_g为爬升效率：

\[
E_{gi}(q)=E_gd_i\left[\frac1{L_g(q)}+\frac1{L_g^0}\right]
+\frac{g_0}{3.6\times10^6\eta_g}
\left[(m_g+q)a_i+m_gb_i\right].
\]

第一项是去程载货和返程空载的水平能耗，第二项是两程爬升能耗。质量m_g已包含电池，不再重复加电池质量。可行性要求

\[
E_{gi}(q)\le(1-\rho)E_g,\qquad
SOC_{return}=1-E_{gi}(q)/E_g\ge\rho.
\]

安全余量约束施加于**完整往返能耗**，不是只为返程保留ρ倍的返程能耗。

### 2.3 最大安全载荷与求解原理

\[
q^{safe}_{gi}(\rho)=\max\{q:0\le q\le Q_g,\ E_{gi}(q)\le(1-\rho)E_g\}.
\]

记Δ_g=L_g^0−L_g^F>0，对q>0有

\[
E'_{gi}(q)=
\frac{E_gd_i}{L_g(q)^2}\frac{3\Delta_g}{2Q_g}\sqrt{q/Q_g}
+\frac{g_0a_i}{3.6\times10^6\eta_g}>0.
\]

因此能耗随载荷单调增加，可用二分法稳定求得唯一能量边界：

- E(0)>(1−ρ)E_g：连空载往返也不可行，标记“不可达”，不能把0 kg写成可行载荷。
- E(Q_g)≤(1−ρ)E_g：最大安全载荷就是标称Q_g。
- 其他情况：在[0,Q_g]二分求E(q)=(1−ρ)E_g，取可行侧下界。

程序进行70次二分。表中小数仅用于展示，组批可行性始终使用未四舍五入的能耗。**安全载荷是质量—能量上限；具体箱组还必须满足体积与不可拆约束，不能仅依据此表组批。**

## 3. 三种机型在15个服务区的最大安全载荷

以下均为ρ=20%的结果，单位kg。
''',cap_text,'',
        '能量约束生效的组合：'+ '；'.join(f'{r.service_id}-{r.type_id}：{r.max_safe_payload_kg:.4f} kg' for r in reduced.itertuples())+'。其他组合由标称载质量限制。',
        r'''
## 4. 不可拆货箱的组批优化模型

### 4.1 可行架次模式

对服务区i，模式p是“某一箱组＋一个机型g(p)”。设a_bp表示货箱b是否属于p，w_b、v_b分别为单箱质量和体积，则

\[
q_p=\sum_{b\in B_i}w_ba_{bp},\quad
v_p=\sum_{b\in B_i}v_ba_{bp},\quad
n_p=\sum_{b\in B_i}a_{bp}.
\]

只保留满足下式的模式集合P_i(ρ)：

\[
1\le n_p,\quad q_p\le Q_{g(p)},\quad v_p\le V_{g(p)},\quad
E_{g(p),i}(q_p)\le(1-\rho)E_{g(p)}.
\]

累计作业时间必须区别于纯飞行时间：

\[
t_p=t^{prep}_{g(p)}+n_pt^{load}_{g(p)}+\tau_{g(p),i}
+t^{handover,0}_{g(p)}+n_pt^{handover,box}_{g(p)}.
\]

本题A/B型每箱装载加交接共60 s，C型66 s；每架次固定准备300 s，基础交接A/B型150 s、C型180 s。

### 4.2 集合划分模型

对以原箱ID定义的模式，x_p∈{0,1}表示是否执行：

\[
\sum_{p\in P_i:\,b\in p}x_p=1,\qquad \forall b\in B_i,\ \forall i.
\]

这保证每箱恰好交付一次。三个目标为

\[
N=\sum_px_p,\qquad E=\sum_pE_px_p,\qquad T=\sum_pt_px_p.
\]

主方案为lex min(N,E,T)：先求N*，在N=N*下最小化E，再在前两者最优的方案中最小化T。无需把架次、kWh和秒未经归一化直接相加，也不会因随意权重导致多飞一架次只换来极小收益。

这一优先关系用于控制起降与交接次数，适合作为运输组织基准；它是明确的决策选择，不意味着能耗优先永远不合理。第6节给出不同偏好的精确对照。

### 4.3 按同物性货箱压缩的精确动态规划

第一问不施加时限，同服务区同类箱具有相同质量和体积，因此可按四类物资的数量向量c=(c_med,c_water,c_food,c_hyg)表示箱组。此压缩不跨服务区，也不合并不同质量或体积的箱子。原始箱ID在求解后逐一还原。

设D_i为服务区需求数量向量、a_p为模式p的数量向量。压缩后的等价整数模型是

\[
\sum_{p\in P_i}a_{pk}y_p=D_{ik},\quad y_p\in\mathbb Z_{\ge0},\quad k=1,2,3,4.
\]

模式可以重复，是因为相同物性的不同货箱可以组成相同数量的批次；还原ID后每个箱仍只出现一次。

令F_i(s)是恰好运完数量状态s时最优的(N,E,T)，则

\[
F_i(0)=(0,0,0),\quad
F_i(s)=\operatorname{lexmin}_{a_p\le s}
\{F_i(s-a_p)+(1,E_p,t_p)\}.
\]

按照状态中货箱总数递增计算，并保存回溯模式。每个非空解都包含某个最后架次，删去它必然落到较小状态，因此递推枚举了所有可行组批，不是贪心近似。

S001的需求向量为(2,8,3,2)，状态数仅(2+1)(8+1)(3+1)(2+1)=324；其他服务区更小。15个服务区没有跨区装载或公共资源耦合，目标可加，因此分别求得逐区字典序最优解后相加就是本问全局字典序最优解。
''',
        f'本次共枚举{stats["number_of_physical_patterns"]}个在ρ=0时物理可行的“数量模式—机型”组合，ρ=20%时其中{stats["number_of_baseline_patterns"]}个可用。',
        '', '## 5. 主方案数值结果','',
        f'**最少往返架次数N*={int(main.trips)}；在此架次数下最低运输能耗E*={main.energy_kwh:.9f} kWh；累计作业时间T*={main.operation_s:.6f} s={main.operation_h:.6f} h。**',
        f'机型使用架次：A型{int(main.A_trips)}、B型{int(main.B_trips)}、C型{int(main.C_trips)}。这些是架次，不是所需实体飞机数量。',
        f'累计纯飞行时间为{main.flight_s:.6f} s；准备、装载和交接共{main.operation_s-main.flight_s:.6f} s。最低返航SOC为{trips.return_soc.min()*100:.6f}%，全部架次满足20%下限。',
        '架次数的简单下界也可直接证明：每个非空服务区至少1架次，共15次；S001总质量154 kg，S002和S003各81 kg，均超过全部机型中的最大载重80 kg，因此这三个服务区各至少增加1次，得到N≥18。上述可行方案恰为18次，故最少架次数不依赖求解器声明即可证实。',
        '', '### 5.1 各服务区汇总','',
        markdown_table(summary.rename(columns={'service_id':'服务区','trips':'架次','energy_kwh':'能耗/kWh','operation_s':'累计作业/s','mass_kg':'质量/kg','volume_m3':'体积/m³','box_count':'箱数','min_return_soc':'最低SOC比例','types':'机型'})),
        '', '### 5.2 逐架次货箱组批','',
        '数量简记为“医、水、食、卫”；完整原始货箱编号保存在plan_N_E_T.csv及工作簿Q1_单点组批。下面的货箱ID表将80箱全部列出。','',
        markdown_table(pd.DataFrame([{'架次':r.flight_id,'服务区':r.service_id,'机型':r.type_id,
            '医/水/食/卫':f'{r.medical_count}/{r.water_count}/{r.food_count}/{r.hygiene_count}',
            '质量/kg':r.mass_kg,'体积/m³':r.volume_m3,'能耗/kWh':r.energy_kwh,
            '作业时间/s':r.operation_s,'返航SOC/%':r.return_soc*100} for r in trips.itertuples()])),
        '', markdown_table(trips[['flight_id','box_ids']].rename(columns={'flight_id':'架次','box_ids':'完整货箱编号'})),
        '', '## 6. 多目标权衡与优先关系','',
        markdown_table(comparison[['priority','trips','energy_kwh','operation_h','A_trips','B_trips','C_trips']].rename(columns={
            'priority':'目标优先顺序','trips':'架次','energy_kwh':'能耗/kWh','operation_h':'累计作业/h','A_trips':'A架次','B_trips':'B架次','C_trips':'C架次'})),
        '', f'相对主方案，能耗优先方案架次变化{int(energy.trips-main.trips):+d}，能耗变化{energy.energy_kwh-main.energy_kwh:+.6f} kWh（{(energy.energy_kwh/main.energy_kwh-1)*100:+.3f}%），累计作业时间变化{(energy.operation_s-main.operation_s)/60:+.3f} min。',
        f'时间优先方案架次变化{int(speed.trips-main.trips):+d}，能耗变化{speed.energy_kwh-main.energy_kwh:+.6f} kWh，累计作业时间变化{(speed.operation_s-main.operation_s)/60:+.3f} min。',
        '本数据的差异集中在S008：主方案用1架次C型装45 kg；纯能耗最优方案改为2架次B型，分别装20 kg和25 kg。其余服务区的最优目标值一致。因此主方案用额外约0.165%的能量换取少1架次和约27.73分钟累计作业时间，作为基准更合适。',
        r'''
从模型看，增加架次会重复固定准备、基础交接和空载返程，通常增加时间；但较少架次有时需使用能量更大的C型，故不保证最少架次同时最低能耗。即便同一机型，L(q)的非线性也使“尽量装满”不能代替能耗优化。

在固定N=N*时，对每个数量状态保留(E,T)全部非支配标签；逐区前沿作组合加和并删除被支配点，得到该条件下完整的E—T前沿。一个点若不存在另一个能耗和时间均不更差、且至少一项更好的点，就称为非支配点。此处完整前沿只针对**最少架次条件**，并未声称枚举所有架次数的三维Pareto集合。
''',
        f'本数据在最少{int(main.trips)}架次条件下共有{len(pareto)}个非支配目标点，详见pareto_at_min_trips.csv。若不同目标优先顺序落在同一点，应如实报告目标一致，不能人为制造权衡。',
        '', '## 7. 返航安全余量敏感性','',
        r'''
### 7.1 对最大安全载荷的影响

当机型仍受标称载质量限制时，安全余量小幅增加可能不改变q_safe；一旦能量约束成为有效边界，隐函数求导给出

\[
\frac{dq^{safe}_{gi}}{d\rho}=-\frac{E_g}{E'_{gi}(q^{safe}_{gi})}<0.
\]

故最大安全载荷关于余量单调不增，但不同距离、地形和机型的变化速度不同。连续载荷下降不必立即改变整数箱组：只有某个具体箱组跨过可行性边界，方案才可能改变。

### 7.2 对组批与目标值的影响

对完整模式p，可承受的最大安全余量为

\[
\bar\rho_p=1-E_p/E_{g(p)}.
\]

当ρ≤ρ̄_p时模式可行，超过阈值后不可行。因此最少架次数关于ρ是单调不减的阶梯函数；即使架次不变，机型和箱组也可能已变化。需要区分可行域缩小与不同主目标下的能耗变化：**纯能耗最优值**在可行域缩小时不可能下降，但字典序(N,E,T)在N发生跳变后，所选方案的E不必单调。

以下每个情景均从头精确重优化，而非只检查原方案。不可行表示至少一个服务区无法用三种机型完成全部货箱，不给出部分交付伪装为完整方案。
''',
        markdown_table(sensitivity[['reserve_fraction','feasible','trips','energy_kwh','operation_h','A_trips','B_trips','C_trips','unserved_services']].rename(columns={
            'reserve_fraction':'余量比例','feasible':'全部可行','trips':'架次','energy_kwh':'能耗/kWh','operation_h':'累计作业/h',
            'A_trips':'A架次','B_trips':'B架次','C_trips':'C架次','unserved_services':'无法完成的服务区'})),
        '', '### 7.3 精确架次变化阈值','',
        r'''离散情景可能漏掉很窄的变化区间，因此另用最大瓶颈动态规划。令B_i(s,k)为使用恰好k架次完成状态s时可支持的最大公共余量：

\[
B_i(0,0)=1,\quad
B_i(s,k)=\max_{a_p\le s}\min\{\bar\rho_p,B_i(s-a_p,k-1)\}.
\]

由此n_i(ρ)=min{k:B_i(D_i,k)≥ρ}，全局N*(ρ)=Σ_i n_i(ρ)。此方法给出模式切换引起的**精确阈值**，精确性限于给定物理数据与浮点数值精度。区间端点等号仍属于较少架次一侧。
''',
        markdown_table(pd.DataFrame([{'余量区间':f'{"[" if r.lower_inclusive else "("}{r.lower*100:.8f}%, {r.upper_inclusive*100:.8f}%]','最少架次':r.minimum_trips} for r in step.itertuples()])),
        '', f'完整交付可支持的最大公共安全余量为**{stats["largest_feasible_reserve"]*100:.8f}%**，瓶颈服务区为'+','.join(stats['limiting_services'])+'；超过该值，即使拆为单箱架次也不能完成全部任务。',
        '瓶颈具体是S008的14 kg饮用水箱：即便由该单箱任务余量表现最好的C型单独运输，其返航SOC也仅35.26764735%。在无限实体飞机/电池的本问中，所有箱单独可运是完整交付可行的充分条件；能耗随载荷单调递增，又使任何单箱在所有机型上不可运成为整体不可行的必要证据。',
        '主方案的最小返航SOC为S004的23.08924839%，因此将安全余量从20%提高到该值时，原18架次方案仍可行且仍最优。超过此值S004至少需2次，总架次增为19；超过27.04974165%后S008也不能一架次完成，总数至少20。25%至32.5%的详细表还显示：架次数不变时，机型和装载分配仍会改变。',
        '', '## 8. 可行性、最优性核验与交付文件','',
        '独立验证程序verify_q1.py不使用输出的箱组能耗作为依据，而按每个箱ID、机型和航段重算质量、体积、载荷航程、能耗、作业时间和SOC；检查80箱恰出现一次。',
        '最优性采用两条证据：完整状态递推的归纳证明，以及逐区独立整数规划交叉求解。MILP首先最小化架次，固定最少架次后最小化能耗，与动态规划数值对比；实际求解状态和差值见verification.json及milp_crosscheck.csv。',
        '浮点比较容差：能耗1e−9 kWh、时间1e−7 s；MILP数值核验另保留其求解器容差。约束以未舍入数值判断。',
        '', '- `tables/maximum_safe_payload.csv`：45组最大安全载荷、能量预算、空载/满载能耗与地形量。',
        '- `tables/Q1_单点组批.csv`：与提交模板同名工作表的九列一致；“往返时间”在本交付中约定为完整单架次作业时间，纯飞行时间另存plan_N_E_T.csv。',
        '- `tables/plan_N_E_T.csv`、`box_assignment_N_E_T.csv`：主方案详细分项与逐箱归属。',
        '- `tables/plan_E_N_T.csv`、`plan_T_N_E.csv`、`plan_N_T_E.csv`：其他优先关系的完整可执行箱组。',
        '- `tables/reserve_sensitivity_all_trips.csv`：所有可行情景的逐架次完整货箱编号。',
        '- `tables/exact_site_trip_thresholds.csv`、`exact_global_trip_steps.csv`：逐区瓶颈阈值与全局阶梯区间。',
        '- `figures/`：最大载荷、箱组约束利用率、多目标对比和安全余量敏感性图，PNG与SVG两种格式。',
        '', '## 9. 复现','',
        '在项目根目录运行 `./run_q1.ps1`，依次执行求解、独立核验与绘图。单独运行可用 `python solve_q1.py`、`python verify_q1.py`、`python visualize_q1.py`。绘图和MILP使用本项目.preprocess_deps中的已安装依赖。',
        '正式用于论文时，应保留本报告关于能耗推导、累计时间定义和地形离散处理的说明；不能将本问累计作业时间解释为第二问的多机最晚返航时间。','']
    (OUT/'问题一_模型建立与求解.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__': run()

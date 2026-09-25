"""独立公式复算、独立模式枚举及HiGHS整数规划最优性交叉核验。"""
from pathlib import Path
from itertools import product
from collections import Counter
import hashlib
import json
import math
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.preprocess_deps'))
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

OUT=ROOT/'q1_results'; T=OUT/'tables'; SRC=ROOT/'preprocessing'/'tables'
boxes=pd.read_csv(SRC/'boxes.csv').set_index('box_id')
types=pd.read_csv(SRC/'transport_types.csv').set_index('type_id')
arcs=pd.read_csv(SRC/'arcs.csv').set_index(['from_id','to_id'])
materials=['医疗物资','饮用水','应急食品','生活卫生用品']
tests=[]; cross=[]
def check(name,ok,detail=''):
    if not bool(ok): raise AssertionError(name+': '+str(detail))
    tests.append(dict(check=name,passed=True,detail=str(detail)))

def calc(s,g,q,n=0):
    typ=types.loc[g]; a=arcs.loc[('O01',s)]; b=arcs.loc[(s,'O01')]
    length=typ.range_empty_m-(typ.range_empty_m-typ.range_full_m)*math.pow(q/typ.max_payload_kg,1.5)
    e=typ.energy_kwh*(a.distance_m/length+b.distance_m/typ.range_empty_m)
    e+=9.80665*((typ.empty_mass_kg+q)*a.climb_m+typ.empty_mass_kg*b.climb_m)/(typ.up_efficiency*3600000)
    f=(a.climb_m+b.climb_m)/typ.up_mps+(a.descent_m+b.descent_m)/typ.down_mps+(a.distance_m+b.distance_m)/typ.cruise_mps
    time=f+typ.prepare_s+typ.handover_base_s+n*(typ.load_per_box_s+typ.handover_per_box_s)
    return float(e),float(time),float(f)

def validate_plan(frame,name):
    ids=[]; errors=[]
    for r in frame.itertuples():
        assigned=r.box_ids.split(';'); ids.extend(assigned)
        b=boxes.loc[assigned]; g=types.loc[r.type_id]
        if not b.service_id.eq(r.service_id).all(): errors.append('跨区')
        q=float(b.mass_kg.sum()); v=float(b.volume_m3.sum()); e,t,f=calc(r.service_id,r.type_id,q,len(b))
        if abs(q-r.mass_kg)>1e-8 or abs(v-r.volume_m3)>1e-10: errors.append('质量体积')
        if q>g.max_payload_kg+1e-8 or v>g.capacity_m3+1e-10: errors.append('容量')
        if e>(1-r.reserve_fraction)*g.energy_kwh+1e-8: errors.append('能量约束')
        if abs(e-r.energy_kwh)>1e-8 or abs(t-r.operation_s)>1e-5 or abs(f-r.flight_s)>1e-5: errors.append('能耗时间复算')
        if abs(1-e/g.energy_kwh-r.return_soc)>1e-9: errors.append('SOC')
    check(name+'80箱恰出现一次',Counter(ids)==Counter(boxes.index))
    check(name+'质量体积能量时间及SOC独立复算',not errors,errors)

for policy in ['N_E_T','E_N_T','T_N_E','N_T_E']:
    validate_plan(pd.read_csv(T/f'plan_{policy}.csv'),policy)
scenarios=pd.read_csv(T/'reserve_sensitivity_all_trips.csv')
for scenario,frame in scenarios.groupby('scenario'): validate_plan(frame,scenario)

cap=pd.read_csv(T/'maximum_safe_payload.csv')
capacity_ok=True
for r in cap.itertuples():
    q=r.max_safe_payload_kg; g=types.loc[r.type_id]
    if pd.isna(q): capacity_ok=capacity_ok and calc(r.service_id,r.type_id,0)[0]>r.energy_budget_kwh
    else:
        e=calc(r.service_id,r.type_id,q)[0]
        capacity_ok=capacity_ok and q<=g.max_payload_kg+1e-8 and e<=r.energy_budget_kwh+1e-8
        if q<g.max_payload_kg-1e-6: capacity_ok=capacity_ok and abs(e-r.energy_budget_kwh)<1e-8
check('45组安全载荷满足能量边界或标称边界',capacity_ok)
curve=pd.read_csv(T/'reserve_capacity_curves.csv')
check('45条载荷曲线随安全余量单调不增',all((np.diff(df.max_safe_payload_kg.fillna(-1))<=1e-7).all() for _,df in curve.groupby(['service_id','type_id'])))

# 与求解器无共享代码的独立物资数量枚举。
catalog={}
for s in sorted(boxes.service_id.unique()):
    b=boxes[boxes.service_id.eq(s)]
    demand=np.array([int(b.material.eq(m).sum()) for m in materials])
    physical=[]
    for count in product(*(range(int(n)+1) for n in demand)):
        n=sum(count)
        if not n: continue
        q=v=0.
        for m,num in zip(materials,count):
            bm=b[b.material.eq(m)]
            if num: q+=num*float(bm.mass_kg.iloc[0]); v+=num*float(bm.volume_m3.iloc[0])
        for typ,g in types.iterrows():
            if q>g.max_payload_kg+1e-9 or v>g.capacity_m3+1e-12: continue
            e,t,_=calc(s,typ,q,n)
            if e>g.energy_kwh+1e-10: continue
            physical.append((count,typ,e,t,1-e/g.energy_kwh))
    catalog[s]=(demand,physical)
supplied=pd.read_csv(T/'all_physical_patterns.csv')
independent={(s,g,tuple(c)) for s,(_,patterns) in catalog.items() for c,g,e,t,r in patterns}
observed={(r.service_id,r.type_id,(r.medical_count,r.water_count,r.food_count,r.hygiene_count)) for r in supplied.itertuples()}
check('742个物理模式独立枚举一致，无遗漏',independent==observed,len(independent))

def optimize(s,rho,objective='N',fixed_n=None):
    demand,all_patterns=catalog[s]
    patterns=[p for p in all_patterns if p[4]>=rho-1e-12]
    if not patterns: return None
    matrix=np.array([p[0] for p in patterns],dtype=float).T
    low=demand.astype(float); high=low.copy()
    if fixed_n is not None:
        matrix=np.vstack([matrix,np.ones(len(patterns))]); low=np.r_[low,fixed_n]; high=np.r_[high,fixed_n]
    costs=np.ones(len(patterns)) if objective=='N' else np.array([p[2 if objective=='E' else 3] for p in patterns])
    result=milp(costs,integrality=np.ones(len(patterns)),bounds=Bounds(np.zeros(len(patterns)),np.full(len(patterns),np.inf)),
                constraints=LinearConstraint(matrix,low,high),options={'mip_rel_gap':0.,'time_limit':60.})
    if result.status==2: return None
    if result.status!=0: raise AssertionError('MILP未证最优：'+str(result.message))
    return result

comparison=pd.read_csv(T/'objective_comparison.csv').set_index('policy')
plans={p:pd.read_csv(T/f'plan_{p}.csv') for p in comparison.index}
for s in sorted(catalog):
    baseline=plans['N_E_T'][plans['N_E_T'].service_id.eq(s)]
    nres=optimize(s,.2,'N'); n=int(round(nres.fun))
    eres=optimize(s,.2,'E',n); tres=optimize(s,.2,'T'); unrestricted=optimize(s,.2,'E')
    dp_e=float(baseline.energy_kwh.sum()); dp_t=float(baseline.operation_s.sum())
    e_global=float(plans['E_N_T'].loc[plans['E_N_T'].service_id.eq(s),'energy_kwh'].sum())
    check(s+'最少架次MILP验证',n==len(baseline))
    check(s+'固定最少架次能耗最优MILP验证',abs(eres.fun-dp_e)<1e-7)
    check(s+'主方案同时达到全局最短累计作业时间',abs(tres.fun-dp_t)<1e-5)
    check(s+'不限架次能耗最优MILP验证',abs(unrestricted.fun-e_global)<1e-7)
    cross.append(dict(service_id=s,dp_trips=len(baseline),milp_min_trips=n,dp_energy_kwh=dp_e,milp_energy_at_min_trips=float(eres.fun),
        energy_difference=dp_e-float(eres.fun),dp_operation_s=dp_t,milp_min_operation_s=float(tres.fun),
        dp_unrestricted_energy=e_global,milp_unrestricted_energy=float(unrestricted.fun),milp_gap=float(eres.mip_gap),status='optimal'))

step=pd.read_csv(T/'exact_global_trip_steps.csv')
for r in step.itertuples():
    rho=(r.lower+r.upper_inclusive)/2
    values=[optimize(s,rho,'N') for s in sorted(catalog)]
    check(f'精确阶梯区间({r.lower:.10f},{r.upper_inclusive:.10f}]中点MILP验证',all(x is not None for x in values) and sum(round(x.fun) for x in values)==r.minimum_trips)

singleton_critical=[]
for bid,b in boxes.iterrows():
    possible=[]
    for typ,g in types.iterrows():
        if b.mass_kg<=g.max_payload_kg and b.volume_m3<=g.capacity_m3:
            possible.append((1-calc(b.service_id,typ,float(b.mass_kg))[0]/g.energy_kwh,typ))
    rho,typ=max(possible)
    singleton_critical.append(dict(box_id=bid,service_id=b.service_id,material=b.material,max_reserve=rho,best_type=typ))
critical=min(row['max_reserve'] for row in singleton_critical)
solver=json.loads((OUT/'solver_summary.json').read_text(encoding='utf-8'))
check('最大可行余量与逐箱单飞极限独立一致',abs(critical-solver['largest_feasible_reserve'])<1e-10)
check('临界余量上方S008确实无解',optimize('S008',critical+1e-7,'N') is None)
check('18架次达到简单质量体积下界',sum(max(math.ceil(b.mass_kg.sum()/80-1e-12),math.ceil(b.volume_m3.sum()/.25-1e-12)) for _,b in boxes.groupby('service_id'))==18)

for source in json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8')):
    check('输入哈希不变 '+source['path'],hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256'])
pd.DataFrame(cross).to_csv(T/'milp_crosscheck.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(singleton_critical).to_csv(T/'single_box_reserve_limits.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(tests).to_csv(T/'verification_checks.csv',index=False,encoding='utf-8-sig')
report=dict(passed=len(tests),failed=0,milp_regions=len(cross),max_energy_difference=max(abs(x['energy_difference']) for x in cross),
            all_milp_status='optimal',all_milp_gap_zero=all(x['milp_gap']==0 for x in cross),
            singleton_max_reserve=critical,bottleneck_boxes=[r for r in singleton_critical if abs(r['max_reserve']-critical)<1e-10],tests=tests)
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='tests'},ensure_ascii=False,indent=2),flush=True)

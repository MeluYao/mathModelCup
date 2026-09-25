# -*- coding: utf-8 -*-
"""用2秒网格细化主方案通信窗口，并重新执行资源修复和结果导出。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import solve_q3 as q3

ROOT=Path(__file__).resolve().parent;TAB=ROOT/'q3_results'/'tables';Q2=ROOT/'q2_results'/'tables';OUT=ROOT/'q3_results'
q3.DT=2.;m=q3.JointModel();base=pd.read_csv(Q2/'plan_balanced.csv');base.attrs['policy']='balanced'
old=pd.read_csv(TAB/'communication_summary.csv');sid_to_idx={s.sid:i for i,s in enumerate(m.sites)}
assignment={i:sid_to_idx[r.hover_site_id] for i,r in old.iterrows() if pd.notna(r.hover_site_id) and str(r.hover_site_id)}
profiles=[]
for p in base.itertuples():
    rows,_=m.route_profile(p);fine=[]
    for tm,pos,_,_,_ in rows:
        direct,margin,blocked=m.link(pos,m.gateway,122,step=30);fine.append((tm,pos,direct,margin,blocked))
    profiles.append((fine,[r for r in fine if not r[2]]))
plan,tasks=m.repair_schedule(base,profiles,assignment,600.)
if plan is None:raise RuntimeError('细网格资源修复失败')
delay_by_route={int(plan.loc[i,'route_index']):float(plan.loc[i,'start_s']-base.loc[i,'start_s']) for i in plan.index}
deliveries=pd.read_csv(Q2/'deliveries_balanced.csv');deliveries['delivery_s']=[r.delivery_s+delay_by_route[int(r.route_index)] for r in deliveries.itertuples()]
plan['flight_id']=plan.flight_id.str.replace('Q2-','Q3-T-',regex=False);deliveries['flight_id']=deliveries.flight_id.str.replace('Q2-','Q3-T-',regex=False)
q3.save(plan,'transport_plan');q3.save(deliveries,'box_deliveries')
rr=[]
for k,x in enumerate(sorted(tasks,key=lambda z:z['start_s']),1):
    s=m.sites[x['site_idx']];rr.append(dict(relay_flight_id=f'Q3-R-{k:03d}',relay_id=x['relay_id'],energy_id=x['energy_id'],hover_site_id=s.sid,x_m=s.x,y_m=s.y,ground_m=s.ground,hover_agl_m=s.agl,flight_alt_m=s.alt,start_s=x['start_s'],service_start_s=x['service_start_s'],service_end_s=x['service_end_s'],return_s=x['return_s'],energy_kwh=x['energy_kwh'],return_soc=x['return_soc'],available_again_s=x['available_again_s'],covered_transport_flights=';'.join(sorted(plan.iloc[list(x['routes'])].flight_id))))
relay=pd.DataFrame(rr);q3.save(relay,'relay_plan');q3.save(relay[['hover_site_id','x_m','y_m','ground_m','hover_agl_m','flight_alt_m']].drop_duplicates(),'hover_sites')

summ=[];timeline=[]
for i,p in enumerate(plan.itertuples()):
    prof=profiles[i];need=prof[1];si=assignment.get(i);site=m.sites[si] if si is not None else None;direct_n=sum(r[2] for r in prof[0]);access=[]
    for tm,pos,direct,dmarg,blocked in prof[0]:
        amarg=np.nan
        if not direct:
            ok,amarg,_=m.link(pos,(site.x,site.y,site.alt),116);access.append(amarg)
        timeline.append(dict(transport_flight_id=p.flight_id,time_s=p.start_s+tm,mode='direct' if direct else 'relay',hover_site_id='' if direct else site.sid,direct_margin_db=dmarg,access_margin_db=amarg,communication_ok=True))
    summ.append(dict(transport_flight_id=p.flight_id,visit_order=p.visit_order,communication_samples=len(prof[0]),direct_samples=direct_n,relay_samples=len(prof[0])-direct_n,direct_fraction=direct_n/len(prof[0]),relay_required=bool(need),hover_site_id='' if site is None else site.sid,min_direct_margin_db=min(r[3] for r in prof[0]),min_access_margin_db=min(access) if access else np.nan,first_relay_need_s=p.start_s+need[0][0] if need else np.nan,last_relay_need_s=p.start_s+need[-1][0] if need else np.nan))
q3.save(pd.DataFrame(summ),'communication_summary');q3.save(pd.DataFrame(timeline),'communication_timeline_2s')

resources=[]
for r in plan.itertuples():
    resources.extend([dict(resource_class='transport_aircraft',resource_id=r.aircraft_id,task_id=r.flight_id,occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.return_s,start_soc=1,end_soc=r.return_soc),dict(resource_class='transport_battery',resource_id=r.battery_id,task_id=r.flight_id,occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.battery_recharged_s,start_soc=1,end_soc=r.return_soc)])
for r in relay.itertuples():
    resources.extend([dict(resource_class='relay_aircraft',resource_id=r.relay_id,task_id=r.relay_flight_id,occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.return_s+m.rtype.turnaround_s,start_soc=1,end_soc=r.return_soc),dict(resource_class='relay_energy',resource_id=r.energy_id,task_id=r.relay_flight_id,occupied_start_s=r.start_s,occupied_end_s=r.return_s,available_again_s=r.available_again_s,start_soc=1,end_soc=r.return_soc)])
q3.save(pd.DataFrame(resources),'resource_usage')

hard=deliveries[deliveries.hard_deadline_s.notna()];soft=np.maximum(deliveries.delivery_s-deliveries.expected_s,0);metrics=dict(policy='balanced_refined_2s',merge_gap_s=600.,hard_violations=int((hard.delivery_s>hard.hard_deadline_s+1e-7).sum()),max_hard_lateness_s=float(np.maximum(hard.delivery_s-hard.hard_deadline_s,0).max()),weighted_tardiness=float((deliveries.priority*soft/deliveries.expected_s).sum()),weighted_delivery_ratio=float((deliveries.priority*deliveries.delivery_s/deliveries.expected_s).sum()/deliveries.priority.sum()),on_time_boxes=int((deliveries.delivery_s<=deliveries.expected_s+1e-7).sum()),joint_makespan_s=float(max(plan.return_s.max(),relay.return_s.max())),transport_energy_kwh=float(plan.energy_kwh.sum()),relay_energy_kwh=float(relay.energy_kwh.sum()),total_energy_kwh=float(plan.energy_kwh.sum()+relay.energy_kwh.sum()),transport_trips=len(plan),relay_trips=len(relay),total_trips=len(plan)+len(relay),transport_shift_s=float(max(delay_by_route.values())),hover_sites=relay.hover_site_id.nunique())
comp=pd.read_csv(TAB/'objective_comparison.csv');comp=comp[~comp.policy.astype(str).str.startswith('balanced_refined')];comp=pd.concat([comp,pd.DataFrame([metrics])],ignore_index=True);q3.save(comp,'objective_comparison')
summary=json.loads((OUT/'solver_summary.json').read_text(encoding='utf-8'));summary['main_metrics']=metrics;summary['communication_time_step_s']=2.;summary['refinement']='主方案用2秒网格重建服务窗口并重新修复资源调度；最终独立验证再用30米DEM视线采样。';q3.dump(summary,OUT/'solver_summary.json')
wb={}
for name,file in [('运输架次','transport_plan'),('逐箱交付','box_deliveries'),('中继架次','relay_plan'),('通信汇总','communication_summary'),('资源使用','resource_usage'),('目标对照','objective_comparison')]:
    z=pd.read_csv(TAB/f'{file}.csv');rows=[]
    for row in z.itertuples(index=False,name=None):rows.append([None if pd.isna(v) else v for v in row])
    wb[name]={'columns':list(z.columns),'rows':rows}
q3.dump(wb,OUT/'workbook_data.json')
print(json.dumps(metrics,ensure_ascii=False,indent=2))

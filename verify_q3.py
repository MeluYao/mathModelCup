# -*- coding: utf-8 -*-
"""问题三联合方案独立复核：细化至2秒通信网格和30米DEM视线采样。"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
import solve_q3 as q3

ROOT=Path(__file__).resolve().parent;TAB=ROOT/'q3_results'/'tables';SRC=ROOT/'preprocessing'/'tables';TOL=1e-6
model=q3.JointModel();boxes=pd.read_csv(SRC/'boxes.csv').set_index('box_id')
plan=pd.read_csv(TAB/'transport_plan.csv');deliveries=pd.read_csv(TAB/'box_deliveries.csv');relay=pd.read_csv(TAB/'relay_plan.csv');resources=pd.read_csv(TAB/'resource_usage.csv');comm=pd.read_csv(TAB/'communication_summary.csv').set_index('transport_flight_id');sites=pd.read_csv(TAB/'hover_sites.csv').set_index('hover_site_id')
expected=json.loads((ROOT/'q3_results'/'solver_summary.json').read_text(encoding='utf-8'))['main_metrics']
checks=[]
def ck(name,ok,detail):checks.append({'check':name,'passed':bool(ok),'detail':str(detail)})

ck('80箱唯一交付',len(deliveries)==80 and deliveries.box_id.nunique()==80 and set(deliveries.box_id)==set(boxes.index),f'rows={len(deliveries)},unique={deliveries.box_id.nunique()}')
ck('服务区一致',all(deliveries.set_index('box_id').loc[b,'service_id']==r.service_id for b,r in boxes.iterrows()),'逐箱核对')
hard=deliveries[deliveries.hard_deadline_s.notna()];ck('硬时限满足',(hard.delivery_s<=hard.hard_deadline_s+TOL).all(),f'min_margin={(hard.hard_deadline_s-hard.delivery_s).min():.6f}s')
ck('运输返航SOC',bool((plan.return_soc>=.2-TOL).all()),f'min={plan.return_soc.min():.9f}')
ck('中继返航SOC',bool((relay.return_soc>=.2-TOL).all()),f'min={relay.return_soc.min():.9f}')
ck('悬停高度上限',bool((relay.hover_agl_m<=model.rtype.max_agl_m+TOL).all()),f'max={relay.hover_agl_m.max():.3f}m')

for klass in ['transport_aircraft','transport_battery','relay_aircraft','relay_energy']:
    ok=True
    for rid,g in resources[resources.resource_class.eq(klass)].groupby('resource_id'):
        g=g.sort_values('occupied_start_s')
        for i in range(1,len(g)):
            if g.iloc[i].occupied_start_s+TOL<g.iloc[i-1].available_again_s:ok=False
    ck(f'{klass}无重叠且周转完成',ok,f"resources={resources[resources.resource_class.eq(klass)].resource_id.nunique()}")

# 中继飞行、悬停能耗复算
emax=0.;etime=0.
for r in relay.itertuples():
    out,back,ef,maxsvc=model.relay_flight(r.x_m,r.y_m,r.flight_alt_m);dur=r.service_end_s-r.service_start_s;e=ef+(model.rtype.hover_power_kw+model.rtype.comm_power_kw)*dur/3600
    emax=max(emax,abs(e-r.energy_kwh));etime=max(etime,abs((r.start_s+model.rtype.prepare_s+out+model.rtype.link_setup_s)-r.service_start_s),abs((r.service_end_s+back)-r.return_s))
ck('中继能耗独立复算',emax<1e-6,f'max_abs_error={emax:.3g}kWh')
ck('中继时刻独立复算',etime<1e-6,f'max_abs_error={etime:.3g}s')

# 2秒网格、30米地形采样重新检查通信
outages=[];min_direct=999.;min_access=999.;min_backhaul=999.;samples=0
for p in plan.itertuples():
    phases=model.trajectory(p);times=set()
    for a,b,*_ in phases:
        times.add(a);times.add(b)
        times.update(float(x) for x in np.arange(math.ceil(a/2)*2,b,2))
    site_id=str(comm.loc[p.flight_id,'hover_site_id']) if pd.notna(comm.loc[p.flight_id,'hover_site_id']) else ''
    site=sites.loc[site_id] if site_id else None
    for tm in sorted(times):
        phase=next((x for x in phases if x[0]-1e-8<=tm<=x[1]+1e-8),phases[-1]);pos=model.pos(phase,tm);abs_t=p.start_s+tm;samples+=1
        direct,dm,_=model.link(pos,model.gateway,122,step=30);min_direct=min(min_direct,dm)
        if direct:continue
        if site is None:outages.append((p.flight_id,abs_t,'no_site'));continue
        cand=(site.x_m,site.y_m,site.flight_alt_m);access,am,_=model.link(pos,cand,116,step=30);back,bm,_=model.link(cand,model.gateway,126,step=30);min_access=min(min_access,am);min_backhaul=min(min_backhaul,bm)
        active=((relay.hover_site_id.eq(site_id))&(relay.service_start_s<=abs_t+TOL)&(relay.service_end_s>=abs_t-TOL)).any()
        if not(access and back and active):outages.append((p.flight_id,abs_t,f'access={access},back={back},active={active}'))
ck('连续通信细网格复核',not outages,f'samples={samples},outages={len(outages)},min_access_margin={min_access:.6f}dB')
ck('中继回传链路',min_backhaul>=-TOL,f'min_margin={min_backhaul:.6f}dB')
ck('联合完成时间',abs(max(plan.return_s.max(),relay.return_s.max())-expected['joint_makespan_s'])<1e-5,f'T={max(plan.return_s.max(),relay.return_s.max()):.9f}s')
ck('总能耗',abs(plan.energy_kwh.sum()+relay.energy_kwh.sum()-expected['total_energy_kwh'])<1e-5,f'E={plan.energy_kwh.sum()+relay.energy_kwh.sum():.9f}kWh')

df=pd.DataFrame(checks);df.to_csv(TAB/'verification_checks.csv',index=False,encoding='utf-8-sig')
debug_path=TAB/'communication_outages_debug.csv'
if outages:pd.DataFrame(outages,columns=['flight_id','time_s','reason']).to_csv(debug_path,index=False,encoding='utf-8-sig')
elif debug_path.exists():debug_path.unlink()
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TAB.glob('*.csv'))}
summary={'passed':bool(df.passed.all()),'checks':len(df),'failed':int((~df.passed).sum()),'communication_samples':samples,'communication_outages':len(outages),'min_access_margin_db':min_access,'min_backhaul_margin_db':min_backhaul,'sha256':manifest}
(ROOT/'q3_results'/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='sha256'},ensure_ascii=False,indent=2))
if not summary['passed']:
    print(df[~df.passed].to_string(index=False));raise SystemExit(1)

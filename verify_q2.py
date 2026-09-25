"""问题二结果的独立算术与资源可行性复核。"""
from pathlib import Path
import hashlib, json, math
import pandas as pd

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'preprocessing'/'tables'
TAB=ROOT/'q2_results'/'tables'
G0=9.80665
TOL=1e-6

boxes=pd.read_csv(SRC/'boxes.csv').set_index('box_id')
types=pd.read_csv(SRC/'transport_types.csv').set_index('type_id')
arcs=pd.read_csv(SRC/'arcs.csv').set_index(['from_id','to_id'])
fleet=pd.read_csv(SRC/'transport_fleet.csv').set_index('aircraft_id')
inventory=pd.read_csv(SRC/'transport_battery_inventory.csv').set_index('type_id')
trips=pd.read_csv(TAB/'plan_balanced.csv')
deliveries=pd.read_csv(TAB/'deliveries_balanced.csv')
legs=pd.read_csv(TAB/'route_legs.csv')

checks=[]
def check(name, ok, detail):
    checks.append({'check':name,'passed':bool(ok),'detail':str(detail)})

check('80个货箱各交付一次',len(deliveries)==len(boxes) and deliveries.box_id.nunique()==len(boxes) and set(deliveries.box_id)==set(boxes.index),f"rows={len(deliveries)}, unique={deliveries.box_id.nunique()}")
service_ok=all(deliveries.set_index('box_id').loc[b,'service_id']==r.service_id for b,r in boxes.iterrows())
check('交付服务区一致',service_ok,'逐箱服务区与原始清单比对')
check('架次货箱不重不漏',set(';'.join(trips.boxes).split(';'))==set(boxes.index) and sum(trips.box_count)==len(boxes),f"box_count_sum={trips.box_count.sum()}")

recomputed=[]
for p in trips.itertuples():
    t=types.loc[p.type_id]
    ids=p.boxes.split(';')
    b=boxes.loc[ids]
    order=p.visit_order.split('-')
    mass=float(b.mass_kg.sum()); volume=float(b.volume_m3.sum())
    remain=mass; energy=0.; clock=float(p.start_s+t.prepare_s+len(ids)*t.load_per_box_s); prev='O01'; comp={}
    for s in order:
        a=arcs.loc[(prev,s)]
        L=t.range_empty_m-(t.range_empty_m-t.range_full_m)*(remain/t.max_payload_kg)**1.5
        energy+=t.energy_kwh*a.distance_m/L+(t.empty_mass_kg+remain)*G0*a.climb_m/(t.up_efficiency*3.6e6)
        clock+=a.climb_m/t.up_mps+a.distance_m/t.cruise_mps+a.descent_m/t.down_mps
        here=b[b.service_id.eq(s)]
        clock+=t.handover_base_s+len(here)*t.handover_per_box_s
        comp[s]=clock
        remain-=float(here.mass_kg.sum()); prev=s
    a=arcs.loc[(prev,'O01')]
    L=t.range_empty_m-(t.range_empty_m-t.range_full_m)*(0/t.max_payload_kg)**1.5
    energy+=t.energy_kwh*a.distance_m/L+(t.empty_mass_kg)*G0*a.climb_m/(t.up_efficiency*3.6e6)
    clock+=a.climb_m/t.up_mps+a.distance_m/t.cruise_mps+a.descent_m/t.down_mps
    max_del=max(abs(float(deliveries.set_index('box_id').loc[x,'delivery_s'])-comp[boxes.loc[x,'service_id']]) for x in ids)
    recomputed.append((p.flight_id,mass,volume,energy,clock,max_del))
    check(f'{p.flight_id}质量体积',mass<=t.max_payload_kg+TOL and volume<=t.capacity_m3+TOL,f'{mass:.6f}/{t.max_payload_kg:.6f} kg, {volume:.6f}/{t.capacity_m3:.6f} m3')
    check(f'{p.flight_id}返航能量',energy<=(1-t.reserve_fraction)*t.energy_kwh+TOL,f'{energy:.9f} <= {(1-t.reserve_fraction)*t.energy_kwh:.9f} kWh')
    check(f'{p.flight_id}算术复算',abs(energy-p.energy_kwh)<TOL and abs(clock-p.return_s)<TOL and max_del<TOL,f'dE={energy-p.energy_kwh:.3g}, dt={clock-p.return_s:.3g}, dCmax={max_del:.3g}')

check('机型与实体飞机一致',all(fleet.loc[r.aircraft_id,'type_id']==r.type_id for r in trips.itertuples()),'逐架次核对')
for aid,g in trips.sort_values('start_s').groupby('aircraft_id'):
    g=g.sort_values('start_s'); ok=all(g.iloc[i].start_s+TOL>=g.iloc[i-1].return_s for i in range(1,len(g)))
    check(f'飞机{aid}无重叠',ok,f'{len(g)}架次')
for bid,g in trips.sort_values('start_s').groupby('battery_id'):
    g=g.sort_values('start_s'); ok=all(g.iloc[i].start_s+TOL>=g.iloc[i-1].battery_recharged_s for i in range(1,len(g)))
    check(f'电池{bid}充满后复用',ok,f'{len(g)}架次')
valid_bats=True
for r in trips.itertuples():
    prefix=f'BAT-{r.type_id}-'
    try: num=int(r.battery_id.split('-')[-1])
    except Exception: num=999
    valid_bats &= r.battery_id.startswith(prefix) and num<=int(inventory.loc[r.type_id,'resource_count'])
check('电池编号与库存一致',valid_bats,f"used={trips.battery_id.nunique()}")
hard=deliveries[deliveries.hard_deadline_s.notna()]
check('全部硬时限满足',bool((hard.delivery_s<=hard.hard_deadline_s+TOL).all()),f"min_margin={(hard.hard_deadline_s-hard.delivery_s).min():.6f}s")
check('最低返航SOC不低于20%',bool((trips.return_soc>=.2-TOL).all()),f"min_SOC={trips.return_soc.min():.9f}")
check('路线分段能耗闭合',all(abs(legs[legs.flight_id.eq(r.flight_id)].energy_kwh.sum()-r.energy_kwh)<TOL for r in trips.itertuples()),'逐架次分段和与计划表比对')
check('总能耗与完工时间',abs(trips.energy_kwh.sum()-67.30489708850475)<TOL and abs(trips.return_s.max()-9943.021474165032)<TOL,f"E={trips.energy_kwh.sum():.9f}, Tmax={trips.return_s.max():.9f}")

out=pd.DataFrame(checks)
out.to_csv(TAB/'verification_checks.csv',index=False,encoding='utf-8-sig')
manifest={}
for p in sorted(TAB.glob('*.csv')):
    manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
summary={'passed':bool(out.passed.all()),'checks':len(out),'failed':int((~out.passed).sum()),'flight_recomputations':len(recomputed),'sha256':manifest}
(ROOT/'q2_results'/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='sha256'},ensure_ascii=False,indent=2))
if not summary['passed']:
    print(out[~out.passed].to_string(index=False)); raise SystemExit(1)

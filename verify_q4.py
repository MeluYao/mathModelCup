# -*- coding: utf-8 -*-
"""问题四结果的独立一致性与资源可行性校验。"""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent;T=ROOT/'q4_results'/'tables';Q3=ROOT/'q3_results'/'tables'
checks=[]
def ck(name,ok,detail): checks.append({'check':name,'passed':bool(ok),'detail':str(detail)})

def main():
    a=pd.read_csv(T/'partition_assignments.csv');g=pd.read_csv(T/'group_summary.csv');r=pd.read_csv(T/'resource_comparison.csv');u=pd.read_csv(T/'task_resource_allocation.csv');d=pd.read_csv(T/'relay_duplication.csv')
    tp=pd.read_csv(Q3/'transport_plan.csv');bd=pd.read_csv(Q3/'box_deliveries.csv');rp=pd.read_csv(Q3/'relay_plan.csv')
    services=set(bd.service_id);flights=set(tp.flight_id)
    for sc in sorted(a.scenario.unique()):
        aa=a[a.scenario.eq(sc)];gg=g[g.scenario.eq(sc)];uu=u[u.scenario.eq(sc)];rr=r[r.scenario.eq(sc)];dd=d[d.scenario.eq(sc)]
        ck(f'{sc}-服务区唯一覆盖',len(aa)==len(services) and set(aa.service_id)==services and aa.service_id.is_unique,f'{len(aa)}/{len(services)}')
        smap=aa.set_index('service_id').group_id.to_dict();same=True
        for x in tp.itertuples():
            ss=x.visit_order.split('-');same &= len({smap[s] for s in ss})==1
        ck(f'{sc}-多点架次不可拆分',same,'所有原架次访问点均在同组')
        derived={gid:set(tp[tp.visit_order.map(lambda z:any(s in set(z.split('-')) for s in set(zs.service_id)))].flight_id) for gid,zs in aa.groupby('group_id')}
        ck(f'{sc}-运输架次完整且唯一',set().union(*derived.values())==flights and sum(map(len,derived.values()))==len(flights),f'{sum(map(len,derived.values()))}/{len(flights)}')
        ck(f'{sc}-货箱覆盖',sum(int(x) for x in gg.boxes)==len(bd),f'{gg.boxes.sum()}/{len(bd)}')
        expected=[]
        for x in rp.itertuples():
            fs=set(str(x.covered_transport_flights).split(';'));gs=sorted(k for k,v in derived.items() if fs&v);expected.append((x.relay_flight_id,';'.join(gs),len(gs)))
        got=[tuple(x) for x in dd[['relay_flight_id','assigned_groups','copy_count']].itertuples(index=False,name=None)]
        ck(f'{sc}-中继复制关系',sorted(expected)==sorted(got),f'{sum(x[2] for x in expected)}个组内中继任务副本')
        no_overlap=True
        for _,z in uu.groupby(['resource_class','local_resource_id']):
            z=z.sort_values('occupied_start_s');ends=z.available_again_s.to_list();starts=z.occupied_start_s.to_list()
            no_overlap &= all(ends[i] <= starts[i+1]+1e-8 for i in range(len(z)-1))
        ck(f'{sc}-资源时序无重叠',no_overlap,f'{len(uu)}条占用记录')
        count_ok=True
        for _,x in gg.iterrows():
            z=uu[uu.group_id.eq(x['group_id'])]
            for rc in rr.resource:
                count_ok &= z[z.resource_class.eq(rc)].local_resource_id.nunique()==int(x[rc])
        ck(f'{sc}-分组资源数一致',count_ok,'区间着色编号数与分组峰值一致')
        sums=gg[list(rr.resource)].sum().to_dict();arith=True
        for x in rr.itertuples():
            arith &= int(x.partition_total_required)==int(sums[x.resource])
            arith &= int(x.deficit)==max(int(x.partition_total_required)-int(x.existing_stock),0)
            arith &= int(x.redundancy)==int(x.partition_total_required)-int(x.centralized_required)
        ck(f'{sc}-库存缺口与冗余算术',arith,'总需求=分组需求之和')
    out=pd.DataFrame(checks);out.to_csv(T/'verification_checks.csv',index=False,encoding='utf-8-sig')
    result={'passed':bool(out.passed.all()),'check_count':len(out),'failed':out.loc[~out.passed].to_dict('records')}
    (ROOT/'q4_results'/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()

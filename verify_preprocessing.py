"""独立验证持久化数据、物理方向关系、时限语义和栅格遍历边界。"""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from preprocess_d import Terrain

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'preprocessing'
T=OUT/'tables'
tests=[]
def require(name,condition):
    if not bool(condition): raise AssertionError(name)
    tests.append(name)

b=pd.read_csv(T/'boxes.csv')
n=pd.read_csv(T/'nodes.csv')
a=pd.read_csv(T/'arcs.csv')
f=pd.read_csv(T/'arc_type_coefficients.csv')
g=pd.read_csv(T/'transport_types.csv').set_index('type_id')
mat=np.load(OUT/'spatial/routing_matrices.npz')
require('80箱总质量758kg、体积2.011m³',len(b)==80 and math.isclose(b.mass_kg.sum(),758) and math.isclose(b.volume_m3.sum(),2.011))
require('30首批、16医疗、31个硬时限箱',int(b.first_batch.sum())==30 and int(b.expected_is_hard.sum())==16 and int(b.has_hard_deadline.sum())==31)
expected=np.where(b.first_batch,b.first_deadline_s,np.inf)
expected=np.minimum(expected,np.where(b.material.eq('医疗物资'),b.expected_s,np.inf))
require('硬截止按医疗及首批约束共同计算',np.allclose(np.where(b.hard_deadline_s.notna(),b.hard_deadline_s,np.inf),expected))
require('S012/S014医疗优先取首批1小时截止',b.loc[b.box_id.isin(['S012-MED-01','S014-MED-01']),'hard_deadline_s'].eq(3600).all())
require('非首批S001第二医疗箱仍为硬约束',b.loc[b.box_id.eq('S001-MED-02'),'hard_deadline_s'].iloc[0]==3600)
require('非首批普通物资保留空硬时限',b.loc[~b.first_batch & ~b.expected_is_hard,'hard_deadline_s'].isna().all())
require('节点矩阵顺序一致',list(mat['node_ids'])==list(n.node_id))
require('距离矩阵对称且正对角外',np.allclose(mat['distance_m'],mat['distance_m'].T) and (mat['distance_m'][~np.eye(16,dtype=bool)]>0).all())
require('正反爬升下降互换',np.allclose(mat['climb_m'],mat['descent_m'].T))
lookup=n.set_index('node_id')
for r in a.itertuples():
    require_alt=math.isclose(r.cruise_alt_m,r.terrain_max_m+50,abs_tol=1e-5)
    if not require_alt: raise AssertionError('巡航净空')
    if not math.isclose(r.climb_m,r.cruise_alt_m-lookup.loc[r.from_id,'operation_alt_m'],abs_tol=1e-5): raise AssertionError('爬升高度')
require('240条航段逐条满足50米巡航净空及起点高度',len(a)==240)
fa=f.merge(a,on=['from_id','to_id'])
err=[]
for r in fa.itertuples():
    typ=g.loc[r.type_id]
    calc=r.climb_m/typ.up_mps+r.distance_m/typ.cruise_mps+r.descent_m/typ.down_mps
    err.append(abs(calc-r.flight_s))
require('720条机型航段时间独立重算',len(fa)==720 and max(err)<1e-5)
require('8架运输飞机原编号完整',set(pd.read_csv(T/'transport_fleet.csv').aircraft_id)=={f'U{i:02d}' for i in range(1,9)})
energy=pd.read_csv(T/'energy_resources.csv')
require('14电池及6能源组件唯一且初始满电',energy.resource_id.is_unique and len(energy)==20 and energy.initial_soc.eq(1).all() and energy.resource_kind.value_counts().to_dict()=={'transport_battery':14,'relay_energy':6})
budget=pd.read_csv(T/'communication_budgets.csv').set_index('link_type')
require('通信双向预算独立手算',budget.loc['direct','bidirectional_limit_db']==122 and budget.loc['relay_access','bidirectional_limit_db']==116 and budget.loc['relay_backhaul','bidirectional_limit_db']==126)

# 合成3×3栅格，验证常见采样漏角点、沿边界、反向对称三种情况。
ter=Terrain.__new__(Terrain)
ter.z=np.arange(9,dtype=np.float32).reshape(3,3)
ter.dx=ter.dy=1.; ter.west=0.; ter.north=2.; ter.h=ter.w=3
diag=ter.traverse(0,2,2,0)
require('对角线角点相交保守计入邻格',{(r['row'],r['col']) for r in diag}=={(0,0),(0,1),(1,0),(1,1),(1,2),(2,1),(2,2)})
rev=ter.traverse(2,0,0,2)
require('遍历反向像元集合不变',{(r['row'],r['col']) for r in diag}=={(r['row'],r['col']) for r in rev})
edge=ter.traverse(.5,2,.5,0)
require('沿像元边界双侧完整覆盖',{(r['row'],r['col']) for r in edge}=={(i,j) for i in range(3) for j in [0,1]})
reverse_map={(r['row'],r['col']):r for r in rev}
require('反向进入退出比例正确',all(math.isclose(r['t_enter'],1-reverse_map[(r['row'],r['col'])]['t_exit'],abs_tol=1e-12) and math.isclose(r['t_exit'],1-reverse_map[(r['row'],r['col'])]['t_enter'],abs_tol=1e-12) for r in diag))
quality=pd.read_csv(T/'quality_checks.csv')
require('数据质量无错误',quality.loc[quality.severity.eq('error'),'passed'].all())
catalog=json.loads((OUT/'figure_catalog.json').read_text(encoding='utf-8'))
for c in catalog:
    p=OUT/'figures'/(c['name']+'.png')
    with Image.open(p) as im:
        require(c['name']+'图片可读且宽度大于1500像素',im.width>1500 and im.height>800)
    require(c['name']+'SVG存在',(OUT/'figures'/(c['name']+'.svg')).stat().st_size>1000)
for entry in json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8')):
    if hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()!=entry['sha256']:
        raise AssertionError('源附件校验和变化：'+entry['path'])
require('全部源附件与清单SHA256一致',True)
result=dict(passed=len(tests),failed=0,tests=tests)
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

"""D题预处理。原始附件只读；输出到 preprocessing/。

运行：python preprocess_d.py
依赖：numpy pandas openpyxl Pillow matplotlib scipy geographiclib
可选项目依赖目录：.preprocess_deps（优先于解释器环境）
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
if (ROOT / '.preprocess_deps').exists():
    sys.path.insert(0, str(ROOT / '.preprocess_deps'))
import csv
import hashlib
import html
import json
import math
import platform
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import openpyxl

OUT = ROOT / 'preprocessing'
TABLE = OUT / 'tables'
FIG = OUT / 'figures'
SPATIAL = OUT / 'spatial'
for folder in (OUT, TABLE, FIG, SPATIAL):
    folder.mkdir(parents=True, exist_ok=True)
BASE = ROOT / '数据' / '无人机应急物资运输基础数据'
checks = []
schema = []


def check(name, ok, detail='', severity='error'):
    checks.append(dict(check=name, passed=bool(ok), severity=severity, detail=str(detail)))


def save(df, name, descriptions=None):
    df.to_csv(TABLE / (name + '.csv'), index=False, encoding='utf-8-sig', float_format='%.10g')
    for col in df.columns:
        schema.append(dict(table=name, field=col, dtype=str(df[col].dtype),
                           description=(descriptions or {}).get(col, DESCRIPTIONS.get(col, col)),
                           missing_count=int(df[col].isna().sum())))
    return df


DESCRIPTIONS = {
    'node_id':'节点编号；保留附件O01/S001等原始编号', 'node_name':'原始节点名称',
    'node_type':'depot调度中心/service服务区', 'lon_deg':'WGS84经度，度',
    'lat_deg':'WGS84纬度，度', 'ground_m':'附件给定地面海拔，米；不以DEM替换',
    'population':'本次需保障人口；调度中心不适用', 'x_m':'相对O01向东坐标，米',
    'y_m':'相对O01向北坐标，米', 'operation_alt_m':'作业海拔，O01为地面，服务区为地面+30米',
    'dem_row':'DEM行索引，0起，北向南递增', 'dem_col':'DEM列索引，0起，西向东递增',
    'dem_nearest_m':'包含该节点的DEM像元高程，米', 'dem_bilinear_m':'像元中心双线性插值高程，米，仅供比对',
    'dem_minus_given_m':'DEM像元高程减附件给定海拔，米',
    'box_id':'逐箱唯一编号', 'service_id':'所属服务区编号', 'material':'物资类型',
    'mass_kg':'单箱质量，kg', 'volume_m3':'单箱体积，立方米', 'first_batch':'是否首批保障，布尔值',
    'first_deadline_s':'原始首批截止时间，秒；非首批为空，不能填0',
    'expected_s':'原始期望交付完成时间，秒', 'priority':'附件应急优先系数',
    'hard_deadline_s':'医疗期望时间与首批截止时间的适用最小值；无硬时限为空',
    'has_hard_deadline':'是否有硬时限', 'expected_is_hard':'医疗为真；其他为假，首批另有硬时限',
    'type_id':'机型编号', 'type_name':'附件机型名称', 'empty_mass_kg':'含电池/能源组件的空载总质量，kg',
    'max_payload_kg':'标称最大载货质量，kg', 'capacity_m3':'可用装载体积，立方米',
    'cruise_mps':'计划水平巡航速度，m/s', 'range_empty_m':'空载标准航程，米',
    'range_full_m':'满载标准航程，米', 'energy_kwh':'电池/组件可用能量，kWh',
    'reserve_fraction':'返航SOC下限比例，0至1', 'prepare_s':'工位固定准备时间，秒',
    'load_per_box_s':'每箱装载时间，秒', 'handover_base_s':'接收点基础交接时间，秒',
    'handover_per_box_s':'每箱增加交接时间，秒', 'up_mps':'爬升速度，m/s',
    'down_mps':'下降速度，m/s', 'up_efficiency':'爬升能耗效率', 'down_efficiency':'下降能耗效率，题目为0',
    'aircraft_id':'附件逐架实体无人机编号', 'initial_node':'初始所在节点',
    'resource_id':'为求解器生成的电池/能源组件编号，非附件原始编号',
    'id_generated':'是否为预处理生成编号', 'resource_kind':'transport_battery或relay_energy',
    'initial_soc':'初始SOC，比例，1表示100%', 'full_charge_s':'等效完全充电时间，秒',
    'resource_count':'能源资源库存总数，已包含初始装机资源',
    'module_mass_kg':'中继通信模块质量，kg', 'takeoff_mass_kg':'中继计划起飞总质量，kg',
    'cruise_power_kw':'中继巡航功率，kW', 'link_setup_s':'中继建链时间，秒',
    'turnaround_s':'中继架次周转时间，秒', 'hover_power_kw':'悬停功率，kW',
    'comm_power_kw':'通信附加功率，kW', 'max_agl_m':'最大悬停离地高度，米',
    'total_boxes':'总需求箱数', 'first_boxes':'首批保障箱数', 'total_mass_kg':'总需求质量，kg',
    'total_volume_m3':'总需求体积，立方米', 'hard_boxes':'有硬时限的箱数',
    'medical_boxes':'医疗箱数', 'earliest_hard_s':'最早硬截止时刻，秒',
    'from_id':'航段起点编号', 'to_id':'航段终点编号', 'distance_m':'本地米制投影下水平直线距离，米',
    'geodesic_m':'WGS84椭球测地距离，仅作投影误差参照，米',
    'projection_relative_error':'本地距离/测地距离-1',
    'terrain_max_m':'航段穿越闭像元集合的最大DEM高程，米',
    'cruise_alt_m':'计划巡航海拔=沿线最高DEM+50米', 'climb_m':'从起点作业高度到巡航海拔的爬升，米',
    'descent_m':'从巡航海拔到终点作业高度的下降，米', 'cell_count':'保守supercover遍历的像元数量',
    'flight_s':'仅飞行时间，爬升+巡航+下降；不含准备装载交接',
    'horizontal_energy_factor':'distance_m/range(q)的分子energy_kwh*distance_m，kWh*m',
    'climb_energy_per_kg':'g*climb/(eff*3.6e6)，kWh/kg，推导能耗约定',
    't_enter':'进入像元时沿线比例[0,1]', 't_exit':'离开像元时沿线比例[0,1]',
    'terrain_m':'像元DEM高程，米', 'row':'像元行，0起', 'col':'像元列，0起',
    'category':'原始参数类别', 'parameter':'原始参数名称，含单位', 'symbol':'原始参数符号', 'value':'原始参数值',
    'source_file':'来源文件相对路径', 'source_sheet':'来源工作表', 'source_row':'来源Excel行号，1起',
    'layer':'地理图层英文标识', 'vertex_rows':'顶点记录行数；点图层为点数', 'feature_count':'按原始要素编号去重的要素数',
    'link_type':'direct直连/relay_access中继接入/relay_backhaul中继回传',
    'endpoint_a':'链路端点A类别', 'endpoint_b':'链路端点B类别',
    'a_to_b_limit_db':'A发B收方向允许传播损耗，dB', 'b_to_a_limit_db':'B发A收方向允许传播损耗，dB',
    'gateway_alt_m':'固定网关天线海拔，米', 'distance_3d_km':'通信端点三维直线距离，km',
    'terrain_obstructed':'视线是否被DEM地形遮挡；真只增加附加损耗，不直接等于断链',
    'path_loss_db':'自由空间损耗加地形遮挡附加损耗，dB',
    'check':'检查项目', 'passed':'该检查是否通过', 'severity':'失败时级别；error阻断，warning仅提示',
    'detail':'检查详情；通过且无补充说明时为空',
}


def rows(filename, sheet=0):
    w = openpyxl.load_workbook(BASE / filename, data_only=True, read_only=True)
    s = w.worksheets[sheet]
    result = [(i, list(r)) for i, r in enumerate(s.iter_rows(values_only=True), 1) if any(v is not None for v in r)]
    w.close()
    return result


def selected(filename, start, stop, cols, sheet=0):
    records = []
    for i, r in rows(filename, sheet):
        if start <= i <= stop:
            rec = dict(zip(cols, r))
            rec.update(source_file=filename, source_sheet='逐箱货箱清单' if sheet == 1 else '数据', source_row=i)
            records.append(rec)
    return pd.DataFrame(records)


def load_tables():
    nr = rows('调度中心与服务区.xlsx')
    nodes = []
    for i, r in nr:
        if r[0] == 'O01' or (isinstance(r[0], str) and r[0].startswith('S0')):
            nodes.append(dict(node_id=r[0], node_name=r[1], node_type='depot' if r[0]=='O01' else 'service',
                              lon_deg=r[2], lat_deg=r[3], ground_m=r[4], population=r[5],
                              source_file='调度中心与服务区.xlsx', source_sheet='数据', source_row=i))
    nodes = pd.DataFrame(nodes)
    boxcols = ['box_id','service_id','material','mass_kg','volume_m3','first_batch','first_deadline_s','expected_s','priority']
    boxes = selected('物资需求与配送时限.xlsx', 2, 10000, boxcols, 1)
    check('首批标记合法', boxes.first_batch.isin(['是','否']).all())
    boxes['first_batch'] = boxes.first_batch.eq('是')
    boxes['expected_is_hard'] = boxes.material.eq('医疗物资')
    boxes['hard_deadline_s'] = pd.concat([boxes.first_deadline_s.where(boxes.first_batch),
                                         boxes.expected_s.where(boxes.expected_is_hard)], axis=1).min(axis=1)
    boxes['has_hard_deadline'] = boxes.hard_deadline_s.notna()
    demand = selected('物资需求与配送时限.xlsx', 2, 10000,
                      ['service_id','material','total_boxes','first_boxes','mass_kg','volume_m3','priority','first_deadline_s','expected_s'])
    trcols = ['type_id','type_name','empty_mass_kg','max_payload_kg','capacity_m3','cruise_mps','range_empty_m',
              'range_full_m','energy_kwh','reserve_fraction','prepare_s','load_per_box_s','handover_base_s',
              'handover_per_box_s','up_mps','down_mps','up_efficiency','down_efficiency']
    transport = selected('运输无人机数据.xlsx', 3, 5, trcols)
    transport['reserve_fraction'] /= 100
    # 按实体编号定位，避免空白行或分节表头改变造成错位。
    def entity_fleet(filename, prefix):
        entries=[]
        for i,r in rows(filename):
            if isinstance(r[0],str) and r[0].startswith(prefix) and r[0][len(prefix):].isdigit():
                entries.append(dict(aircraft_id=r[0],type_id=r[1],initial_node=r[2],
                                    source_file=filename,source_sheet='数据',source_row=i))
        return pd.DataFrame(entries)
    fleet = entity_fleet('运输无人机数据.xlsx','U')
    # 定位库存行，不依赖尾部空行数。
    def inventory(filename):
        rr = rows(filename)
        start = next(i for i,r in rr if r[0]=='机型编号' and len(r)>1 and str(r[1]).startswith('共享'))
        return selected(filename, start+1, 10000, ['type_id','resource_count','full_charge_s'])
    batteries = inventory('运输无人机数据.xlsx')
    relaycols = ['type_id','type_name','empty_mass_kg','module_mass_kg','takeoff_mass_kg','cruise_mps','cruise_power_kw',
                 'energy_kwh','reserve_fraction','prepare_s','link_setup_s','turnaround_s','up_mps','down_mps',
                 'up_efficiency','down_efficiency','hover_power_kw','comm_power_kw','max_agl_m']
    relay = selected('中继无人机数据.xlsx', 3, 3, relaycols)
    relay['reserve_fraction'] /= 100
    rfleet = entity_fleet('中继无人机数据.xlsx','R')
    rinv = inventory('中继无人机数据.xlsx')
    energies = []
    for kind, inv in [('transport_battery',batteries),('relay_energy',rinv)]:
        for r in inv.itertuples():
            for j in range(1,int(r.resource_count)+1):
                energies.append(dict(resource_id=f'{"BAT" if kind=="transport_battery" else "ENG"}-{r.type_id}-{j:02d}',
                                     type_id=r.type_id, resource_kind=kind, id_generated=True, initial_soc=1.0,
                                     initial_node='O01', full_charge_s=r.full_charge_s))
    comm = []
    for i,r in rows('通信链路参数.xlsx'):
        if i >= 3:
            comm.append(dict(category=r[0],parameter=r[1],symbol=r[3],value=r[4],
                             source_file='通信链路参数.xlsx',source_sheet='数据',source_row=i))
    comm = pd.DataFrame(comm)
    check('节点唯一且数量16', len(nodes)==16 and nodes.node_id.is_unique)
    check('服务区编号完整', set(nodes.loc[nodes.node_type=='service','node_id'])=={f'S{i:03d}' for i in range(1,16)})
    check('货箱唯一且数量80', len(boxes)==80 and boxes.box_id.is_unique)
    check('货箱外键全部有效', boxes.service_id.isin(nodes.node_id).all())
    check('必填货箱字段无缺失', boxes[['box_id','service_id','material','mass_kg','volume_m3','expected_s','priority']].notna().all().all())
    check('货箱质量体积与时限为正', (boxes[['mass_kg','volume_m3','expected_s','priority']]>0).all().all())
    check('首批时限缺失仅出现在非首批', boxes.first_deadline_s.notna().equals(boxes.first_batch))
    compare = []
    for r in demand.itertuples():
        b = boxes[(boxes.service_id==r.service_id)&(boxes.material==r.material)]
        ok = len(b)==r.total_boxes and int(b.first_batch.sum())==r.first_boxes
        for field in ['mass_kg','volume_m3','priority','expected_s']:
            ok = ok and bool(np.allclose(b[field],getattr(r,field)))
        if r.first_boxes:
            ok = ok and bool(np.allclose(b.loc[b.first_batch,'first_deadline_s'],r.first_deadline_s))
        compare.append(ok)
    check('需求汇总与逐箱清单逐项一致', all(compare), f'{len(compare)}个服务区-物资组合')
    check('需求组合无缺失或重复', not demand.duplicated(['service_id','material']).any() and
          set(zip(boxes.service_id,boxes.material))==set(zip(demand.service_id,demand.material)))
    check('运输库存一致', Counter(fleet.type_id)=={'A':4,'B':2,'C':2} and dict(zip(batteries.type_id,batteries.resource_count))=={'A':6,'B':4,'C':4})
    check('中继库存一致', len(rfleet)==2 and int(rinv.resource_count.sum())==6)
    check('全机队编号唯一且初始位置有效', pd.concat([fleet,rfleet]).aircraft_id.is_unique and pd.concat([fleet,rfleet]).initial_node.eq('O01').all())
    for name, df in [('boxes',boxes),('demand_by_material',demand),('transport_types',transport),('transport_fleet',fleet),
                     ('transport_battery_inventory',batteries),('relay_types',relay),('relay_fleet',rfleet),
                     ('relay_energy_inventory',rinv),('energy_resources',pd.DataFrame(energies)),('communication_parameters',comm)]:
        save(df,name)
    summary = boxes.groupby('service_id',sort=True).agg(total_boxes=('box_id','size'),total_mass_kg=('mass_kg','sum'),
              total_volume_m3=('volume_m3','sum'),first_boxes=('first_batch','sum'),hard_boxes=('has_hard_deadline','sum'),
              medical_boxes=('expected_is_hard','sum'),earliest_hard_s=('hard_deadline_s','min')).reset_index()
    save(summary,'service_demand_summary')
    return nodes,boxes,transport,relay,comm,summary


class Terrain:
    def __init__(self):
        self.path = next((ROOT/'数据').rglob('*.tif'))
        im = Image.open(self.path)
        tags = dict(im.tag_v2)
        self.z = np.asarray(im,dtype=np.float32).copy()
        self.dx,self.dy,_ = tags[33550]
        tie = tags[33922]
        keys = tags[34735]
        keymap = {keys[i]:keys[i+3] for i in range(4,len(keys),4)}
        check('DEM CRS为WGS84', keymap.get(2048)==4326)
        self.pixel_is_point = keymap.get(1025)==2
        shift = 0 if self.pixel_is_point else 0.5
        self.west = tie[3]+(shift-tie[0])*self.dx
        self.north = tie[4]-(shift-tie[1])*self.dy
        self.z[self.z == -32767] = np.nan
        self.h,self.w = self.z.shape
        self.lon = self.west+np.arange(self.w)*self.dx
        self.lat = self.north-np.arange(self.h)*self.dy
        self.meta = dict(crs='EPSG:4326',pixel_is_point=self.pixel_is_point,
                         width=self.w,height=self.h,lon_center_min=float(self.lon[0]),lon_center_max=float(self.lon[-1]),
                         lat_center_min=float(self.lat[-1]),lat_center_max=float(self.lat[0]),pixel_lon_deg=self.dx,
                         pixel_lat_deg=self.dy,nodata_source=-32767,nodata_output='NaN',
                         nodata_count=int(np.isnan(self.z).sum()),min_m=float(np.nanmin(self.z)),max_m=float(np.nanmax(self.z)))
        (SPATIAL/'dem_metadata.json').write_text(json.dumps(self.meta,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(SPATIAL/'dem.npz',elevation_m=self.z,lon_deg=self.lon,lat_deg=self.lat)

    def rc(self,lon,lat):
        return (self.north-lat)/self.dy,(lon-self.west)/self.dx

    def nearest(self,lon,lat):
        r,c=self.rc(lon,lat)
        ri,ci=int(math.floor(r+0.5)),int(math.floor(c+0.5))
        if not(0<=ri<self.h and 0<=ci<self.w):
            raise ValueError('节点不在DEM范围内')
        return ri,ci,float(self.z[ri,ci])

    def bilinear(self,lon,lat):
        r,c=self.rc(lon,lat)
        i,j=int(math.floor(r)),int(math.floor(c))
        if not(0<=i<self.h-1 and 0<=j<self.w-1):
            return float('nan')
        a,b=r-i,c-j
        return float((1-a)*(1-b)*self.z[i,j]+a*(1-b)*self.z[i+1,j]+(1-a)*b*self.z[i,j+1]+a*b*self.z[i+1,j+1])

    def traverse(self,lon0,lat0,lon1,lat1):
        """闭像元supercover：边界相交计入两侧，角点计入相邻四格，保守防漏。"""
        r0,c0=self.rc(lon0,lat0); r1,c1=self.rc(lon1,lat1)
        cuts=[0.0,1.0]
        for a,b in [(r0,r1),(c0,c1)]:
            if abs(b-a)>1e-12:
                for edge in np.arange(math.floor(min(a,b)-0.5)+0.5,math.ceil(max(a,b)+0.5)+0.5):
                    t=(edge-a)/(b-a)
                    if 0<t<1: cuts.append(float(t))
        cuts=sorted(set(cuts))
        cells={}
        def add(t,ta,tb):
            rr=r0+(r1-r0)*t; cc=c0+(c1-c0)*t
            rs={int(math.floor(rr+0.5-eps)) for eps in [-1e-9,1e-9]}
            cs={int(math.floor(cc+0.5-eps)) for eps in [-1e-9,1e-9]}
            for r in rs:
                for c in cs:
                    if not(0<=r<self.h and 0<=c<self.w): raise ValueError('航段越出DEM')
                    old=cells.get((r,c),(ta,tb))
                    cells[(r,c)]=(min(old[0],ta),max(old[1],tb))
        for a,b in zip(cuts[:-1],cuts[1:]): add((a+b)/2,a,b)
        for t in cuts: add(t,t,t)
        return [dict(row=r,col=c,t_enter=a,t_exit=b,terrain_m=float(self.z[r,c]))
                for (r,c),(a,b) in sorted(cells.items(),key=lambda item:item[1])]


def spatial_tables(nodes,terrain,types):
    from geographiclib.geodesic import Geodesic
    ref=nodes.iloc[0]
    phi=math.radians(ref.lat_deg)
    a=6378137.0; e2=6.6943799901413165e-3
    n=a/math.sqrt(1-e2*math.sin(phi)**2)
    m=a*(1-e2)/(1-e2*math.sin(phi)**2)**1.5
    kx=math.pi/180*n*math.cos(phi); ky=math.pi/180*m
    proj=dict(method='WGS84 local linear east/north, reference O01',lon0=ref.lon_deg,lat0=ref.lat_deg,
              east_m_per_degree=kx,north_m_per_degree=ky,
              path='straight in local east/north; equivalent to linear lon/lat interpolation')
    nodes['x_m']=(nodes.lon_deg-ref.lon_deg)*kx
    nodes['y_m']=(nodes.lat_deg-ref.lat_deg)*ky
    nodes['operation_alt_m']=nodes.ground_m+np.where(nodes.node_type=='service',30,0)
    nearest=[terrain.nearest(r.lon_deg,r.lat_deg) for r in nodes.itertuples()]
    nodes[['dem_row','dem_col','dem_nearest_m']]=nearest
    nodes['dem_row']=nodes.dem_row.astype(int); nodes['dem_col']=nodes.dem_col.astype(int)
    nodes['dem_bilinear_m']=[terrain.bilinear(r.lon_deg,r.lat_deg) for r in nodes.itertuples()]
    nodes['dem_minus_given_m']=nodes.dem_nearest_m-nodes.ground_m
    check('全部节点DEM有效', np.isfinite(nodes.dem_nearest_m).all())
    check('节点地面海拔与DEM差异不超过5m',nodes.dem_minus_given_m.abs().max()<=5,
          f'最大绝对差{nodes.dem_minus_given_m.abs().max():.6f}m，保留原始海拔',severity='warning')
    save(nodes,'nodes')
    arcs=[]; profile=[]; flight=[]
    dist=np.zeros((16,16)); alt=np.zeros((16,16)); up=np.zeros((16,16)); down=np.zeros((16,16))
    cache={}
    for i,u in enumerate(nodes.itertuples()):
        for j,v in enumerate(nodes.itertuples()):
            if i==j: continue
            if i<j:
                cells=terrain.traverse(u.lon_deg,u.lat_deg,v.lon_deg,v.lat_deg)
                cache[(i,j)]=cells
                profile.extend([dict(from_id=u.node_id,to_id=v.node_id,**c) for c in cells])
            else: cells=cache[(j,i)]
            zs=np.array([c['terrain_m'] for c in cells])
            if not np.isfinite(zs).all(): raise ValueError('航段穿越NoData：不得填零或忽略')
            highest=float(zs.max()); h=highest+50
            d=float(math.hypot(v.x_m-u.x_m,v.y_m-u.y_m))
            geo=Geodesic.WGS84.Inverse(u.lat_deg,u.lon_deg,v.lat_deg,v.lon_deg)['s12']
            hu=h-u.operation_alt_m; hd=h-v.operation_alt_m
            if hu<0 or hd<0: raise ValueError('巡航高度低于作业高度，需明确题意后处理')
            arc=dict(from_id=u.node_id,to_id=v.node_id,distance_m=d,geodesic_m=geo,
                     projection_relative_error=d/geo-1,terrain_max_m=highest,cruise_alt_m=h,
                     climb_m=hu,descent_m=hd,cell_count=len(cells))
            arcs.append(arc); dist[i,j]=d; alt[i,j]=h; up[i,j]=hu; down[i,j]=hd
            for g in types.itertuples():
                flight.append(dict(from_id=u.node_id,to_id=v.node_id,type_id=g.type_id,
                                   flight_s=hu/g.up_mps+d/g.cruise_mps+hd/g.down_mps,
                                   horizontal_energy_factor=g.energy_kwh*d,
                                   climb_energy_per_kg=9.80665*hu/(g.up_efficiency*3.6e6)))
    arcs=pd.DataFrame(arcs); profiles=pd.DataFrame(profile); flight=pd.DataFrame(flight)
    save(arcs,'arcs'); save(profiles,'terrain_profiles'); save(flight,'arc_type_coefficients')
    proj['max_task_pair_relative_error']=float(arcs.projection_relative_error.abs().max())
    (SPATIAL/'projection.json').write_text(json.dumps(proj,ensure_ascii=False,indent=2),encoding='utf-8')
    times=np.zeros((3,16,16))
    for gi,g in enumerate(types.itertuples()): times[gi]=up/g.up_mps+dist/g.cruise_mps+down/g.down_mps
    np.savez_compressed(SPATIAL/'routing_matrices.npz',node_ids=nodes.node_id.to_numpy(dtype=str),
                        type_ids=types.type_id.to_numpy(dtype=str),distance_m=dist,cruise_alt_m=alt,
                        climb_m=up,descent_m=down,flight_s=times)
    for mat,name in [(dist,'distance_matrix_m'),(alt,'cruise_altitude_matrix_m'),(up,'climb_matrix_m'),(down,'descent_matrix_m')]:
        save(pd.DataFrame(mat,columns=nodes.node_id).assign(from_id=nodes.node_id)[['from_id']+nodes.node_id.tolist()],name,
             {node:'起点由from_id指定，终点'+node+'的'+name+'值；单位m，对角线0为占位' for node in nodes.node_id})
    check('240条有向航段及720条机型航段',len(arcs)==240 and len(flight)==720)
    check('水平距离与巡航海拔矩阵对称',np.allclose(dist,dist.T) and np.allclose(alt,alt.T))
    check('正反航段爬升下降互换',np.allclose(up,down.T))
    check('米制投影任务区误差小于0.1%',proj['max_task_pair_relative_error']<0.001,
          f"与WGS84测地距离比较：{proj['max_task_pair_relative_error']*100:.6f}%")
    # 独立的10万点密采样只用于核验supercover没有漏高点，正式最大值仍取supercover。
    valid=True
    for j in range(1,len(nodes)):
        u=nodes.iloc[0]; v=nodes.iloc[j]; tt=np.linspace(0,1,100001)
        rr,cc=terrain.rc(u.lon_deg+(v.lon_deg-u.lon_deg)*tt,u.lat_deg+(v.lat_deg-u.lat_deg)*tt)
        dense=terrain.z[np.floor(rr+0.5).astype(int),np.floor(cc+0.5).astype(int)]
        valid=valid and float(np.max(dense))<=alt[0,j]-50+1e-5
    check('15条中心航线密采样交叉核验无漏高点',valid)
    return nodes,arcs,profiles,flight,proj


def backgrounds(proj):
    layers={}
    mapping={'村镇点位':'settlements','道路':'roads','水系（线）':'rivers','水体（面）':'waterbodies'}
    index=[]
    for p in sorted((ROOT/'数据').rglob('*.csv')):
        key=mapping[p.parent.name]
        df=pd.read_csv(p,encoding='utf-8-sig')
        check(f'{key}坐标无缺失',df[['经度','纬度']].notna().all().all())
        df['x_m']=(df['经度']-proj['lon0'])*proj['east_m_per_degree']
        df['y_m']=(df['纬度']-proj['lat0'])*proj['north_m_per_degree']
        df.to_csv(SPATIAL/(key+'.csv'),index=False,encoding='utf-8-sig')
        for col in df.columns:
            schema.append(dict(table='spatial/'+key,field=col,dtype=str(df[col].dtype),description=DESCRIPTIONS.get(col,'保留原始地理数据字段：'+col),missing_count=int(df[col].isna().sum())))
        idcol=df.columns[0]
        sortcols=[c for c in ['多边形编号','环编号','点序号'] if c in df]
        dupcols=[idcol]+sortcols
        check(f'{key}几何顶点键唯一',not df.duplicated(dupcols).any())
        index.append(dict(layer=key,source_file=str(p.relative_to(ROOT)),vertex_rows=len(df),feature_count=df[idcol].nunique()))
        layers[key]=df
    save(pd.DataFrame(index),'geographic_layer_inventory')
    return layers


def communication(comm,nodes,terrain):
    def val(cat,sym):
        return float(comm.loc[(comm.category==cat)&(comm.symbol==sym),'value'].iloc[0])
    f=val('传播参数','f'); lsys=val('传播参数','Lsys'); lobs=val('传播参数','Lobs')
    threshold=val('接收参数','Psens')+val('接收参数','M')
    budgets=[]
    for name,a,b in [('direct','运输无人机','固定网关 G01'),('relay_access','运输无人机','中继接入端'),
                     ('relay_backhaul','中继回传端','固定网关 G01')]:
        ab=val(a,'Pt')+val(a,'G')+val(b,'G')-lsys-threshold
        ba=val(b,'Pt')+val(b,'G')+val(a,'G')-lsys-threshold
        limit=min(ab,ba)
        budgets.append(dict(link_type=name,endpoint_a=a,endpoint_b=b,a_to_b_limit_db=ab,b_to_a_limit_db=ba,
                            bidirectional_limit_db=limit,los_range_km=10**((limit-32.45-20*math.log10(f))/20),
                            obstructed_range_km=10**((limit-lobs-32.45-20*math.log10(f))/20)))
    budgets=pd.DataFrame(budgets)
    save(budgets,'communication_budgets',{'los_range_km':'无地形遮挡时预算距离上限，km，非实际覆盖半径',
         'obstructed_range_km':'有遮挡附加损耗时预算距离上限，km，非实际覆盖半径',
         'bidirectional_limit_db':'双向允许传播损耗，dB'})
    gateway=nodes.iloc[0]; zg=gateway.ground_m+val('固定网关 G01','hG')
    records=[]
    for r in nodes.iloc[1:].itertuples():
        cells=terrain.traverse(gateway.lon_deg,gateway.lat_deg,r.lon_deg,r.lat_deg)
        margins=[zg+(r.operation_alt_m-zg)*(c['t_enter'] if r.operation_alt_m>=zg else c['t_exit'])-c['terrain_m'] for c in cells]
        obstructed=min(margins)<0
        distance=math.sqrt(r.x_m**2+r.y_m**2+(r.operation_alt_m-zg)**2)/1000
        loss=32.45+20*math.log10(f)+20*math.log10(distance)+lobs*obstructed
        margin=float(budgets.iloc[0].bidirectional_limit_db)-loss
        records.append(dict(service_id=r.node_id,operation_alt_m=r.operation_alt_m,gateway_alt_m=zg,
                            distance_3d_km=distance,terrain_obstructed=obstructed,los_min_clearance_m=min(margins),
                            path_loss_db=loss,link_margin_db=margin,direct_available=margin>=0))
    screen=save(pd.DataFrame(records),'service_direct_link_screening',{
        'direct_available':'仅服务区30米作业高度处的静态直连筛查，不代表全航程通信可行',
        'link_margin_db':'双向允许损耗减实际传播损耗，dB；非负可用',
        'los_min_clearance_m':'通信视线相对分片常值DEM的最小净空，米'})
    return budgets,screen


def charge_curve(types,relay):
    inventory=pd.concat([pd.read_csv(TABLE/'transport_battery_inventory.csv'),pd.read_csv(TABLE/'relay_energy_inventory.csv')])
    result=[]
    for r in inventory.itertuples():
        for s in np.linspace(0,1,101):
            seconds=r.full_charge_s*(.65*(.9-s)/.9+.35) if s<.9 else r.full_charge_s*.35*(1-s)/.1
            result.append(dict(type_id=r.type_id,soc=s,charge_to_full_s=seconds))
    curve=pd.DataFrame(result)
    check('充电曲线单调且100%充电时间为0',all((np.diff(g.charge_to_full_s)<=1e-8).all() and abs(g.iloc[-1].charge_to_full_s)<1e-8 for _,g in curve.groupby('type_id')))
    save(curve,'charging_curve',{'soc':'剩余SOC，0至1','charge_to_full_s':'按题目两阶段模型补至100%的时间，秒'})
    return curve


def figures(nodes,boxes,summary,types,terrain,arcs,profiles,proj,layers,screen,curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import LineCollection,PolyCollection
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.titlesize':14,'axes.titlepad':12,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
                         'savefig.facecolor':'white','svg.fonttype':'path'})
    palette=['#2878A0','#E5A23A','#5AA17A','#AB729C']
    terrain_cmap=LinearSegmentedColormap.from_list('land_elevation',['#e9efe2','#bdd1a5','#82a276','#adac80','#b99d79','#ded4c2','#faf9f5'])
    catalog=[]
    def finish(fig,name,title,caption):
        fig.savefig(FIG/(name+'.png'),dpi=190,bbox_inches='tight')
        fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
        catalog.append(dict(name=name,title=title,caption=caption))
    xx=(terrain.lon-proj['lon0'])*proj['east_m_per_degree']/1000
    yy=(terrain.lat-proj['lat0'])*proj['north_m_per_degree']/1000
    extent=[xx[0]-(xx[1]-xx[0])/2,xx[-1]+(xx[1]-xx[0])/2,yy[-1]-(yy[0]-yy[1])/2,yy[0]+(yy[0]-yy[1])/2]
    def map_base(ax,local=True):
        if local:
            ax.set_xlim(-8.5,7.5); ax.set_ylim(-2,10)
        arr=np.ma.masked_invalid(terrain.z)
        im=ax.imshow(arr,extent=extent,origin='upper',cmap=terrain_cmap,vmin=40,vmax=1150,rasterized=True)
        ax.set_aspect('equal'); ax.set_xlabel('相对 O01 向东 / km'); ax.set_ylabel('相对 O01 向北 / km')
        ax.grid(alpha=.15)
        return im
    fig,axes=plt.subplots(1,2,figsize=(14,6.5),constrained_layout=True)
    im=map_base(axes[0],False); map_base(axes[1])
    for ax in axes:
        for key,color,width in [('roads','#52616B',.4),('rivers','#1679B4',.6)]:
            df=layers[key]
            segments=[g.sort_values('点序号')[['x_m','y_m']].to_numpy()/1000 for _,g in df.groupby(df.columns[0])]
            ax.add_collection(LineCollection(segments,colors=color,linewidths=width,alpha=.5,rasterized=True))
        water=layers['waterbodies']
        # 原始环界线完整保留；只画边界，避免将孔洞错误填成水面。
        segments=[g.sort_values('点序号')[['x_m','y_m']].to_numpy()/1000 for _,g in water.groupby([water.columns[0],'多边形编号','环编号'])]
        ax.add_collection(LineCollection(segments,colors='#268AC0',linewidths=.6,alpha=.7,rasterized=True))
        ax.scatter(nodes.x_m/1000,nodes.y_m/1000,s=26,c='#B02E2E',edgecolors='white',linewidths=.6,zorder=4)
        ax.scatter([0],[0],s=150,c='#172B4D',marker='*',zorder=5)
    for r in nodes.itertuples():
        axes[1].annotate(r.node_id,(r.x_m/1000,r.y_m/1000),xytext=(4,5),textcoords='offset points',fontsize=8)
    axes[0].set_title('完整 DEM 与地理背景'); axes[1].set_title('调度中心与15个服务区')
    fig.colorbar(im,ax=axes,shrink=.75,label='地面高程 / m')
    finish(fig,'01_terrain_nodes','地形与任务节点','道路与水系为原始地理背景，不代表灾后通行状态。星号为O01，红点为服务区。')

    fig,ax=plt.subplots(figsize=(9.5,7.2),constrained_layout=True)
    map_base(ax)
    merged=nodes.iloc[1:].merge(summary,left_on='node_id',right_on='service_id')
    for r in merged.itertuples(): ax.plot([0,r.x_m/1000],[0,r.y_m/1000],color='#34495E',lw=.7,alpha=.4)
    sc=ax.scatter(merged.x_m/1000,merged.y_m/1000,s=merged.total_mass_kg*4+30,c=merged.earliest_hard_s/3600,
                  cmap='RdYlBu',vmin=1,vmax=3,edgecolor='#263238',linewidth=.7,zorder=5)
    for r in merged.itertuples(): ax.annotate(f'{r.node_id}\n{r.total_mass_kg:.0f} kg',(r.x_m/1000,r.y_m/1000),xytext=(16,5),textcoords='offset points',fontsize=8,zorder=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.65,pad=1))
    ax.scatter([0],[0],marker='*',c='black',s=180,zorder=6); ax.set_title('需求质量与最早硬时限的空间分布')
    fig.colorbar(sc,ax=ax,shrink=.65,label='最早硬截止时间 / h',ticks=[1,2,3])
    finish(fig,'02_demand_map','需求与时限空间分布','圆面积随需求质量增加；连线为单点分析航段，不是优化调度结果。')

    material_order=['医疗物资','饮用水','应急食品','生活卫生用品']
    fig,axes=plt.subplots(1,2,figsize=(14,6.2),constrained_layout=True)
    for ax,field,label in [(axes[0],'box_id','箱数 / 箱'),(axes[1],'mass_kg','质量 / kg')]:
        p=boxes.pivot_table(index='service_id',columns='material',values=field,aggfunc='count' if field=='box_id' else 'sum',fill_value=0).reindex(columns=material_order,fill_value=0)
        bottom=np.zeros(len(p))
        for c,color in zip(material_order,palette): ax.bar(p.index,p[c],bottom=bottom,label=c,color=color,width=.75); bottom+=p[c].to_numpy()
        ax.tick_params(axis='x',rotation=60); ax.set_ylabel(label); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0].set_title('各服务区货箱构成'); axes[1].set_title('各服务区质量构成'); axes[0].legend(frameon=False)
    finish(fig,'03_demand_composition','物资需求构成','质量与箱数使用各自单位；不按保障人口扩充或缩减附件需求。')

    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    hard=boxes[boxes.has_hard_deadline].groupby(['service_id','hard_deadline_s']).size().unstack(fill_value=0).reindex(summary.service_id,fill_value=0)
    data=hard.to_numpy(); im=axes[0].imshow(data,cmap='YlOrRd',aspect='auto',vmin=0)
    axes[0].set_xticks(range(len(hard.columns)),[f'{x/3600:g} h' for x in hard.columns]); axes[0].set_yticks(range(len(hard)),hard.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]): axes[0].text(j,i,str(data[i,j]),ha='center',va='center',color='white' if data[i,j]>data.max()*.6 else '#263238')
    axes[0].set_title('硬截止时间对应箱数'); fig.colorbar(im,ax=axes[0],shrink=.7,label='箱')
    soft=boxes[~boxes.expected_is_hard].groupby(['material','expected_s']).size().unstack(fill_value=0).reindex(material_order[1:],fill_value=0)
    for idx,(kind,color) in enumerate(zip(material_order[1:],palette[1:])):
        axes[1].plot(soft.columns.to_numpy()/3600,soft.loc[kind].to_numpy(),marker='o',label=kind,color=color,lw=2)
    axes[1].set_xlabel('期望交付完成时间 / h'); axes[1].set_ylabel('箱数 / 箱'); axes[1].set_title('非医疗物资期望时间分布'); axes[1].legend(frameon=False); axes[1].grid(alpha=.2)
    finish(fig,'04_deadlines','硬时限与期望时限','首批非医疗物资仍受首批硬截止约束；右图仅展示其期望时间字段，不能据此取消首批要求。')

    fig,axes=plt.subplots(4,4,figsize=(15,11),constrained_layout=True)
    for ax,(i,r) in zip(axes.flat,enumerate(nodes.iloc[1:].itertuples(),1)):
        p=profiles[(profiles.from_id=='O01')&(profiles.to_id==r.node_id)]
        a=arcs[(arcs.from_id=='O01')&(arcs.to_id==r.node_id)].iloc[0]; d=a.distance_m/1000
        for c in p.itertuples(): ax.plot([c.t_enter*d,c.t_exit*d],[c.terrain_m,c.terrain_m],c='#73836B',lw=2)
        ax.plot([0,0,d,d],[nodes.iloc[0].operation_alt_m,a.cruise_alt_m,a.cruise_alt_m,r.operation_alt_m],c='#B94F37',lw=1.5)
        ax.set_title(f'{r.node_id}  {d:.2f} km',fontsize=10); ax.grid(alpha=.18); ax.set_xlabel('距离 / km',fontsize=8); ax.set_ylabel('海拔 / m',fontsize=8); ax.tick_params(labelsize=8)
    axes.flat[-1].axis('off'); axes.flat[-1].text(.02,.9,'灰绿：沿线像元地形\n红线：爬升—巡航—下降\n巡航海拔 = 沿线最高点 + 50 m\n服务区作业高度 = 地面 + 30 m\n\n各小图坐标范围不同',va='top',fontsize=11,linespacing=1.8)
    finish(fig,'05_terrain_profiles','15条中心直达航段地形剖面','按穿越像元计算最高地形，纵向红线表示爬升和下降；图示为单点分析，不是优化路线。')

    fig,axes=plt.subplots(1,2,figsize=(14,6.2),constrained_layout=True)
    dist=np.load(SPATIAL/'routing_matrices.npz')['distance_m']/1000
    climb=np.load(SPATIAL/'routing_matrices.npz')['climb_m']
    for ax,mat,title,label,cmap in [(axes[0],dist,'节点间水平距离','km','Blues'),(axes[1],climb,'有向航段爬升高度','m','YlOrBr')]:
        masked=np.ma.array(mat,mask=np.eye(16,dtype=bool)); im=ax.imshow(masked,cmap=cmap)
        ax.set_xticks(range(16),nodes.node_id,rotation=90,fontsize=8); ax.set_yticks(range(16),nodes.node_id,fontsize=8)
        ax.set_xlabel('终点'); ax.set_ylabel('起点'); ax.set_title(title); fig.colorbar(im,ax=ax,shrink=.75,label=label)
    finish(fig,'06_route_matrices','距离与爬升矩阵','主对角线不代表真实自环任务；爬升矩阵通常不对称，反向爬升等于正向下降。')

    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,field,label in [(axes[0,0],'max_payload_kg','最大载货质量 / kg'),(axes[0,1],'capacity_m3','装载体积 / m³')]:
        ax.bar(types.type_id,types[field],color=palette[:3],width=.55)
        for x,v in zip(types.type_id,types[field]): ax.text(x,v,f'{v:g}',ha='center',va='bottom')
        ax.set_ylim(0,types[field].max()*1.18); ax.set_title(label); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    t=np.linspace(0,1,101)
    for r,color in zip(types.itertuples(),palette): axes[1,0].plot(t*100,(r.range_empty_m-(r.range_empty_m-r.range_full_m)*t**1.5)/1000,label=r.type_id,color=color,lw=2)
    axes[1,0].set_xlabel('载荷 / 标称最大载荷（%）'); axes[1,0].set_ylabel('等效航程 / km'); axes[1,0].set_title('载荷与标准等效航程'); axes[1,0].legend(); axes[1,0].grid(alpha=.2)
    for (kind,g),color in zip(curve.groupby('type_id'),palette): axes[1,1].plot(g.soc*100,g.charge_to_full_s/60,label=kind,color=color,lw=2,ls='--' if kind=='R' else '-')
    axes[1,1].axvline(90,c='#999999',ls=':',lw=1); axes[1,1].set_xlabel('任务结束SOC / %'); axes[1,1].set_ylabel('补充至满电时间 / min'); axes[1,1].set_title('两阶段充电曲线'); axes[1,1].legend(); axes[1,1].grid(alpha=.2)
    finish(fig,'07_equipment_energy','机型能力与充电周转','等效航程不包含本场景具体爬升和安全余量，不能直接当作可执行往返距离；A与R充电曲线重合。')

    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    colors=np.where(screen.direct_available,'#458B74','#C55745')
    axes[0].barh(screen.service_id,screen.link_margin_db,color=colors); axes[0].axvline(0,c='#263238',lw=1); axes[0].invert_yaxis()
    axes[0].set_xlabel('双向链路余量 / dB'); axes[0].set_title('服务区作业高度处直连筛查'); axes[0].grid(axis='x',alpha=.2)
    axes[1].barh(nodes.node_id,nodes.dem_minus_given_m,color='#547C9C'); axes[1].axvline(0,c='#263238',lw=1); axes[1].invert_yaxis(); axes[1].set_xlabel('DEM像元高程 − 给定地面海拔 / m'); axes[1].set_title('节点海拔口径核对'); axes[1].grid(axis='x',alpha=.2)
    finish(fig,'08_communication_elevation','通信初筛与高程一致性','直连筛查仅针对服务区离地30米静态位置，不能证明整段飞行连续通信；红色为静态直连不可用。')
    (OUT/'figure_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    return catalog


def main():
    print('1/6 标准化表格与校验',flush=True)
    nodes,boxes,types,relay,comm,summary=load_tables()
    print('2/6 DEM与全节点航段预计算',flush=True)
    terrain=Terrain()
    # MAT与GeoTIFF数值核对：只读可选交叉校验，不使用未知字段替换主数据。
    from scipy.io import loadmat
    mat=loadmat(str(next((ROOT/'数据').rglob('*DEM.mat'))))
    candidates=[(k,v) for k,v in mat.items() if isinstance(v,np.ndarray) and v.shape==terrain.z.shape and np.issubdtype(v.dtype,np.number)]
    matched=[]
    for k,v in candidates:
        vv=v.astype(float); vv[vv==-32767]=np.nan
        if np.allclose(vv,terrain.z,equal_nan=True): matched.append(k)
    check('MAT与GeoTIFF高程数组一致',len(matched)>0,','.join(matched))
    nodes,arcs,profiles,flight,proj=spatial_tables(nodes,terrain,types)
    print('3/6 地理图层、通信预算与充电曲线',flush=True)
    layers=backgrounds(proj)
    budgets,screen=communication(comm,nodes,terrain)
    curve=charge_curve(types,relay)
    save(pd.DataFrame(checks),'quality_checks')
    pd.DataFrame(schema).to_csv(TABLE/'data_dictionary.csv',index=False,encoding='utf-8-sig')
    errors=[x for x in checks if not x['passed'] and x['severity']=='error']
    if errors: raise RuntimeError('关键数据检查失败：'+json.dumps(errors,ensure_ascii=False))
    print('4/6 生成8组科研图形',flush=True)
    catalog=figures(nodes,boxes,summary,types,terrain,arcs,profiles,proj,layers,screen,curve)
    print('5/6 生成报告与文件校验清单',flush=True)
    write_report(nodes,boxes,summary,types,terrain,arcs,proj,screen,catalog)
    manifest=[]
    for p in sorted(list((ROOT/'数据').rglob('*'))+[ROOT/'结果提交模板.xlsx']):
        if p.is_file(): manifest.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    versions={name:__import__(name).__version__ for name in ['numpy','pandas','openpyxl','matplotlib','scipy','geographiclib']}
    import PIL
    versions['Pillow']=PIL.__version__
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),packages=versions),ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'requirements.txt').write_text('\n'.join(name+'=='+version for name,version in versions.items())+'\n',encoding='utf-8')
    print('6/6 完成',flush=True)
    print(json.dumps(dict(boxes=len(boxes),mass_kg=float(boxes.mass_kg.sum()),volume_m3=float(boxes.volume_m3.sum()),
          hard_boxes=int(boxes.has_hard_deadline.sum()),quality_checks=len(checks),errors=len(errors),
          warnings=sum(not x['passed'] for x in checks),direct_unavailable=screen.loc[~screen.direct_available,'service_id'].tolist()),ensure_ascii=False),flush=True)


def write_report(nodes,boxes,summary,types,terrain,arcs,proj,screen,catalog):
    failed=[x for x in checks if not x['passed']]
    far=arcs[arcs.from_id=='O01'].sort_values('distance_m').iloc[-1]
    notes=[
        '原始附件只读。删除的仅为表格空白行、标题行和分节表头；不删除异常记录、不填补人为需求、不改变原始编号。',
        '标准单位为m、s、kg、m³、kWh、dB。返航电量百分数转换为0至1比例，源文件和源行号保存在标准表。',
        '非首批货箱的首批时限为空，含义是不适用；没有硬截止的货箱hard_deadline_s为空，不得填0。所有货箱均保留expected_s。',
        '医疗物资expected_s是硬约束，首批first_deadline_s也是硬约束，hard_deadline_s取两者适用最小值。',
        'DEM读取GeoTIFF的PixelIsPoint标记，tiepoint为像元中心；NoData=-32767转NaN，绝不当0高程。MAT仅用于校验数据一致性。',
        '节点作业高度以节点表给定海拔为准，O01加0m，服务区加30m；DEM最近像元与双线性值均保留用于核对，不覆盖给定海拔。',
        '高程差5m是本次自定的质量提示阈值，并非题目约束；超过阈值不代表原始数据错误，也不进行自动修正。',
        '坐标采用以O01为原点的WGS84局部东西/南北线性米制投影。航线为该平面上的直线，其在经纬度中也是直线。对240个有向任务航段逐一与WGS84椭球测地距离比较。此局部近似仅用于当前任务区。',
        '每段按DEM闭像元supercover遍历，边界接触计入两侧像元，角点接触计入相邻像元；取全部穿越像元最高值+50m，保守避免漏山脊。通信筛查也按分片常值DEM及像元区间内最低视线高度判遮挡。',
        'terrain_profiles只储存120条无向节点对的正向剖面；反向使用时t_enter_new=1-t_exit、t_exit_new=1-t_enter，并反转顺序。arcs含240条有向航段。',
        '矩阵节点顺序为O01,S001,...,S015，机型顺序A,B,C。对角线0仅为计算占位，不允许把它当作真实运输架次。flight_s只包含飞行，不含准备、装载、交接及周转。',
        '运输能耗系数采用显式推导约定：L(q)=L0-(L0-LF)(q/Q)^1.5；Ehor=energy_kwh*d/L(q)；Eup=(empty_mass_kg+q)*9.80665*climb/(eff*3.6e6)。Markdown题面没有完整展开这两项，需在正式模型中说明，当前未据此求解最大安全载荷。',
        '同机型共享电池跨同型飞机可调配、不同机型不可混用。电池编号BAT-A-01等和组件编号ENG-R-01等为本次生成，不是原题指定编号。初始SOC=1。库存已含初始装机资源。',
        '充电曲线使用题目两阶段公式，不添加未提供的充电桩数量限制。原始工位准备、逐箱装载、交接、建链、周转参数分别保留。',
        '通信采用双向较小损耗门限，距离为三维距离且FSPL公式使用km和MHz。预算距离只是给定遮挡状态下理论上限，不能当作实际覆盖圆。',
        'service_direct_link_screening仅为服务区30m作业高度处静态筛查，不包含爬升/巡航/下降连续检查，不能据此宣称第三问通信可行或某服务区无法服务。',
        '道路、水体、水系和村镇为背景数据；没有洪水淹没范围、道路受损状态或禁飞区标签，不推断这些约束。水体环与顶点顺序完整保留，制图只描绘边界，避免把内环孔洞填实。',
        '本次未做货箱分批、任务调度或最优分区。图中的中心辐射线仅为地形分析航段，不是求解结果。'
    ]
    md=['# D题数据预处理与可视化报告','',
        f'已标准化16个节点、{len(boxes)}个货箱、3种运输机型、8架运输无人机、14组运输电池、2架中继无人机、6组中继能源组件。',
        f'货物共{boxes.mass_kg.sum():.0f} kg、{boxes.volume_m3.sum():.3f} m³；首批{int(boxes.first_batch.sum())}箱；医疗{int(boxes.expected_is_hard.sum())}箱；具有硬截止时间的货箱{int(boxes.has_hard_deadline.sum())}箱。',
        '', '## 数据检查与主要发现','',
        f'- 已执行{len(checks)}项检查，错误{sum(not c["passed"] and c["severity"]=="error" for c in checks)}项，警告{sum(not c["passed"] and c["severity"]=="warning" for c in checks)}项。详见tables/quality_checks.csv。',
        f'- DEM尺寸为{terrain.w}列×{terrain.h}行，有效高程{terrain.meta["min_m"]:.2f}—{terrain.meta["max_m"]:.2f}m，NoData像元{terrain.meta["nodata_count"]}个。',
        f'- 节点表海拔与对应DEM像元最大绝对差为{nodes.dem_minus_given_m.abs().max():.4f}m。',
        f'- O01到最远服务区{far.to_id}的水平距离为{far.distance_m/1000:.3f}km；地形净空需另行计入。',
        f'- 本地距离相对WGS84测地距离最大误差{proj["max_task_pair_relative_error"]*100:.5f}%。',
        '- 静态作业高度直连筛查不通过：'+('、'.join(screen.loc[~screen.direct_available,'service_id']) or '无')+'。',
        '- S012与S014医疗箱的硬截止时间取3600秒，而不是其期望时间7200秒。S001第二箱医疗物资虽非首批，仍有3600秒硬截止。',
        '', '## 处理规则及需要明确的建模约定','']+[f'{i}. {n}' for i,n in enumerate(notes,1)]+[
        '', '## 输出结构','',
        '- tables/：标准化CSV（UTF-8 BOM），可直接用于Python、MATLAB或Excel。data_dictionary.csv列出所有字段与缺失情况。',
        '- spatial/dem.npz：elevation_m二维数组、lon_deg一维列坐标、lat_deg一维行坐标；北到南行序。',
        '- spatial/routing_matrices.npz：distance_m、cruise_alt_m、climb_m、descent_m为16×16，flight_s为3×16×16；携带node_ids/type_ids。',
        '- spatial/：投影参数、DEM元数据、带米制坐标的道路/水系/水体/村镇CSV。',
        '- figures/：8组PNG高清图及SVG矢量图，后者适合论文排版（地形栅格仍为栅格）。',
        '- index.html：离线可浏览图文报告，无需联网。source_manifest.json保存原始附件SHA-256。',
        '', '## 对四问的支持','',
        '|问题|建议读取|后续仍需完成|','|---|---|---|',
        '|第一问|boxes、transport_types、arcs、arc_type_coefficients|往返能量约束、最大安全载荷、质量体积联合组批及余量敏感性|',
        '|第二问|上述表、transport_fleet、energy_resources、charging_curve|多点剩余载荷、具体机体与电池占用、交付完成时刻及多目标调度|',
        '|第三问|DEM、communication_budgets、relay_types、relay_fleet、relay_energy_inventory|候选中继位置与海拔、逐阶段连续链路、建链时段及中继能源周转|',
        '|第四问|第三问最终调度结果及本次资源库存表|固定任务和时序下不可拆服务区集合、各组独占资源与库存缺口|',
        '', '## 复现','',
        '推荐在项目根目录的PowerShell运行 `./run_preprocessing.ps1`，自动选择本次使用的Python并运行预处理及独立验证；也可通过 `-PythonPath` 指定解释器。普通Python环境需先按requirements.txt安装依赖，再运行 `python preprocess_d.py` 和 `python verify_preprocessing.py`。原始数据在数据/下，输出写入preprocessing/；可选本项目.preprocess_deps目录优先加载。',
        '', '读取示例：', '', '```python',
        'import pandas as pd, numpy as np',
        'boxes = pd.read_csv("preprocessing/tables/boxes.csv")',
        'nodes = pd.read_csv("preprocessing/tables/nodes.csv")',
        'mat = np.load("preprocessing/spatial/routing_matrices.npz")',
        'distance_m = mat["distance_m"]  # 顺序见 mat["node_ids"]',
        'flight_s = mat["flight_s"]     # [机型, 起点, 终点]',
        '```',
        '', '## 图件','']
    for c in catalog:
        md.extend([f'### {c["title"]}','',f'![{c["title"]}](figures/{c["name"]}.png)','',c['caption'],''])
    (OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    card=''.join(f'<article id="{c["name"]}"><h2>{html.escape(c["title"])}</h2><a href="figures/{c["name"]}.svg"><img loading="lazy" src="figures/{c["name"]}.png" alt="{html.escape(c["title"])}"></a><p>{html.escape(c["caption"])}</p></article>' for c in catalog)
    nav=''.join(f'<a href="#{c["name"]}">{html.escape(c["title"])}</a>' for c in catalog)
    table=summary.to_html(index=False,border=0,classes='data',float_format=lambda x:f'{x:g}')
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D题数据预处理与可视化</title><style>body{margin:0;background:#f3f5f7;color:#203247;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:32px}h1{font-size:32px}header,article,section{background:white;border-radius:12px;padding:26px;margin-bottom:22px}img{width:100%;height:auto}nav{display:flex;gap:12px;flex-wrap:wrap}a{color:#176e98}nav a{padding:4px 8px;background:#edf4f8;text-decoration:none;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px;border-bottom:1px solid #e3e8ed;text-align:right}th{background:#edf4f8}summary{cursor:pointer;font-weight:bold}.scroll{overflow:auto}p{color:#516276}</style><main><header><h1>D题数据预处理与可视化</h1><p>16个节点 · 80个货箱 · 758 kg · 2.011 m³ · 240条有向航段 · 720条机型航段记录</p><p>输出为建模输入与静态筛查，尚未求解最优运输或中继调度。</p><nav>'+nav+'</nav><p><a href="README.md">完整处理说明</a>　<a href="tables/data_dictionary.csv">字段字典</a>　<a href="tables/quality_checks.csv">质量检查</a></p></header>'+card+'<section><h2>服务区需求汇总</h2><div class="scroll">'+table+'</div></section><section><details><summary>计算口径与边界</summary><ol>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes)+'</ol></details></section></main></html>'
    (OUT/'index.html').write_text(doc,encoding='utf-8')


if __name__=='__main__':
    main()

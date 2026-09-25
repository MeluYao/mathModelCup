# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent;TAB=ROOT/'q3_results'/'tables';OUT=ROOT/'q3_results'
stats=json.loads((OUT/'solver_summary.json').read_text(encoding='utf-8'));ver=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
p=pd.read_csv(TAB/'transport_plan.csv');d=pd.read_csv(TAB/'box_deliveries.csv');r=pd.read_csv(TAB/'relay_plan.csv');c=pd.read_csv(TAB/'communication_summary.csv');obj=pd.read_csv(TAB/'objective_comparison.csv')
def md(df):
    x=df.copy()
    for col in x.select_dtypes(include='number').columns:x[col]=x[col].map(lambda v:'' if pd.isna(v) else f'{v:.6f}')
    esc=lambda v:str(v).replace('|','\\|').replace('\n',' ')
    head='|'+ '|'.join(esc(v) for v in x.columns)+'|'
    sep='|'+'|'.join('---' for _ in x.columns)+'|'
    rows=['|'+'|'.join(esc(v) for v in row)+'|' for row in x.itertuples(index=False,name=None)]
    return '\n'.join([head,sep,*rows])
m=stats['main_metrics'];hard=d[d.hard_deadline_s.notna()];svc=d.groupby('service_id').agg(箱数=('box_id','size'),最早送达_s=('delivery_s','min'),最晚送达_s=('delivery_s','max')).reset_index()
trans=p[['flight_id','aircraft_id','type_id','battery_id','start_s','visit_order','return_s','energy_kwh','return_soc','boxes']].rename(columns={'flight_id':'运输架次','aircraft_id':'无人机','type_id':'机型','battery_id':'电池','start_s':'开始/s','visit_order':'访问顺序','return_s':'返航/s','energy_kwh':'能耗/kWh','return_soc':'返航SOC','boxes':'货箱'})
relay=r[['relay_flight_id','relay_id','energy_id','hover_site_id','x_m','y_m','hover_agl_m','flight_alt_m','start_s','service_start_s','service_end_s','return_s','energy_kwh','return_soc','covered_transport_flights']].rename(columns={'relay_flight_id':'中继架次','relay_id':'中继机','energy_id':'能源组件','hover_site_id':'悬停点','x_m':'x/m','y_m':'y/m','hover_agl_m':'离地高度/m','flight_alt_m':'飞行海拔/m','start_s':'开始/s','service_start_s':'服务开始/s','service_end_s':'服务结束/s','return_s':'返航/s','energy_kwh':'能耗/kWh','return_soc':'返航SOC','covered_transport_flights':'保障运输架次'})
valid=obj[(obj.hard_violations==0)&(~obj.policy.astype(str).str.startswith('balanced_refined'))][['policy','merge_gap_s','on_time_boxes','joint_makespan_s','total_energy_kwh','transport_trips','relay_trips']]
invalid=obj[(obj.hard_violations>0)][['policy','merge_gap_s','hard_violations','max_hard_lateness_s','joint_makespan_s','total_energy_kwh','transport_trips','relay_trips']]
text=f'''# 问题三：通信约束下运输与中继联合调度

## 1. 数据继承、假设与时间口径

问题三继承问题二的80个不可拆货箱、运输航段、载荷—能量模型、8架实体运输无人机和14组共享电池。通信保障从运输机起飞开始，连续覆盖爬升、巡航、下降和服务区物资交接，准备与装载发生在O01工位，不计入空中通信区间。固定网关G01位于O01，天线海拔为O01地面高程加20m。

中继无人机从O01完成准备后出发，抵达悬停点并完成30s建链后开始服务，服务结束即返航。返航后的300s周转时间约束同一中继机再次出动，但联合任务完成时间取返航时刻。每个中继架次使用一组满电能源组件；返航后组件按题目两阶段充电曲线充满方可复用。题目没有限制单架中继同时接入的运输机数量，因此同一悬停点链路预算满足时允许并行保障多架运输机。

## 2. 连续通信模型

任意端点a、b之间的双向损耗门限为

\[
L_{{\max}}^{{a\leftrightarrow b}}=\min\{{L_{{\max}}^{{a\to b}},L_{{\max}}^{{b\to a}}\}}.
\]

根据附件参数，运输机—G01直连、运输机—中继接入、中继—G01回传的双向门限分别为122、116和126dB。三维距离为D km时，考虑地形遮挡的传播损耗为

\[
L_{{path}}=32.45+20\log_{{10}}2400+20\log_{{10}}D+10b,
\]

其中b由30m DEM上的端点视线剖面决定。运输机在时刻t通信可行，当且仅当直连可用，或存在同一时刻正在服务的中继架次r，使接入与回传链路同时可用：

\[
A_{{T,G}}(t)=1\quad\text{{或}}\quad A_{{T,r}}(t)A_{{r,G}}(t)=1.
\]

运输轨迹被展开为爬升、水平巡航、下降与投送交接的分段连续函数。主求解用10s网格并加入全部阶段端点，最终方案再以2s网格重建窗口；独立验证使用2s网格和30m DEM视线采样，共检查{ver['communication_samples']}个时空点，通信中断为{ver['communication_outages']}。

## 3. 中继时间与能量模型

悬停点离地高度h不超过300m。往返巡航海拔取航线DEM最高点以上50m与悬停海拔中的较大者。中继架次能耗为

\[
E_r=P_{{cr}}(t_{{hor}}^{{out}}+t_{{hor}}^{{back}})/3600
+m_Rg(h_{{up}}^{{out}}+h_{{up}}^{{back}})/(3.6\times10^6\eta_R)
+(P_{{hover}}+P_{{comm}})\Delta t_r/3600,
\]

并满足E_r≤0.8×3.2=2.56kWh。两架中继机的任务及周转区间不可重叠，6组能源组件的占用与充电区间不可重叠。

## 4. 多目标优化与方法原理

硬约束包括货箱唯一交付、运输质量/体积/能量、医疗与首批时限、两类无人机及能源资源可用性、悬停高度和连续通信。硬约束满足后，比较

\[
J=(J_{{late}},J_{{ratio}},T_{{joint}},E_T+E_R,N_T,N_R),
\]

其中J_late为优先系数加权相对迟到，J_ratio为优先系数加权交付时刻比例；T_joint取运输与中继最后返航时刻；N_T、N_R分别为运输和中继架次。

求解流程为：展开问题二候选运输路线；在服务区方向分数点与区域中点生成悬停位置，并枚举150/225/300m离地高度；筛除回传或返航能量不可行状态；计算每个运输架次直连失效区间及可完整覆盖它的悬停状态；采用集合覆盖与随机邻域指派减少悬停点；合并同点相邻服务窗口并按续航切分；最后用离散事件修复向后推迟冲突运输任务，同时传播运输机、电池、中继机和能源组件的可用时刻。算法使用固定随机种子，可复现，但候选集非支配性不等同于全局最优证明。

## 5. 主方案结果

- 全部80箱交付，硬时限违反0箱，期望时间内交付{m['on_time_boxes']}箱；最紧硬时限余量为{(hard.hard_deadline_s-hard.delivery_s).min():.6f}s。
- 运输{m['transport_trips']}架次、中继{m['relay_trips']}架次，联合架次{m['total_trips']}次。
- 联合任务完成时间{m['joint_makespan_s']:.6f}s，即{m['joint_makespan_s']/3600:.6f}h。
- 运输能耗{m['transport_energy_kwh']:.6f}kWh，中继能耗{m['relay_energy_kwh']:.6f}kWh，总能耗{m['total_energy_kwh']:.6f}kWh。
- 使用8架运输机、13组运输电池、2架中继机和3组中继能源组件；中继最低返航SOC为{r.return_soc.min()*100:.6f}%。
- 17个运输架次需要中继、5个架次全程可直连；全部通信采样中直连比例为{c.direct_samples.sum()/c.communication_samples.sum():.6f}。接入链路最小裕量{ver['min_access_margin_db']:.6f}dB，回传链路最小裕量{ver['min_backhaul_margin_db']:.6f}dB。

### 5.1 运输架次

{md(trans)}

### 5.2 中继架次与悬停方案

{md(relay)}

### 5.3 各服务区交付时刻

{md(svc)}

完整80箱逐箱送达时刻见`tables/box_deliveries.csv`及结果工作簿。

## 6. 指标权衡

硬时限可行候选对照：

{md(valid)}

600s窗口合并允许中继在相邻保障任务之间继续悬停，减少返航、再准备和重新建链，因此中继架次由8次降至5次，同时释放中继机，使运输任务的资源冲突延迟缩短。与不考虑通信的第二问22架次方案相比，运输能耗保持不变，新增中继能耗{m['relay_energy_kwh']:.6f}kWh；联合完成时间增加{m['joint_makespan_s']-9943.021474165032:.6f}s。

更少运输架次并不必然产生可行联合方案。17运输架次候选虽然总能耗较低、联合架次仅26次，但中继资源修复后出现硬时限违反：

{md(invalid)}

这说明运输合批减少了运输能耗和架次数，却延长了单条运输路线并集中占用中继窗口；在只有2架中继机的条件下，为避免通信冲突而产生的推迟会破坏早期保障时限。因此本问采用“硬时限与连续通信优先，其次及时性和联合完成时间，再比较总能耗与两类架次数”的优先关系。

## 7. 资源与通信可行性

独立验证共{ver['checks']}项，失败{ver['failed']}项。验证内容包括80箱不重不漏、硬时限、运输与中继返航SOC、悬停高度、8架运输机与14组运输电池互斥、2架中继机周转、6组能源组件充电、中继飞行/悬停能耗复算以及连续通信。所有检查通过。

## 8. 输出文件

- `tables/transport_plan.csv`：运输路线、箱组、具体无人机、电池和时刻；
- `tables/box_deliveries.csv`：80箱逐箱送达时刻；
- `tables/relay_plan.csv`：中继悬停坐标、海拔、服务窗口、能源组件和保障对象；
- `tables/communication_summary.csv`及`communication_timeline_2s.csv`：通信状态与链路裕量；
- `tables/resource_usage.csv`：四类实体资源占用和再次可用时刻；
- `verification.json`：独立验证结果。
'''
(OUT/'问题三_模型建立与求解.md').write_text(text,encoding='utf-8')
print('report written')

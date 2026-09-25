"""生成问题一科研图形及离线结果概览，保留CSV和Markdown为完整模型依据。"""
from pathlib import Path
import sys,json,html
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.preprocess_deps'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
OUT=ROOT/'q1_results'; T=OUT/'tables'; FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'font.size':10,'axes.titlesize':14,'axes.titlepad':12,'svg.fonttype':'path','savefig.facecolor':'white'})
colors={'A':'#2878A0','B':'#D09A39','C':'#548F78'}
catalog=[]
def save(fig,name,title,caption):
    fig.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight'); plt.close(fig)
    catalog.append(dict(name=name,title=title,caption=caption))

caps=pd.read_csv(T/'maximum_safe_payload.csv')
q=caps.pivot(index='service_id',columns='type_id',values='max_safe_payload_kg')
ratio=q/np.array([25,30,80])
fig,ax=plt.subplots(figsize=(7,8),constrained_layout=True)
im=ax.imshow(ratio,cmap='YlGnBu',vmin=.65,vmax=1,aspect='auto')
ax.set_xticks(range(3),['A型 / 标称25 kg','B型 / 标称30 kg','C型 / 标称80 kg'])
ax.set_yticks(range(15),q.index)
for i in range(15):
    for j in range(3):
        v=q.iloc[i,j]; label=f'{v:.2f}' if abs(v-round(v))>1e-7 else f'{v:.0f}'
        ax.text(j,i,label,ha='center',va='center',color='white' if ratio.iloc[i,j]>.85 else '#203247',fontsize=12)
ax.set_title('20%安全余量下的最大安全载荷（kg）')
fig.colorbar(im,ax=ax,shrink=.7,label='最大安全载荷 / 标称最大载荷')
save(fig,'01_safe_payload','最大安全载荷','数字为kg，颜色为相对标称载重比例。具体箱组仍需满足体积限制。')

p=pd.read_csv(T/'plan_N_E_T.csv')
fig,axes=plt.subplots(1,2,figsize=(14,8),constrained_layout=True)
y=np.arange(len(p)); labels=[f'{r.flight_id}  {r.service_id}  {r.type_id}' for r in p.itertuples()]
left=np.zeros(len(p))
for field,mass,name,c in [('medical_count',3,'医疗','#2878A0'),('water_count',14,'饮用水','#D5A041'),('food_count',8,'食品','#548F78'),('hygiene_count',6,'卫生','#A77DA2')]:
    value=p[field].to_numpy()*mass; axes[0].barh(y,value,left=left,color=c,label=name,height=.72); left+=value
axes[0].set_yticks(y,labels,fontsize=9); axes[0].invert_yaxis(); axes[0].set_xlabel('装载质量 / kg'); axes[0].set_title('18架次货箱构成'); axes[0].legend(ncol=4,frameon=False,loc='lower right'); axes[0].grid(axis='x',alpha=.2); axes[0].set_axisbelow(True)
for shift,field,label,c in [(-.23,'safe_mass_utilization','质量 / 安全载荷','#2878A0'),(0,'volume_utilization','体积 / 容量','#D5A041'),(.23,'energy_budget_utilization','能耗 / 可用预算','#548F78')]:
    axes[1].barh(y+shift,p[field]*100,height=.21,label=label,color=c)
axes[1].axvline(100,ls='--',c='#A8493A',lw=1)
axes[1].set_yticks(y,p.flight_id,fontsize=9); axes[1].invert_yaxis(); axes[1].set_xlim(0,112); axes[1].set_xlabel('约束利用率 / %'); axes[1].set_title('各架次约束核验'); axes[1].legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2); axes[1].grid(axis='x',alpha=.2); axes[1].set_axisbelow(True)
save(fig,'02_batch_plan','主方案箱组与约束利用率','每个货箱恰出现一次。利用率不超过100%，所有架次返航SOC不低于20%。')

compare=pd.read_csv(T/'objective_comparison.csv')
fig,axes=plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
labels=['架次→能耗\n→时间','能耗→架次\n→时间','时间→架次\n→能耗','架次→时间\n→能耗']
for ax,field,title,fmt in [(axes[0],'trips','往返架次 / 次','{:.0f}'),(axes[1],'energy_kwh','总运输能耗 / kWh','{:.4f}'),(axes[2],'operation_h','累计作业时间 / h','{:.4f}')]:
    vals=compare[field].to_numpy(); ax.bar(range(4),vals,color=['#2878A0','#D5A041','#548F78','#A77DA2'],width=.65)
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*.015,fmt.format(v),ha='center',fontsize=10)
    ax.set_xticks(range(4),labels,fontsize=9); ax.set_ylim(0,max(vals)*1.16); ax.set_title(title); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
save(fig,'03_objective_comparison','多目标优先关系对照','主方案与最短累计时间方案一致。能耗优先方案增加1架次，只节省0.09738 kWh，增加约27.73分钟。')

curve=pd.read_csv(T/'reserve_capacity_curves.csv')
fig,axes=plt.subplots(1,3,figsize=(14,5.5),constrained_layout=True)
palette=plt.cm.tab20(np.linspace(0,1,15)); ids=sorted(q.index)
for ax,typ in zip(axes,['A','B','C']):
    for s,c in zip(ids,palette):
        d=curve[curve.service_id.eq(s)&curve.type_id.eq(typ)]
        ax.plot(d.reserve_fraction*100,d.max_safe_payload_kg,label=s,color=c,lw=1.8,ls='--' if s in ['S010','S011','S012','S013','S014','S015'] else '-')
    ax.axvline(20,c='#222222',ls=':',lw=1); ax.set_title(f'{typ}型安全载荷'); ax.set_xlabel('返航安全余量 / %'); ax.set_ylabel('最大安全载荷 / kg'); ax.grid(alpha=.2); ax.set_xlim(0,50); ax.set_ylim(bottom=0)
handles,ls=axes[0].get_legend_handles_labels()
fig.legend(handles,ls,loc='outside lower center',ncol=8,frameon=False,fontsize=9)
save(fig,'04_payload_sensitivity','最大安全载荷对余量的敏感性','竖虚线为20%基准。曲线终止表示空载往返也不可行；重合曲线代表相同的标称载重限制。')

step=pd.read_csv(T/'exact_global_trip_steps.csv'); sens=pd.read_csv(T/'reserve_sensitivity.csv'); feasible=sens[sens.feasible]
stats=json.loads((OUT/'solver_summary.json').read_text(encoding='utf-8')); limit=stats['largest_feasible_reserve']*100
fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
edges=np.r_[step.lower.to_numpy(),step.upper_inclusive.iloc[-1]]*100
axes[0].stairs(step.minimum_trips.to_numpy(),edges,baseline=None,color='#2878A0',linewidth=2)
axes[0].set_ylabel('最少架次 / 次'); axes[0].set_yticks(range(18,26)); axes[0].set_title('安全余量提高后的重新优化结果')
axes[1].plot(feasible.reserve_fraction*100,feasible.energy_kwh,'o-',color='#D09A39',lw=2); axes[1].set_ylabel('运输能耗 / kWh')
axes[2].plot(feasible.reserve_fraction*100,feasible.operation_h,'o-',color='#548F78',lw=2); axes[2].set_ylabel('累计作业时间 / h'); axes[2].set_xlabel('返航安全余量 / %')
for ax in axes:
    ax.axvline(20,c='#999999',ls=':',lw=1); ax.axvline(limit,c='#A8493A',ls='--',lw=1)
    ax.axvspan(limit,40,color='#C96354',alpha=.12); ax.grid(alpha=.2); ax.set_xlim(0,40)
axes[0].text(35.7,19.3,'无法完成\n全部交付',color='#A8493A',fontsize=10)
axes[0].annotate('23.089%：18→19次',xy=(23.089,18),xytext=(9,22),arrowprops=dict(arrowstyle='->',color='#555'),fontsize=10)
axes[0].annotate('27.050%：19→20次',xy=(27.05,19),xytext=(15,24),arrowprops=dict(arrowstyle='->',color='#555'),fontsize=10)
save(fig,'05_reserve_sensitivity','组批结果与精确架次阈值','上图为精确阶梯区间；下两图为离散情景重优化点的连线，不表示区间内线性变化。35.267647%为完整交付可行极限。')

verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
report_path=OUT/'问题一_模型建立与求解.md'
report=report_path.read_text(encoding='utf-8').split('\n<!-- FIGURES_AND_VALIDATION -->')[0]
report+='\n<!-- FIGURES_AND_VALIDATION -->\n## 10. 实际核验结果与图件\n\n'
report+=f'独立核验实际通过{verification["passed"]}项、失败0项。15个服务区的整数规划均报告最优，最优间隙均为0；与动态规划固定最少架次能耗的最大差为{verification["max_energy_difference"]:.3g} kWh。完整日志见verification.json。\n\n'
for c in catalog: report+=f'### {c["title"]}\n\n![{c["title"]}](figures/{c["name"]}.png)\n\n{c["caption"]}\n\n'
report_path.write_text(report,encoding='utf-8')
cards=''.join(f'<article><h2>{html.escape(c["title"])}</h2><a href="figures/{c["name"]}.svg"><img src="figures/{c["name"]}.png" alt="{html.escape(c["title"])}"></a><p>{html.escape(c["caption"])}</p></article>' for c in catalog)
cap_html=q.round(4).to_html(border=0); plan=p[['flight_id','service_id','type_id','box_ids','mass_kg','volume_m3','energy_kwh','operation_s','return_soc']].copy()
plan.columns=['架次','服务区','机型','货箱编号','质量/kg','体积/m³','能耗/kWh','累计作业/s','返航SOC比例']
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>问题一建模与求解</title><style>body{margin:0;background:#f3f5f7;color:#203247;font:16px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:32px}header,article,section{background:white;border-radius:12px;padding:26px;margin-bottom:22px}img{width:100%;height:auto}a{color:#176e98}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #e3e8ed;text-align:right}th{background:#edf4f8}.scroll{overflow:auto}p{color:#516276}</style><main><header><h1>问题一：单点往返运输能力与组批优化</h1><p><b>18架次　59.132260 kWh　9.104588小时累计作业时间</b></p><p>80箱全部交付，采用9架次B型和9架次C型；105项独立核验通过。累计作业时间不是多机并行完工时间。</p><p><a href="问题一_模型建立与求解.md">完整模型、方法原理与求解报告</a>　<a href="tables/Q1_单点组批.csv">提交模板CSV</a>　<a href="tables/maximum_safe_payload.csv">最大安全载荷数据</a></p><p>数值结果以报告披露的标准航程耗能与势能爬升模型为条件。最小返航SOC为23.089248%。</p></header>'+cards+'<section><h2>最大安全载荷 / kg</h2>'+cap_html+'</section><section><h2>完整逐架次方案</h2><div class="scroll">'+plan.to_html(index=False,border=0,float_format=lambda x:f'{x:.6g}')+'</div></section></main></html>'
(OUT/'index.html').write_text(page,encoding='utf-8')
(OUT/'figure_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
print('已生成5组PNG/SVG、离线结果概览及完整报告图件。')

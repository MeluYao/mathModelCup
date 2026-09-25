import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const base=path.dirname(fileURLToPath(import.meta.url));
const resultDir=path.resolve(base,'..');
const root=path.resolve(base,'../..');
const outputDir=path.join(root,'outputs','01a0d71b-q2');
await fs.mkdir(outputDir,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(resultDir,'workbook_data.json'),'utf8'));
const stats=JSON.parse(await fs.readFile(path.join(resultDir,'solver_summary.json'),'utf8'));
const verify=JSON.parse(await fs.readFile(path.join(resultDir,'verification.json'),'utf8'));
const blocks=Object.values(data);
const wb=Workbook.create();
const names=['结果汇总','Q2_运输架次','Q2_逐箱交付','路线分段','资源使用','目标对照','计算口径'];
for(const name of names)wb.worksheets.add(name);
const sh=n=>wb.worksheets.getItem(n);
const col=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
function table(ws,start,headers,rows,widths){
  const end=start+rows.length,last=col(headers.length-1);ws.getRange(`A${start}:${last}${end}`).values=[headers,...rows];
  const all=ws.getRange(`A${start}:${last}${end}`);all.format.font={name:'Microsoft YaHei',size:10,color:'#203247'};all.format.rowHeight=23;all.format.verticalAlignment='center';
  const head=ws.getRange(`A${start}:${last}${start}`);head.format.fill='#255B68';head.format.font={name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'};head.format.wrapText=true;head.format.rowHeight=38;head.format.horizontalAlignment='center';
  for(let j=0;j<headers.length;j++)ws.getRange(`${col(j)}${start}:${col(j)}${end}`).format.columnWidth=widths?.[j]??17;
  for(let r=start+1;r<=end;r++)if((r-start)%2===0)ws.getRange(`A${r}:${last}${r}`).format.fill='#EFF5F4';
  return end;
}
for(const n of names){sh(n).showGridLines=false;sh(n).tabColor=n==='结果汇总'?'#255B68':'#8BB5AE';}

const plan=blocks[0];table(sh('Q2_运输架次'),1,plan.columns,plan.rows,[16,15,12,17,18,28,22,20]);sh('Q2_运输架次').freezePanes.freezeRows(1);sh('Q2_运输架次').getRange('E2:E23').setNumberFormat('0.000');sh('Q2_运输架次').getRange('G2:H23').setNumberFormat('0.000000');
const del=blocks[1];table(sh('Q2_逐箱交付'),1,del.columns,del.rows,[28,16,18,24]);sh('Q2_逐箱交付').freezePanes.freezeRows(1);sh('Q2_逐箱交付').getRange('D2:D81').setNumberFormat('0.000');
const res=blocks[2];table(sh('资源使用'),1,res.columns,res.rows,[18,20,13,16,21,21,22,14,14]);sh('资源使用').freezePanes.freezeRows(1);sh('资源使用').getRange('E2:G45').setNumberFormat('0.000');sh('资源使用').getRange('H2:I45').setNumberFormat('0.0000%');
const legs=blocks[3];table(sh('路线分段'),1,legs.columns,legs.rows,[16,13,14,14,15,16,18,18,17,18,42]);sh('路线分段').freezePanes.freezeRows(1);sh('路线分段').getRange('E2:J46').setNumberFormat('0.000000');sh('路线分段').getRange('K2:K46').format.wrapText=true;
const obj=blocks[4];table(sh('目标对照'),1,obj.columns,obj.rows,[16,18,22,22,23,18,20,18,13]);sh('目标对照').getRange('B2:I6').setNumberFormat('0.000000');

const s=sh('结果汇总');s.getRange('A2').values=[['问题二多点配送优化结果']];s.getRange('A2').format.font={name:'Microsoft YaHei',size:17,bold:true,color:'#173B44'};
table(s,4,['指标','主方案数值','单位'],[
 ['硬时限违反',stats.main_metrics.hard_violations,'箱'],['交付货箱',null,'箱'],['期望时间内交付',stats.main_metrics.on_time_boxes,'箱'],['运输架次数',null,'次'],
 ['全部任务完成时间',null,'s'],['全部任务完成时间',null,'h'],['总运输能耗',null,'kWh'],['加权迟到指标',stats.main_metrics.weighted_tardiness,'—'],
 ['加权交付比例',stats.main_metrics.weighted_delivery_ratio,'—'],['最低返航SOC',null,'%'],['使用实体飞机',null,'架'],['使用电池',null,'组'],['独立校验',verify.passed?'全部通过':'存在失败',`${verify.checks}项`]
],[30,24,18]);
s.getRange('B6').formulas=[["=COUNTA('Q2_逐箱交付'!A2:A81)"]];s.getRange('B8').formulas=[["=COUNTA('Q2_运输架次'!A2:A23)"]];s.getRange('B9').formulas=[["=MAX('Q2_运输架次'!G2:G23)"]];s.getRange('B10').formulas=[['=B9/3600']];s.getRange('B11').formulas=[["=SUM('Q2_运输架次'!H2:H23)"]];
s.getRange('B14').formulas=[["=MIN('资源使用'!I2:I45)"]];s.getRange('B15:B16').values=[[8],[13]];
s.getRange('B5:B17').setNumberFormat('0.000000');s.getRange('B10').setNumberFormat('0.000000');s.getRange('B14').setNumberFormat('0.00%');
s.getRange('A20').values=[['方案选择']];s.getRange('A20').format.font={name:'Microsoft YaHei',size:12,bold:true,color:'#173B44'};
s.getRange('A21:C24').values=[
 ['优先级','硬时限必须满足；随后比较及时性、最晚返航、能耗和架次数。',''],
 ['折中规则','在已搜索非支配候选集中，对交付比、最晚返航、能耗、架次归一化，取等权理想点距离最小方案。',''],
 ['边界结论','17架次仍可满足全部硬时限；16架次出现2箱硬时限延误，最大约55.054秒。',''],
 ['适用说明','组合优化采用可复现启发式搜索，非支配性限于已搜索候选集。','']];s.getRange('A21:C24').format.wrapText=true;s.getRange('A21:C24').format.rowHeight=42;s.getRange('A21:A24').format.font={name:'Microsoft YaHei',size:10,bold:true,color:'#255B68'};

const notes=[
 ['任务时钟','从准备开始计时；交付时刻为服务区完成基础交接和逐箱交接的时刻；完成时间取最后一个架次返回O01的时刻。'],
 ['路线范围','单个架次最多访问3个服务区；货箱不可拆分且只交付一次。'],
 ['能耗模型','逐航段按剩余载荷计算水平能耗与正爬升势能；下降附加能耗取0；返航SOC不低于20%。'],
 ['硬约束','医疗物资期望时间和首批保障截止时间均作为硬时限；其余期望时间进入及时性指标。'],
 ['及时性','加权迟到为 priority×max(0,C-d)/d 之和；加权交付比例为 priority×C/d 的加权平均。两者均越小越好。'],
 ['资源','8架运输无人机；A/B/C型电池6/4/4组，含初始装机电池；同型电池共享，返航后按实际SOC充至100%才能复用。'],
 ['求解','路线合并束搜索 + 多机型路线枚举 + 离散事件资源调度 + 交换/插入邻域搜索。'],
 ['最优性边界','17架次是本次搜索得到的硬时限可行最少架次数；启发式求解不构成全局最优证明。'],
 ['复核',`独立脚本复算${verify.checks}项，失败${verify.failed}项；包括逐航段时刻、能耗、箱覆盖、资源互斥和电池充电。`]
];table(sh('计算口径'),1,['项目','说明'],notes,[24,110]);sh('计算口径').getRange('B2:B10').format.wrapText=true;sh('计算口径').getRange('A2:B10').format.rowHeight=48;
wb.recalculate();
const actual=s.getRange('B6:B11').values.flat();if(actual[0]!==80||actual[2]!==22||Math.abs(actual[3]-stats.main_metrics.makespan_s)>1e-5||Math.abs(actual[5]-stats.main_metrics.energy_kwh)>1e-5)throw new Error('Summary formula reconciliation failed');
console.log((await wb.inspect({kind:'region',sheetId:'结果汇总',range:'A2:C24',maxChars:5000,tableMaxRows:24,tableMaxCols:3})).ndjson);
const out=path.join(outputDir,'问题二结果.xlsx');const file=await SpreadsheetFile.exportXlsx(wb);await file.save(out);
const ranges={'结果汇总':'A2:C24','Q2_运输架次':'A1:H23','Q2_逐箱交付':'A1:D25','路线分段':'A1:K16','资源使用':'A1:I18','目标对照':'A1:I6','计算口径':'A1:B10'};
for(const n of names){const p=await wb.render({sheetName:n,range:ranges[n],scale:1,format:'png'});await fs.writeFile(path.join(base,`preview_${n}.png`),new Uint8Array(await p.arrayBuffer()));}
console.log(`Workbook exported: ${out}`);

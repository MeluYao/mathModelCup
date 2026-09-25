import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const base=path.dirname(fileURLToPath(import.meta.url));
const resultDir=path.resolve(base,'..');
const root=path.resolve(base,'../..');
const outputDir=path.join(root,'outputs','01a0d71b-q1');
await fs.mkdir(outputDir,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(resultDir,'workbook_data.json'),'utf8'));
const stats=JSON.parse(await fs.readFile(path.join(resultDir,'solver_summary.json'),'utf8'));
const verify=JSON.parse(await fs.readFile(path.join(resultDir,'verification.json'),'utf8'));
const wb=Workbook.create();
const names=['结果汇总','Q1_单点组批','最大安全载荷','目标与余量对照','逐箱归属','逐区汇总','计算口径'];
for(const name of names)wb.worksheets.add(name);
const sheet=name=>wb.worksheets.getItem(name);
const col=n=>{let result='';for(n++;n;n=Math.floor((n-1)/26))result=String.fromCharCode(65+(n-1)%26)+result;return result;};
function table(sh,start,headers,rows,widths){
  const end=start+rows.length,last=col(headers.length-1);
  sh.getRange(`A${start}:${last}${end}`).values=[headers,...rows];
  const all=sh.getRange(`A${start}:${last}${end}`);
  all.format.font={name:'Microsoft YaHei',size:10,color:'#203247'};
  all.format.rowHeight=24;all.format.verticalAlignment='center';
  const head=sh.getRange(`A${start}:${last}${start}`);
  head.format.fill='#284D65';head.format.font={name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'};
  head.format.wrapText=true;head.format.rowHeight=40;head.format.horizontalAlignment='center';
  for(let j=0;j<headers.length;j++)sh.getRange(`${col(j)}${start}:${col(j)}${end}`).format.columnWidth=widths?.[j]??18;
  for(let r=start+1;r<=end;r++)if((r-start)%2===0)sh.getRange(`A${r}:${last}${r}`).format.fill='#F0F4F7';
  return end;
}
for(const name of names){sheet(name).showGridLines=false;sheet(name).tabColor=name==='结果汇总'?'#284D65':'#93ACBB';}

const plan=data['Q1_单点组批'];
const planRows=plan.rows.map(r=>{const v=[...r];const ids=v[3].split(';');const lines=[];for(let i=0;i<ids.length;i+=3)lines.push(ids.slice(i,i+3).join('; '));v[3]=lines.join(';\n');return v;});
table(sheet('Q1_单点组批'),1,plan.columns,planRows,[14,14,12,60,15,16,18,18,18]);
sheet('Q1_单点组批').freezePanes.freezeRows(1);
sheet('Q1_单点组批').getRange('D2:D19').format.wrapText=true;
for(let i=0;i<planRows.length;i++)sheet('Q1_单点组批').getRange(`A${i+2}:I${i+2}`).format.rowHeight=Math.max(26,planRows[i][3].split('\n').length*17+8);
sheet('Q1_单点组批').getRange('E2:E19').setNumberFormat('0.00');
sheet('Q1_单点组批').getRange('F2:F19').setNumberFormat('0.000');
sheet('Q1_单点组批').getRange('G2:I19').setNumberFormat('0.000000');

const summary=sheet('结果汇总');
summary.getRange('A2').values=[['问题一求解结果']];summary.getRange('A2').format.font={name:'Microsoft YaHei',size:16,bold:true};
table(summary,4,['指标','主方案数值','单位'],[
  ['返航安全余量',.2,'比例'],['往返架次数',null,'次'],['总运输能耗',null,'kWh'],
  ['累计作业时间',null,'s'],['累计作业时间',null,'h'],['交付货箱',null,'箱'],
  ['总质量',null,'kg'],['总体积',null,'m³'],['最低返航SOC',null,'%'],
  ['B型使用架次',null,'次'],['C型使用架次',null,'次'],['独立核验通过项数',verify.passed,'项']
],[28,24,16]);
summary.getRange('B5').setNumberFormat('0.0%');
summary.getRange('B6').formulas=[["=COUNTA('Q1_单点组批'!A2:A19)"]];
summary.getRange('B7').formulas=[["=SUM('Q1_单点组批'!H2:H19)"]];
summary.getRange('B8').formulas=[["=SUM('Q1_单点组批'!G2:G19)"]];
summary.getRange('B9').formulas=[['=B8/3600']];
summary.getRange('B10').formulas=[["=COUNTA('逐箱归属'!A2:A81)"]];
summary.getRange('B11').formulas=[["=SUM('Q1_单点组批'!E2:E19)"]];
summary.getRange('B12').formulas=[["=SUM('Q1_单点组批'!F2:F19)"]];
summary.getRange('B13').formulas=[["=MIN('Q1_单点组批'!I2:I19)"]];
summary.getRange('B14').formulas=[["=COUNTIFS('Q1_单点组批'!C2:C19,\"B\")"]];
summary.getRange('B15').formulas=[["=COUNTIFS('Q1_单点组批'!C2:C19,\"C\")"]];
summary.getRange('B7:B9').setNumberFormat('0.000000');summary.getRange('B12:B13').setNumberFormat('0.000000');
summary.getRange('A18').values=[['主目标：先最少架次，再最低能耗，最后最短累计作业时间。']];
summary.getRange('A19').values=[['累计作业时间包含准备、装载、飞行、交接，不是并行完工时间。']];
summary.getRange('A20').values=[['工作簿为求解结果快照；修改输入后需重新运行优化程序。']];
summary.getRange('A18:C20').format.font={name:'Microsoft YaHei',size:10,color:'#516276'};

const capacity=data['最大安全载荷'];
const grouped=new Map();
for(const r of capacity.rows){if(!grouped.has(r[0]))grouped.set(r[0],[r[0],null,null,null]);grouped.get(r[0])[{A:1,B:2,C:3}[r[1]]]=r[2];}
table(sheet('最大安全载荷'),1,['服务区','A型安全载荷/kg','B型安全载荷/kg','C型安全载荷/kg'],[...grouped.values()],[16,24,24,24]);
sheet('最大安全载荷').getRange('B2:D16').setNumberFormat('0.0000');sheet('最大安全载荷').freezePanes.freezeRows(1);
sheet('最大安全载荷').getRange('A18').values=[['安全余量20%；本表为质量—能量上限，装箱还须满足体积约束。']];

const comp=data['目标对照'];
table(sheet('目标与余量对照'),1,['优先顺序','架次','能耗/kWh','累计作业/h','A架次','B架次','C架次'],comp.rows.map(r=>[r[1],r[2],r[3],r[5],r[7],r[8],r[9]]),[34,12,20,20,12,12,12]);
const sens=data['安全余量敏感性'];
table(sheet('目标与余量对照'),8,['安全余量','全部可行','架次','能耗/kWh','累计作业/h','无法完成的服务区'],sens.rows.map(r=>[r[0],r[1]?'是':'否',r[2],r[3],r[5],r[6]||'']),[34,12,20,20,18,40]);
sheet('目标与余量对照').getRange('A9:A22').setNumberFormat('0.0%');
sheet('目标与余量对照').getRange('C2:D5').setNumberFormat('0.000000');
sheet('目标与余量对照').getRange('D9:E22').setNumberFormat('0.000000');
sheet('目标与余量对照').getRange('F9:F22').format.wrapText=true;
sheet('目标与余量对照').getRange('A9:F22').format.rowHeight=34;

const boxes=data['逐箱归属'];
table(sheet('逐箱归属'),1,['货箱编号','架次编号','服务区','机型'],boxes.rows.map(r=>[r[1],r[2],r[3],r[4]]),[26,18,16,14]);
sheet('逐箱归属').freezePanes.freezeRows(1);
const regions=data['逐区汇总'];
table(sheet('逐区汇总'),1,['服务区','架次','能耗/kWh','累计作业/s','质量/kg','体积/m³','箱数','最低SOC','机型'],regions.rows,[14,12,20,22,16,16,12,18,16]);
sheet('逐区汇总').getRange('C2:F16').setNumberFormat('0.000000');sheet('逐区汇总').getRange('H2:H16').setNumberFormat('0.0000%');sheet('逐区汇总').freezePanes.freezeRows(1);
const notes=[
 ['来源','题目附件：运输无人机数据.xlsx、物资需求与配送时限.xlsx、调度中心与服务区.xlsx及30米DEM；经预处理后求解。'],
 ['范围','仅O01—一个服务区—O01。第一问不限制实体机数、电池数或充电周转，不施加第二问配送时限。'],
 ['能耗约定','Ehor=Euse*d/L(q)；Eup=(m0+q)*9.80665*h/(eta*3.6e6)。去程载货、返程空载，下降附加能耗为0。'],
 ['推导说明','现有Markdown题面未完整展开上述两项能耗表达式，本结果以该物理推导约定为条件。'],
 ['时间口径','Q1_单点组批中的“往返时间”采用准备+装载+往返飞行+交接。纯飞行时间保存在详细CSV中。'],
 ['求解方法','完整模式枚举、数量状态动态规划。相同物性货箱压缩后求解，随后还原到原始80个唯一货箱ID。'],
 ['最优性','18架次达到质量体积下界；逐区MILP证明固定最少架次时能耗最优，全部最优间隙为0。'],
 ['安全余量','超过23.08924839%至少19架次；超过27.04974165%至少20架次；超过35.26764735%无法完整交付。'],
 ['精度','约束使用未舍入数值判断；表格仅显示有限小数。保留原始节点海拔，不用DEM值覆盖。'],
 ['结果类型','结果为静态优化快照；汇总指标与明细以公式链接，箱组本身需通过程序重新优化。']
];
table(sheet('计算口径'),1,['项目','说明'],notes,[20,105]);
sheet('计算口径').getRange('B2:B11').format.wrapText=true;sheet('计算口径').getRange('A2:B11').format.rowHeight=46;
wb.recalculate();
console.log((await wb.inspect({kind:'region',sheetId:'结果汇总',range:'A4:C16',maxChars:3000,tableMaxRows:13,tableMaxCols:3})).ndjson);
const expected=[18,stats.main_metrics[1],stats.main_metrics[2]];
const actual=summary.getRange('B6:B8').values.flat();
for(let i=0;i<3;i++)if(Math.abs(actual[i]-expected[i])>1e-5)throw new Error(`Summary mismatch ${actual[i]} vs ${expected[i]}`);
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(outputDir,'问题一结果.xlsx'));
const ranges={'结果汇总':'A2:C20','Q1_单点组批':'A1:I7','最大安全载荷':'A1:D18','目标与余量对照':'A1:G22','逐箱归属':'A1:D16','逐区汇总':'A1:I16','计算口径':'A1:B11'};
for(const name of names){const preview=await wb.render({sheetName:name,range:ranges[name],scale:1,format:'png'});await fs.writeFile(path.join(base,`preview_${name}.png`),new Uint8Array(await preview.arrayBuffer()));}
console.log('Workbook exported and 7 sheet previews rendered: '+path.join(outputDir,'问题一结果.xlsx'));

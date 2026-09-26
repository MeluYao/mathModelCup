# D题第二问历史四方案搜索摘要（非共同发布解）

> 本文件记录 2026-09-25 的历史搜索结果，其中 Hybrid 得到 24 架次方案。当前正式发布采用 `baseline/` 中的 22 架次共同发布解；本文件不作为最终推荐依据。

按统一词典序目标与同目标运行时间规则，本次历史运行推荐：hybrid。目标并列时优先选择无回退、运行时间更短且可解释性更强的方案。

method,validation,solver_status,normalized_weighted_delivery_time,makespan_s,energy_kwh,trip_count,runtime_s,incumbent_source,seed_retained,native_improved_seed,fallback
integrated_milp,PASS,FEASIBLE,0.516328617103963,9408.443943661947,64.83172808040707,22,8.434124200000001,provided_seed,True,False,
candidate,PASS,FEASIBLE,0.516328617103963,9408.443943661947,64.83172808040707,22,5.821643999999999,provided_seed,True,False,
alns,PASS,FEASIBLE,0.516328617103963,9408.443943661947,64.83172808040707,22,2.8645879,provided_seed,True,False,
hybrid,PASS,FEASIBLE,0.5158908995531455,9534.345058907296,69.52235221186811,24,7.408925899999996,hybrid_search,False,True,


说明：共同种子在各算法内部参与搜索；seed_retained=true 表示原生搜索未严格改进种子，不再进行报告层结果覆盖。

# D题第二问四方案比较摘要

当前最佳目标均保留共同种子，尚无算法严格改进该上界。若按保留种子后的原生运行时间，alns 最短；相同终值不构成全局最优证明。

method,validation,solver_status,normalized_weighted_delivery_time,makespan_s,energy_kwh,trip_count,runtime_s,incumbent_source,seed_retained,native_improved_seed,fallback
integrated_milp,PASS,FEASIBLE,0.516328617103963,9408.443943661947,64.83172808040707,22,3.1949018000000002,provided_seed,True,False,
candidate,PASS,FEASIBLE,0.516328617103963,9408.443943661947,64.83172808040707,22,2.9585226000000002,provided_seed,True,False,
alns,PASS,FEASIBLE,0.516328617103963,9408.443943661947,64.83172808040707,22,1.5527647000000009,provided_seed,True,False,
hybrid,PASS,FEASIBLE,0.516328617103963,9408.443943661947,64.83172808040707,22,4.3730607,provided_seed,True,False,


说明：共同种子在各算法内部参与搜索；seed_retained=true 表示原生搜索未严格改进种子，不再进行报告层结果覆盖。

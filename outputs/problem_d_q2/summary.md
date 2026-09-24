# D题第二问四方案比较摘要

当前最佳目标对应的入口均来自回退解，尚不能据此判定四种算法的寻优优劣。若只考虑阶段一保底运行时间，alns 最短；正式推荐需等待阶段二独立解。

method,validation,solver_status,normalized_weighted_delivery_time,makespan_s,energy_kwh,trip_count,runtime_s,fallback
integrated_milp,PASS,FEASIBLE,1.7068739241147075,37939.8995020773,174.0327124030489,76,5.5597834,
candidate,PASS,FEASIBLE_FALLBACK,1.0945607061728408,18824.583381460106,191.66702883602272,80,1.4248890999999997,resource_aware_single_box
alns,PASS,FEASIBLE_FALLBACK,1.0945607061728408,18824.583381460106,191.66702883602272,80,1.4021706999999992,resource_aware_single_box
hybrid,PASS,FEASIBLE_FALLBACK,1.0945607061728408,18824.583381460106,191.66702883602272,80,2.3120610999999993,resource_aware_single_box


说明：比较仅在通过独立约束校验的方案之间进行；fallback 非空表示该方法本轮未获得自己的可行 incumbent。

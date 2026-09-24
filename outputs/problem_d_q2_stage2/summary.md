# D题第二问四方案比较摘要

按统一词典序目标与同目标运行时间规则，当前推荐：candidate。目标并列时优先选择无回退、运行时间更短且可解释性更强的方案。

method,validation,solver_status,normalized_weighted_delivery_time,makespan_s,energy_kwh,trip_count,runtime_s,fallback
integrated_milp,PASS,FEASIBLE,1.0945607061728408,18824.583381460106,191.66702883602272,80,11.185225199999998,
candidate,PASS,FEASIBLE,1.0945607061728408,18824.583381460106,191.66702883602272,80,8.207410000000003,
alns,PASS,FEASIBLE,1.0945607061728408,18824.583381460106,191.66702883602272,80,10.909389000000001,
hybrid,PASS,FEASIBLE,1.0945607061728408,18824.583381460106,191.66702883602272,80,19.6402974,


说明：比较仅在通过独立约束校验的方案之间进行；fallback 非空表示该方法本轮未获得自己的可行 incumbent。

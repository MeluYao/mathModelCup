# D题第二问四方案比较摘要

按统一词典序目标与同目标运行时间规则，当前推荐：published_priority。目标并列时优先选择无回退、运行时间更短且可解释性更强的方案。

method,validation,solver_status,normalized_weighted_delivery_time,makespan_s,energy_kwh,trip_count,runtime_s,incumbent_source,seed_retained,native_improved_seed,fallback
published_priority,PASS,FEASIBLE,0.516328617103963,9408.443943661947,64.83172808040707,22,2.0390929,grouped_local_search,False,False,
trip_count_priority,PASS,FEASIBLE,0.7592250494690277,12439.780983079312,63.335903870203,15,0.9429879000000003,grouped_local_search,False,False,


说明：共同种子在各算法内部参与搜索；seed_retained=true 表示原生搜索未严格改进种子，不再进行报告层结果覆盖。

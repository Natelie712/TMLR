# Independent negative-steering lambda search

| Model | Status |
| --- | --- |
| qwen3 | complete |
| llama3b | complete |
| llama2 | complete |

| Model | Behavior | Negative-selected lambda | Training Linear | Training selected | Validation Linear | Validation selected | Test Linear | Test selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | AI Coordination | 0.001 | -4.368 | -4.037 | -2.030 | -1.991 | +0.163 | +0.246 |
| Qwen3-1.7B | Corrigibility | 0.0 | -34.294 | -49.494 | -30.521 | -52.027 | -36.480 | -57.329 |
| Qwen3-1.7B | Hallucination | 0.0 | -8.539 | -23.482 | -6.280 | -20.672 | -6.942 | -26.152 |
| Qwen3-1.7B | Myopic Reward | 0.005 | -20.836 | -29.140 | -18.088 | -28.499 | -19.113 | -26.945 |
| Qwen3-1.7B | Survival Instinct | 0.05 | -5.232 | -0.092 | +0.490 | +0.169 | -2.350 | +1.195 |
| Qwen3-1.7B | Sycophancy | 0.005 | -1.590 | -5.782 | -5.202 | -8.975 | -1.751 | -0.651 |
| Qwen3-1.7B | Refusal | 0.005 | -5.360 | -15.341 | -3.829 | -15.382 | -5.041 | -12.963 |
| Llama-3.2-3B-Instruct | AI Coordination | 1.0 | -0.760 | -0.755 | -0.812 | -0.810 | -0.782 | -0.779 |
| Llama-3.2-3B-Instruct | Corrigibility | 1.0 | -1.063 | -1.122 | -2.540 | -2.570 | -1.269 | -1.317 |
| Llama-3.2-3B-Instruct | Hallucination | 1.0 | +2.197 | +2.157 | +1.720 | +1.693 | +2.155 | +2.119 |
| Llama-3.2-3B-Instruct | Myopic Reward | 0.01 | +2.203 | +2.192 | +2.103 | +2.096 | +2.875 | +2.839 |
| Llama-3.2-3B-Instruct | Survival Instinct | 0.0 | +0.091 | +0.397 | +0.012 | +0.267 | +0.488 | +0.989 |
| Llama-3.2-3B-Instruct | Sycophancy | 1.0 | +0.038 | +0.373 | -0.073 | +0.480 | +0.024 | -0.183 |
| Llama-3.2-3B-Instruct | Refusal | 1.0 | -1.217 | -1.126 | -1.285 | -1.193 | -1.923 | -1.756 |
| Llama-2-7B-Chat | AI Coordination | 0.0 | -16.395 | -16.648 | -13.263 | -13.727 | -14.903 | -15.285 |
| Llama-2-7B-Chat | Corrigibility | 0.0 | -36.859 | -36.815 | -37.826 | -37.483 | -41.979 | -42.206 |
| Llama-2-7B-Chat | Hallucination | 0.0 | -17.930 | -19.054 | -16.182 | -17.189 | -21.499 | -23.604 |
| Llama-2-7B-Chat | Myopic Reward | 0.0 | -27.142 | -27.403 | -24.792 | -25.197 | -30.584 | -31.154 |
| Llama-2-7B-Chat | Survival Instinct | 0.0 | -15.660 | -15.218 | -11.149 | -10.790 | -8.732 | -6.795 |
| Llama-2-7B-Chat | Sycophancy | 1.0 | -7.302 | -10.601 | -7.008 | -11.279 | -11.487 | -9.532 |
| Llama-2-7B-Chat | Refusal | 0.0 | -17.282 | -17.700 | -13.501 | -13.799 | -20.619 | -21.201 |

Values are negative steering minus true no-hook target probability, in percentage points. More negative means stronger suppression of the official behavior target, not a negative kernel weight. Selection uses the full frozen validation grid only; pending cells are never filled from positive results. Training is in-sample. Test is an exploratory repeatedly used fixed benchmark. Pointwise 10,000 paired source-group bootstrap intervals are in summary.csv; validation selection bias and multiple comparisons limit inference. Natural displacement is not matched across kernels.

scope
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      5/1    0.000    0.000  253.030  253.030 {built-in method builtins.exec}
        1    0.000    0.000  253.030  253.030 <string>:1(<module>)
        1    0.000    0.000  253.030  253.030 /data/home/wangzz/github/scope/scope/solver.py:244(solve)
        1    0.361    0.361  253.029  253.029 {built-in method scope._scope.pywrap_Universal}
     4848    0.045    0.000  156.068    0.032 /data/home/wangzz/github/scope/scope/solver.py:724(hess)
     4848    0.059    0.000  153.760    0.032 /data/home/wangzz/github/scope/scope/solver.py:712(hess_)
    10248    0.039    0.000  136.203    0.013 /data/home/wangzz/github/scope/scope/solver.py:682(diff_fn)
     5400    0.048    0.000   96.072    0.018 /data/home/wangzz/github/scope/scope/solver.py:706(grad)
     5400    0.096    0.000   93.388    0.017 /data/home/wangzz/github/scope/scope/solver.py:692(grad_)
    10827    0.010    0.000   34.236    0.003 /data/home/wangzz/github/scope/scope/solver.py:675(loss_)
    10827    0.750    0.000   34.226    0.003 /data/home/wangzz/github/scope/pytest/create_test_model.py:16(linear_model)

总耗时253.03s，优化26次；
主动求值579次，耗时2s，平均3.16ms；求导5400次，耗时96s，平均18ms，求二阶导4848次，耗时156s，平均32ms。
比GraHTP慢的原因：
1. 60%时间花在求二阶导上了
2. 求导单次消耗大，可能按需求导的策略导致反向优化了，这一策略给每次求导带来了10ms的负担。
  Timer unit: 1e-06 s

  Line #      Hits         Time  Per Hit   % Time  Line Contents
  ==============================================================
    681                                                   @profile
    682                                                   def diff_fn(compute_params, aux_params, full_params, compute_index, data):
    683     10248   86702042.4   8460.4     74.0              full_params = full_params.at[compute_index].set(compute_params)
    684     10248   30434222.6   2969.8     26.0              return loss_(full_params, aux_params, data)
3. 求值单次消耗大，非常奇怪，可能是因为不同目的求值对应的消耗不同

scope_jit
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      5/1    0.000    0.000    2.099    2.099 {built-in method builtins.exec}
        1    0.000    0.000    2.099    2.099 <string>:1(<module>)
        1    0.000    0.000    2.099    2.099 /data/home/wangzz/github/scope/scope/solver.py:244(solve)
        1    0.129    0.129    2.098    2.098 {built-in method scope._scope.pywrap_Universal}
     5637    0.454    0.000    1.134    0.000 /data/home/wangzz/github/scope/scope/solver.py:706(grad)
     4848    0.380    0.000    0.701    0.000 /data/home/wangzz/github/scope/scope/solver.py:724(hess)
      816    0.044    0.000    0.135    0.000 /data/home/wangzz/github/scope/scope/solver.py:689(loss)
        2    0.000    0.000    0.086    0.043 /data/home/wangzz/github/scope/scope/solver.py:692(grad_)
        1    0.000    0.000    0.061    0.061 /data/home/wangzz/github/scope/scope/solver.py:712(hess_)
        4    0.000    0.000    0.039    0.010 /data/home/wangzz/github/scope/scope/solver.py:675(loss_)


总耗时2.1s，优化26次;
主动求值816次，耗时135ms，平均0.165ms；求导5637次，耗时1134ms，平均0.2ms，求二阶导4848次，耗时701ms，平均0.145ms；
其中编译时间分别占10ms、86ms、61ms。
加速效果十分显著。

GraHTP
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      5/1    0.000    0.000   19.347   19.347 {built-in method builtins.exec}
        1    0.000    0.000   19.347   19.347 <string>:1(<module>)
        1    0.000    0.000   19.347   19.347 /data/home/wangzz/github/scope/scope/base_solver.py:66(solve)
        1    0.009    0.009   19.346   19.346 /data/home/wangzz/github/scope/scope/solver.py:808(_solve)
      100    0.003    0.000   18.486    0.185 /data/home/wangzz/github/scope/scope/base_solver.py:421(_cache_nlopt)
      100    0.233    0.002   18.478    0.185 {built-in method nlopt._nlopt.opt_optimize}
     2594    0.023    0.000   18.245    0.007 /data/home/wangzz/github/scope/scope/base_solver.py:428(cache_opt_fn)
     2694    0.018    0.000   14.840    0.006 /data/home/wangzz/github/scope/scope/base_solver.py:202(loss_grad)
     5289    4.087    0.001   11.786    0.002 /data/home/wangzz/github/scope/pytest/create_test_model.py:16(linear_model)
     2595    0.010    0.000    4.207    0.002 /data/home/wangzz/github/scope/scope/base_solver.py:200(loss_fn)
  
总耗时19.35s，其中优化100次，用时18.48s，每次优化平均求导26次；
主动求值2595次，耗时4.2s，平均1.6ms，求导2694次，耗时14.8s，平均5.5ms。

GraHTP_jit
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      5/1    0.000    0.000    0.964    0.964 {built-in method builtins.exec}
        1    0.001    0.001    0.964    0.964 <string>:1(<module>)
        1    0.000    0.000    0.964    0.964 /data/home/wangzz/github/scope/scope/base_solver.py:66(solve)
        1    0.005    0.005    0.963    0.963 /data/home/wangzz/github/scope/scope/solver.py:808(_solve)
      100    0.001    0.000    0.675    0.007 /data/home/wangzz/github/scope/scope/base_solver.py:421(_cache_nlopt)
      100    0.081    0.001    0.671    0.007 {built-in method nlopt._nlopt.opt_optimize}
     2801    0.017    0.000    0.589    0.000 /data/home/wangzz/github/scope/scope/base_solver.py:428(cache_opt_fn)
     2901    0.222    0.000    0.548    0.000 /data/home/wangzz/github/scope/scope/base_solver.py:237(loss_grad)
     2802    0.148    0.000    0.293    0.000 /data/home/wangzz/github/scope/scope/base_solver.py:226(loss_fn)
        1    0.000    0.000    0.029    0.029 /data/home/wangzz/github/scope/scope/base_solver.py:230(grad_)
        2    0.000    0.000    0.023    0.012 /data/home/wangzz/github/scope/scope/base_solver.py:223(loss_)
        2    0.000    0.000    0.023    0.012 /data/home/wangzz/github/scope/pytest/create_test_model.py:16(linear_model)

总耗时964ms，其中优化100次，用时675ms，每次优化平均求导28次；
主动求值2802次，耗时293ms，平均0.10ms，求导2901次，耗时548ms，平均0.19ms；
其中编译时间分别占12ms、29ms。


README
---

**训练**  
设置好total_episodes（训练轮数）、noise_level（噪声水平）、模型和tensorboard保存路径以及其他参数  
本文设置：total_episodes = 50  
启动：`python cleanrl_train.py `  

**测试**  
设置好total_episodes（训练轮数）、eval_episodes（测试轮数）、noise_levels（要测试的噪声）以及模型路径  
本文设置：total_episodes = 50、eval_episodes = 10、noise_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  
启动：`python cleanrl_test.py `  

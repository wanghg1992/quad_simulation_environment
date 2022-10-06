from gym.envs.registration import register

register(
	id='Quadruped-v0', # 环境名,版本号v0必须有
	entry_point='gym_quad_env.envs:QuadrupedEnv' # 文件夹名.文件名:类名
	# 根据需要定义其他参数
)


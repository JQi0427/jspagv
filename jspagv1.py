import pandas as pd
import numpy as np


class Encode:
    def __init__(self, pt, ms, agv, J_num, M_num, A_num, population_size, num, agv_num):
        self.pt = pt  # processing time
        self.ms = ms  # machine sequence
        self.agv = agv  # agv transform time
        self.J_num = J_num  # Job num
        self.M_num = M_num  # Machine num
        self.A_num = A_num  # AGV num
        self.population_size = population_size
        self.num = num  # Iteration
        self.agv_num = agv_num  # AGV iteration
    def initJobSequence(self):
        population_joblist = []
        for i in range(self.population_size):
            nxm_random_num = list(
                np.random.permutation(self.num))  # generate a random permutation of 0 to num_job*num_mc-1
            population_joblist.append(nxm_random_num)  # add to the job_sequence
            for j in range(self.num):
                population_joblist[i][j] = population_joblist[i][
                                            j] % self.J_num  # convert to job number format, every job appears m times
        return population_joblist

    def initAGVSequence(self):
        population_AGVlist = []
        for i in range(self.population_size):
            nxm_random_num = list(
                np.random.permutation(self.agv_num))  # generate a random permutation of 0 to num_job*num_mc-1
            population_AGVlist.append(nxm_random_num)  # add to the machine_sequence
            for j in range(self.agv_num):
                population_AGVlist[i][j] = population_AGVlist[i][
                                            j] % self.A_num  # convert to job number format, every job appears m times
        return population_AGVlist


pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0])
at_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="AGV Time", index_col=[0])

dfshape = pt_tmp.shape
J_num = dfshape[0]
M_num = dfshape[1]

A_num = 3
population_size = 1
num = J_num * M_num
agv_num = (M_num+2)*(M_num+2)
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(J_num)]  # process time
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(J_num)]  # machine sequence
agv = [list(map(int, at_tmp.iloc[i])) for i in range(M_num+2)]  # AGV sequence

JSPAGV = Encode(pt, ms, agv, J_num, M_num, A_num, population_size, num, agv_num)

init_jobs = JSPAGV.initJobSequence()
print(init_jobs)
print(JSPAGV.initAGVSequence())

# 计算最大机器完工时间
init_time = [0] * M_num
init_sequence = [0] * M_num
for i in range(num):
    temp_job = init_jobs[0][i]
    temp_machine = ms[temp_job][init_sequence[temp_job]] - 1
    init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]]
    init_sequence[temp_job] += 1

print("机器的运行时间：", init_time)
print(f"机器的最大运行时间来自机器{init_time.index(max(init_time)) + 1},时间为：{max(init_time)}")

# 计算含AGV的最大完工时间
init_agv = JSPAGV.initAGVSequence()
init_time = [0] * M_num
init_sequence = [0] * M_num
for i in range(num):
    temp_job = init_jobs[0][i]  # 获取工序
    temp_machine = ms[temp_job][init_sequence[temp_job]] - 1  # 获取对应的机器
    temp_agv = init_agv[0][i]  # 获取对应的agv





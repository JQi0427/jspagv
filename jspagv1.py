import pandas as pd
import numpy as np


class Encode:
    def __init__(self, pt, ms, agv, J_num, M_num, A_num, population_size, num, agv_num):
        self.pt = pt  # processing time
        self.ms = ms  # machine sequence
        self.agv = agv  # AGV transform time
        self.J_num = J_num  # Job num
        self.M_num = M_num  # Machine num
        self.A_num = A_num  # AGV num
        self.population_size = population_size
        self.num = num  # Iteration
        self.agv_num = agv_num  # AGV iteration¥
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


pt_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="Machines Sequence", index_col=[0])
at_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="AGV Time", index_col=[0])

dfshape = pt_tmp.shape
J_num = dfshape[0]
M_num = dfshape[1]


A_num = 3
population_size = 1
num = J_num * M_num
agv_num = M_num*(M_num+1)
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(J_num)]  # process time
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(J_num)]  # machine sequence
agv = [list(map(int, at_tmp.iloc[i])) for i in range(M_num+2)]  # AGV sequence



JSPAGV = Encode(pt, ms, agv, J_num, M_num, A_num, population_size, num, agv_num)

init_jobs = JSPAGV.initJobSequence()
print(init_jobs)
init_agv = JSPAGV.initAGVSequence()
print(init_agv)

# 计算最大机器完工时间
init_time = [0] * M_num
init_sequence = [0] * M_num
for i in range(num):
    temp_job = init_jobs[0][i]
    temp_machine = ms[temp_job][init_sequence[temp_job]]
    init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]]
    init_sequence[temp_job] += 1

print("机器的运行时间：", init_time)
print(f"机器的最大运行时间来自机器{init_time.index(max(init_time))},时间为：{max(init_time)}")

# 计算含AGV的最大完工时间
#init_agv = JSPAGV.initAGVSequence()
#print(init_agv)
init_time = [0] * M_num
init_sequence = [0] * M_num
boolean_agv = [0]*3  # 布尔型变量，如果agv空闲为0，agv不空闲则为1
location_agv = [0]*3  # agv位置初始化都在仓库
t = 1
for i in range(num):
    temp_job = init_jobs[0][i]  # achieve job operation
    temp_machine = ms[temp_job][init_sequence[temp_job]]   # achieve related machine Mn+1
    temp_agv = init_agv[0][i]  # achieve related agv sequence
    if init_sequence[temp_job] != 0 & init_sequence[temp_job] != 5:
        # Gets the machine where the workpiece was located in the previous operation
        last_machine = ms[temp_job][init_sequence[temp_job]-1] + 1
        if boolean_agv[temp_agv] == 0:  # 判断agv是否空闲,0:空闲
            init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][temp_machine + 1] + \
                                  agv[location_agv[temp_agv]][last_machine]
            location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
            boolean_agv[temp_agv] = 1
        else:
            machine = location_agv[temp_agv] - 1  # 得到agv完成上个任务所在的机器，用那个机器的上的时间减去ps，得出agv到达机器的时间
            temp_time = init_time[machine]
            # 用agv到达机器的时间和当前时间对比，算出等待agv的时间

    if init_sequence[temp_job] == 0:  # 目前操作是工件的第一个工序时
        if boolean_agv[temp_agv] == 0:  # 判断agv是否空闲,0:空闲
            if location_agv[temp_agv] == 0:  # agv在仓库
                init_time[temp_machine] += + pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine+1]  # 计算agv在仓库的时间
                location_agv[temp_agv] = temp_machine+1  # 记录该agv完成任务后的位置
                boolean_agv[temp_agv] = 1
            else:  # agv不在仓库
                init_time[temp_machine] += + pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine+1] + \
                                      agv[0][location_agv[temp_agv]]
                location_agv[temp_agv] = temp_machine+1  # 记录该agv完成任务后的位置
                boolean_agv[temp_agv] = 1
    init_sequence[temp_job] += 1
print(init_time)







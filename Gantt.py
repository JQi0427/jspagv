import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        self.agv_num = agv_num  # AGV iteration

    def initJobSequence(self):
        population_joblist = []
        for i in range(int(self.population_size)):
            nxm_random_num = list(
                np.random.permutation(self.num))  # generate a random permutation of 0 to num_job*num_mc-1
            population_joblist.append(nxm_random_num)  # add to the job_sequence
            for j in range(self.num):
                population_joblist[i][j] = population_joblist[i][
                                               j] % self.J_num  # convert to job number format, every job appears m times
        return population_joblist

    def initAGVSequence(self):
        population_AGVlist = []
        for i in range(int(self.population_size)):
            nxm_random_num = list(
                np.random.permutation(self.agv_num))  # generate a random permutation of 0 to num_job*num_mc-1
            population_AGVlist.append(nxm_random_num)  # add to the machine_sequence
            for j in range(self.agv_num):
                population_AGVlist[i][j] = population_AGVlist[i][
                                               j] % self.A_num  # convert to job number format, every job appears m times
        return population_AGVlist


pt_tmp = pd.read_excel("JSP_dataset_ft04.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset_ft04.xlsx", sheet_name="Machines Sequence", index_col=[0])
at_tmp = pd.read_excel("JSP_dataset_ft04.xlsx", sheet_name="AGV Time", index_col=[0])

dfshape = pt_tmp.shape
J_num = dfshape[0]
M_num = dfshape[1]

A_num = 3
population_size = 1
num = J_num * M_num
agv_num = M_num * (M_num + 1)
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(J_num)]  # process time
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(J_num)]  # machine sequence
agv = [list(map(int, at_tmp.iloc[i])) for i in range(M_num + 2)]  # AGV sequence

JSPAGV = Encode(pt, ms, agv, J_num, M_num, A_num, population_size, num, agv_num)

#init_jobs = JSPAGV.initJobSequence()
#init_agv = JSPAGV.initAGVSequence()
init_jobs = [[1, 0, 3, 0, 1, 3, 1, 2, 3, 2, 0, 1, 3, 2, 2, 0]]
init_agv = [[1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]]
#init_jobs = [[3, 5, 2, 4, 3, 5, 2, 5, 3, 1, 0, 0, 0, 1, 5, 3, 1, 0, 1, 0, 0, 2, 3, 4, 2, 5, 5, 1, 4, 4, 4, 3, 2, 4, 1, 2]]
#init_agv = [[2, 0, 2, 1, 0, 2, 2, 1, 1, 1, 1, 0, 2, 0, 1, 0, 0, 2, 2, 0, 0, 0, 1, 2, 1, 0, 2, 0, 0, 2, 1, 1, 2, 0, 1, 2, 0, 1, 1, 2, 2, 1]]
print(init_jobs)
print(init_agv)

# 计算含AGV的最大完工时间
# init_agv = JSPAGV.initAGVSequence()
# print(init_agv)
init_time = [0] * M_num
init_sequence = [0] * M_num
boolean_agv = [0] * 3  # 布尔型变量，如果agv空闲为0，agv不空闲则为1
location_agv = [0] * 3  # agv位置初始化都在仓库
process_time = [0] * 3  # 记录agv不空闲时，那台机器上的pt，如现在agv从机器运输到机器二，记录该工件在机器二上的加工时间
t = len(init_agv[0]) - 1
tasks = [0] * M_num
lasttime = [0] * 3  # 记录agv运输成品到成品库的时间
agv_time = [0] * 3  # 记录每个agv完成任务的结束时间
job_time = []
job_starttime = []
operation = []
for i in range(6):
    row = []
    for j in range(6):
        row.append(0)
    operation.append(row)
for i in range(6):
    row = []
    for j in range(6):
        row.append(0)
    job_time.append(row)
for i in range(6):
    row = []
    for j in range(6):
        row.append(0)
    job_starttime.append(row)

for i in range(num):
    temp_job = init_jobs[0][i]  # achieve job operation
    temp_machine = ms[temp_job][init_sequence[temp_job]]  # achieve related machine Mn+1
    temp_agv = init_agv[0][i]  # achieve related agv sequence
    stattime = init_time[temp_machine]
    if location_agv[temp_agv] != 0:  # 避免agv在初始化位置仓库产生的影响
        machine = location_agv[temp_agv] - 1  # agv在上一个任务结束时的位置
        if machine != 6:  # 避免agv在成品库
            temp_time = agv_time[temp_agv]  # 该agv到达上一个任务的时间，上一个机器的时间减去加工时间 = agv到达时间
        else:
            temp_time = lasttime[temp_agv]
    else:
        temp_time = 0
    if init_time[temp_machine] > temp_time:  # 对比agv达到上一个任务的结束时间和该任务的起始时间，判断agv是否空闲
        boolean_agv[temp_agv] = 0
    else:
        boolean_agv[temp_agv] = 1  # 1
    if init_sequence[temp_job] != 0:  # 不是工件的第一个工序
        # 获取上一次操作中工件所在的机器
        last_machine = ms[temp_job][init_sequence[temp_job] - 1] + 1
        if boolean_agv[temp_agv] == 0:  # 判断agv是否空闲,0:空闲
            # 判断该工件的前一个工序是否完成
            if job_time[temp_job][init_sequence[temp_job] - 1] < init_time[temp_machine]:  # 上一个工序完成了
                if init_time[temp_machine] > agv_time[temp_agv] + agv[location_agv[temp_agv]][last_machine] + \
                        agv[last_machine][temp_machine + 1]:  # agv[][]:agv对应的运输时间
                    if agv_time[temp_agv] + agv[location_agv[temp_agv]][last_machine] < job_time[temp_job][
                        init_sequence[temp_job] - 1]:
                        agv_time[temp_agv] = job_time[temp_job][init_sequence[temp_job] - 1] + agv[last_machine][
                            temp_machine + 1]
                    else:
                        agv_time[temp_agv] = agv_time[temp_agv] + agv[location_agv[temp_agv]][last_machine] + \
                                             agv[last_machine][temp_machine + 1]  # 获取agv到达temp——machine的时间
                    init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]]
                    job_starttime[temp_job][init_sequence[temp_job]] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                else:
                    k = init_time[temp_machine] - agv_time[temp_agv]
                    init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][temp_machine + 1] +\
                                               agv[location_agv[temp_agv]][last_machine] - k
                    agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                    job_starttime[temp_job][init_sequence[temp_job]] = \
                        init_time[temp_machine] - (pt[temp_job][init_sequence[temp_job]] + agv[last_machine][temp_machine + 1] +\
                        agv[location_agv[temp_agv]][last_machine]) + k
            else:  # 上一个工序没完成
                time = init_time[last_machine - 1] - init_time[temp_machine]
                if time > agv[location_agv[temp_agv]][last_machine - 1]:
                    init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][temp_machine + 1] + time
                    agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                    job_starttime[temp_job][init_sequence[temp_job]] = \
                        init_time[temp_machine] - (pt[temp_job][init_sequence[temp_job]] + agv[last_machine][temp_machine + 1])
                else:
                    #job_starttime[temp_job][init_sequence[temp_job]] = init_time[last_machine - 1] - agv_time[temp_agv]
                    init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][
                        temp_machine + 1] + agv[location_agv[temp_agv]][last_machine]
                    job_starttime[temp_job][init_sequence[temp_job]] = \
                        init_time[temp_machine] -(pt[temp_job][init_sequence[temp_job]] + agv[last_machine][temp_machine + 1]\
                        + agv[location_agv[temp_agv]][last_machine])
                    agv_time[temp_agv]  = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                    agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
            location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
            process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
        else:  # agv不空闲
            difference = temp_time - init_time[temp_machine]
            # 判断该工件的前一个工序是否完成
            if job_time[temp_job][init_sequence[temp_job] - 1] < init_time[temp_machine]:  # 上一个工序完成了
                init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][temp_machine + 1] + \
                                           agv[location_agv[temp_agv]][last_machine] + difference
                agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                job_starttime[temp_job][init_sequence[temp_job]] = \
                    init_time[temp_machine] - (pt[temp_job][init_sequence[temp_job]] + agv[last_machine][temp_machine + 1] + \
                                           agv[location_agv[temp_agv]][last_machine])
            else:
                if agv_time[temp_agv] + agv[location_agv[temp_agv]][last_machine] < job_time[temp_job][
                    init_sequence[temp_job] - 1]:
                    init_time[temp_machine] = job_time[temp_job][init_sequence[temp_job] - 1] + agv[last_machine][
                        temp_machine + 1] + pt[temp_job][init_sequence[temp_job]]
                    job_starttime[temp_job][init_sequence[temp_job]] = \
                        init_time[temp_machine] - (agv[last_machine][temp_machine + 1] + pt[temp_job][init_sequence[temp_job]])
                else:
                    init_time[temp_machine] = agv_time[temp_agv] + agv[location_agv[temp_agv]][last_machine] + \
                                              agv[last_machine][temp_machine + 1] + pt[temp_job][init_sequence[temp_job]]
                    if job_time[temp_job][init_sequence[temp_job] - 1] > agv_time[temp_agv]:
                        job_starttime[temp_job][init_sequence[temp_job]] = \
                            init_time[temp_machine] - (agv[location_agv[temp_agv]][last_machine] + agv[last_machine][temp_machine + 1] +\
                                                   pt[temp_job][init_sequence[temp_job]]) + job_time[temp_job][init_sequence[temp_job] - 1]\
                                                    - agv_time[temp_agv]
                    else:
                        job_starttime[temp_job][init_sequence[temp_job]] = \
                            init_time[temp_machine] - (agv[location_agv[temp_agv]][last_machine] + agv[last_machine][temp_machine + 1] \
                                                       + pt[temp_job][init_sequence[temp_job]])
                agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
            location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
            process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
    else:  # 目前操作是工件的第一个工序时
        if boolean_agv[temp_agv] == 0:  # 判断agv是否空闲,0:空闲
            if location_agv[temp_agv] == 0:  # agv在仓库
                init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine + 1]  # 计算agv在仓库的时间
                agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                job_starttime[temp_job][init_sequence[temp_job]] = \
                    init_time[temp_machine] - (pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine + 1])
                location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
                process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
            else:  # agv不在仓库
                if init_time[temp_machine] > agv[0][temp_machine + 1] + agv[0][location_agv[temp_agv]] + agv_time[
                    temp_agv]:
                    init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]]
                    agv_time[temp_agv] = agv[0][temp_machine + 1] + agv[0][location_agv[temp_agv]] + agv_time[temp_agv]
                    job_starttime[temp_job][init_sequence[temp_job]] = init_time[temp_machine] - pt[temp_job][
                        init_sequence[temp_job]]
                else:
                    job_starttime[temp_job][init_sequence[temp_job]] = init_time[temp_machine]
                    init_time[temp_machine] = pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine + 1] + \
                                                 agv[0][location_agv[temp_agv]] + agv_time[temp_agv]
                    agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
                process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
        else:
            if location_agv[temp_agv] != 0:  # 避免agv在初始化位置仓库产生的影响
                machine = location_agv[temp_agv] - 1  # agv在上一个任务结束时的位置
                if machine != 6:  # 避免agv在成品库
                    temp_time = agv_time[temp_agv]  # 该agv到达上一个任务的时间，上一个机器的时间减去加工时间 = agv到达时间
                else:
                    temp_time = lasttime[temp_agv]
            else:
                temp_time = 0
            if init_time[temp_machine] > temp_time + agv[0][location_agv[temp_agv]]:
                init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine + 1]
                agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                job_starttime[temp_job][init_sequence[temp_job]] = \
                    init_time[temp_machine] - (pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine + 1])
            else:
                difference = temp_time + agv[0][location_agv[temp_agv]] - init_time[temp_machine]

                init_time[temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine + 1] + difference
                agv_time[temp_agv] = init_time[temp_machine] - pt[temp_job][init_sequence[temp_job]]
                job_starttime[temp_job][init_sequence[temp_job]] = \
                    init_time[temp_machine] - (pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine + 1] + \
                                               agv[0][location_agv[temp_agv]])
            location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
            process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
    job_time[temp_job][init_sequence[temp_job]] = init_time[temp_machine]
    operation[temp_job][init_sequence[temp_job]] = init_time[temp_machine]
    if init_sequence[temp_job] == J_num - 1:  # 计算从机器上运输到产品库的时间
        a = init_agv[0][t]  # 获取分配的agv
        if location_agv[a] - 1 != M_num:
            temp_time = init_time[location_agv[a] - 1] - process_time[a]  # 该agv到达上一个任务的时间，上一个机器的时间减去加工时间 = agv到达时间
        else:
            temp_time = lasttime[a]
        if init_time[temp_machine] > temp_time:  # 对比agv达到上一个任务的结束时间和该任务的起始时间，判断agv是否空闲
            boolean_agv[a] = 0
        else:
            boolean_agv[a] = 1
        if boolean_agv[a] == 0:  # 判断agv是否空闲,0:空闲
            if init_time[temp_machine] - temp_time > agv[location_agv[a]][temp_machine + 1]:
                operation[temp_job][init_sequence[temp_job]] = init_time[temp_machine] + agv[temp_machine + 1][M_num + 1]
            else:
                operation[temp_job][init_sequence[temp_job]] = init_time[temp_machine] + agv[location_agv[a]][
                    temp_machine + 1] - (init_time[temp_machine] - temp_time) + agv[temp_machine + 1][M_num + 1]
        else:
            operation[temp_job][init_sequence[temp_job]] = init_time[temp_machine] + temp_time - init_time[
                temp_machine] + agv[location_agv[a]][temp_machine + 1] + agv[temp_machine + 1][M_num + 1]
        location_agv[a] = M_num +1
        t = t - 1
        agv_time[a] = operation[temp_job][init_sequence[temp_job]]
        lasttime[a] = operation[temp_job][init_sequence[temp_job]]
    init_sequence[temp_job] += 1
    endtime = init_time[temp_machine]
    # print("endtime",endtime)
    tasks[temp_machine] = (stattime, endtime)
    # print(tasks)
print("init_time",init_time)
print("job_starttime",job_starttime)
print("job_time",job_time)


# 绘制甘特图
fig, ax = plt.subplots(2, 1)

# 设置y轴刻度
y_ticks = [f'M{i + 1}' for i in range(J_num)]
ax[0].set_yticks(range(J_num))
ax[0].set_yticklabels(y_ticks)

# 绘制任务条形图
init_sequence = [0] * M_num
color = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink']
for i in range(num):
    job = init_jobs[0][i]
    machine = ms[job][init_sequence[job]]
    end_time = job_time[job][init_sequence[job]]
    start_time = job_starttime[job][init_sequence[job]]
    ax[0].barh(machine, width=end_time - start_time,
               left=start_time, height=0.5, align='center', color=color[job])
    init_sequence[job] += 1

# 设置x轴和图表标题
ax[0].set_title('Gantt Chart')

# 绘制AGV甘特图
y_ticks = [f'AGV{i}' for i in range(A_num)]
ax[1].set_yticks(range(A_num))
ax[1].set_yticklabels(y_ticks)
init_sequence = [0] * M_num
color = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink']
for i in range(num):
    job = init_jobs[0][i]
    agv = init_agv[0][i]
    end_time = job_time[job][init_sequence[job]] - pt[job][init_sequence[job]]
    start_time = job_starttime[job][init_sequence[job]]
    ax[1].barh(agv, width=end_time - start_time,
               left=start_time, height=0.5, align='center', color=color[job])
    init_sequence[job] += 1
for i in range(1,7):
    agv = init_agv[0][-i]
    end_time = operation[i-1][5]
    start_time = job_time[i-1][5]
    ax[1].barh(agv, width=end_time - start_time,
               left=start_time, height=0.5, align='center', color='black')
ax[1].set_xlabel('Time')

# 显示图表
plt.tight_layout()
plt.show()



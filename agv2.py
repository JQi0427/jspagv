import pandas as pd
import numpy as np
import time
import copy

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

    def initAGVSequence(self):
        population_AGVlist = []
        for i in range(int(self.population_size/2)):
            nxm_random_num = list(
                np.random.permutation(self.agv_num))  # generate a random permutation of 0 to num_job*num_mc-1
            population_AGVlist.append(nxm_random_num)  # add to the machine_sequence
            for j in range(self.agv_num):
                population_AGVlist[i][j] = population_AGVlist[i][
                                            j] % self.A_num  # convert to job number format, every job appears m times
        return population_AGVlist

    def initJobSequence(self):
        population_joblist = []
        for i in range(int(self.population_size)):
            nxm_random_num = list(np.random.permutation(self.num))
            population_joblist.append(nxm_random_num)
            for j in range(self.num):
                population_joblist[i][j] = population_joblist[i][j] % self.J_num

        print("交叉前的种群列表：", population_joblist)

        parent_list = copy.deepcopy(population_joblist)
        offspring_list = copy.deepcopy(population_joblist)
        S = list(np.random.permutation(self.population_size))

        for m in range(int(self.population_size / 2)):
            crossover_prob = np.random.rand()
            if crossover_rate >= crossover_prob:
                parent_1 = population_joblist[S[2 * m]][:]
                parent_2 = population_joblist[S[2 * m + 1]][:]
                child_1 = parent_1[:]
                child_2 = parent_2[:]
                cutpoint = list(np.random.choice(self.num, 2, replace=False))
                cutpoint.sort()

                child_1[cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
                child_2[cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]
                offspring_list[S[2 * m]] = child_1[:]
                offspring_list[S[2 * m + 1]] = child_2[:]

            for m in range(self.population_size):
                job_count = {}
                larger, less = [], []
                for i in range(self.J_num):
                    if i in offspring_list[m]:
                        count = offspring_list[m].count(i)
                        pos = offspring_list[m].index(i)
                        job_count[i] = [count, pos]
                    else:
                        count = 0
                        job_count[i] = [count, 0]
                    if count > self.M_num:
                        larger.append(i)
                    elif count < self.M_num:
                        less.append(i)

                for k in range(len(larger)):
                    chg_job = larger[k]
                    while job_count[chg_job][0] > self.M_num:
                        for d in range(len(less)):
                            if job_count[less[d]][0] < self.M_num:
                                index = [i for i in range(len(offspring_list[m])) if
                                         offspring_list[m][i] == chg_job]
                                offspring_list[m][index[0]] = less[d]
                                job_count[chg_job][1] = index[0]
                                job_count[chg_job][0] = job_count[chg_job][0] - 1
                                job_count[less[d]][0] = job_count[less[d]][0] + 1
                            if job_count[chg_job][0] == self.M_num:
                                break

        return offspring_list



pt_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="Machines Sequence", index_col=[0])
at_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="AGV Time", index_col=[0])

dfshape = pt_tmp.shape
J_num = dfshape[0]
M_num = dfshape[1]


A_num = 3
num = J_num * M_num
agv_num = M_num*(M_num+1)
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(J_num)]  # process time
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(J_num)]  # machine sequence
agv = [list(map(int, at_tmp.iloc[i])) for i in range(M_num+2)]  # AGV sequence

population_size =  2 # default value is 30
crossover_rate = 0.8 # default value is 0.8


start_time = time.time()

JSPAGV = Encode(pt, ms, agv, J_num, M_num, A_num, population_size, num, agv_num)

init_jobs = JSPAGV.initJobSequence()

print("交叉后的种群列表：", init_jobs)
init_agv = JSPAGV.initAGVSequence()
print("agv序列：",init_agv)

init_time = [[0] * M_num]*population_size
#计算时间:
for j in range(population_size):
    init_sequence = [0] * M_num
    boolean_agv = [0] * 3  # 布尔型变量，如果agv空闲为0，agv不空闲则为1
    location_agv = [0] * 3  # agv位置初始化都在仓库
    process_time = [0] * 3  # 记录agv不空闲时，那台机器上的pt，如现在agv从机器晕运输到机器二，记录该工件在机器二上的加工时间
    t = len(init_agv[0]) - 1
    lasttime = [0] * 3  # 记录agv运输成品到成品库的时间
    for i in range(num):
        temp_job = init_jobs[j][i]  # achieve job operation
        temp_machine = ms[temp_job][init_sequence[temp_job]]  # achieve related machine Mn+1
        temp_agv = init_agv[0][i]  # achieve related agv sequence
        if init_sequence[temp_job] != 0:  # 不是工件的第一个工序
            # Gets the machine where the workpiece was located in the previous operation
            last_machine = ms[temp_job][init_sequence[temp_job] - 1] + 1
            if location_agv[temp_agv] != 0:  # 避免agv在初始化位置仓库产生的影响
                machine = location_agv[temp_agv] - 1  # agv在上一个任务结束时的位置
                if machine != 6:
                    temp_time = init_time[j][machine] - process_time[temp_agv]  # 该agv到达上一个任务的时间，上一个机器的时间减去加工时间 = agv到达时间
                else:
                    temp_time = lasttime[temp_agv]
            else:
                temp_time = 0
            if init_time[j][temp_machine] > temp_time:  # 对比agv达到上一个任务的结束时间和该任务的起始时间，判断agv是否空闲
                boolean_agv[temp_agv] = 0
            else:
                boolean_agv[temp_agv] = 1
            if boolean_agv[temp_agv] == 0:  # 判断agv是否空闲,0:空闲
                # 判断该工件的前一个工序是否完成
                if init_time[j][last_machine - 1] < init_time[j][temp_machine]:  # 上一个工序完成了
                    time = init_time[j][temp_machine] - init_time[j][last_machine - 1]
                    if time > agv[location_agv[temp_agv]][last_machine - 1] + agv[last_machine][temp_machine + 1]:
                        init_time[j][temp_machine] += pt[temp_job][init_sequence[temp_job]]
                    else:
                        init_time[j][temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][
                            temp_machine + 1] + agv[location_agv[temp_agv]][last_machine] - time
                else:
                    time = init_time[j][last_machine - 1] - init_time[j][temp_machine]
                    if time > agv[location_agv[temp_agv]][last_machine - 1]:  # 上一个工序没完成
                        init_time[j][temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][
                            temp_machine + 1] + time
                    else:
                        init_time[j][temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][
                            temp_machine + 1] + agv[location_agv[temp_agv]][last_machine]
                location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
                process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
            else:  # agv不空闲
                # 用agv到达机器的时间和当前时间对比，算出等待agv的时间
                difference = temp_time - init_time[j][temp_machine]
                # 判断该工件的前一个工序是否完成
                if init_time[j][last_machine - 1] < init_time[j][temp_machine]:
                    time = init_time[j][temp_machine] - init_time[j][last_machine - 1]
                    if time > agv[location_agv[temp_agv]][last_machine - 1] + agv[last_machine][
                        temp_machine + 1] + difference:
                        init_time[j][temp_machine] += pt[temp_job][init_sequence[temp_job]]
                    else:
                        init_time[j][temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][
                            temp_machine + 1] + agv[location_agv[temp_agv]][last_machine] + difference - time
                else:
                    time = init_time[j][last_machine - 1] - init_time[j][temp_machine]
                    if time > agv[location_agv[temp_agv]][last_machine - 1] + difference:
                        init_time[j][temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][
                            temp_machine + 1] + time
                    else:
                        init_time[j][temp_machine] += pt[temp_job][init_sequence[temp_job]] + agv[last_machine][
                            temp_machine + 1] + agv[location_agv[temp_agv]][last_machine] + difference
                location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
                process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
        else:  # 目前操作是工件的第一个工序时
            if boolean_agv[temp_agv] == 0:  # 判断agv是否空闲,0:空闲
                if location_agv[temp_agv] == 0:  # agv在仓库
                    init_time[j][temp_machine] += + pt[temp_job][init_sequence[temp_job]] + agv[0][
                        temp_machine + 1]  # 计算agv在仓库的时间
                    location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
                    process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
                else:  # agv不在仓库
                    init_time[j][temp_machine] += + pt[temp_job][init_sequence[temp_job]] + agv[0][temp_machine + 1] + \
                                               agv[0][location_agv[temp_agv]]
                    location_agv[temp_agv] = temp_machine + 1  # 记录该agv完成任务后的位置
                    process_time[temp_agv] = pt[temp_job][init_sequence[temp_job]]
        if init_sequence[temp_job] == 5:  # 计算从机器上运输到产品库的时间
            a = init_agv[0][t]  # 获取分配的agv
            if location_agv[a] - 1 != 6:
                temp_time = init_time[j][location_agv[a] - 1] - process_time[a]  # 该agv到达上一个任务的时间，上一个机器的时间减去加工时间 = agv到达时间
            else:
                temp_time = lasttime[a]
            if init_time[j][temp_machine] > temp_time:  # 对比agv达到上一个任务的结束时间和该任务的起始时间，判断agv是否空闲
                boolean_agv[a] = 0
            else:
                boolean_agv[a] = 1
            if boolean_agv[a] == 0:  # 判断agv是否空闲,0:空闲
                if init_time[j][temp_machine] - temp_time > agv[location_agv[a]][temp_machine + 1]:
                    init_time[j][temp_machine] += agv[temp_machine + 1][7]
                else:
                    init_time[j][temp_machine] += agv[location_agv[a]][temp_machine + 1] - (
                                init_time[j][temp_machine] - temp_time) + agv[temp_machine + 1][7]
            else:
                init_time[j][temp_machine] += temp_time - init_time[j][temp_machine] + agv[location_agv[a]][
                    temp_machine + 1] + agv[temp_machine + 1][7]
            location_agv[a] = 7
            t = t - 1
            lasttime[a] = init_time[j][temp_machine]
        init_sequence[temp_job] += 1


print(init_time)



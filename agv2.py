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
        for i in range(int(self.population_size / 2)):
            nxm_random_num = list(
                np.random.permutation(self.agv_num))  # generate a random permutation of 0 to num_job*num_mc-1
            population_AGVlist.append(nxm_random_num)  # add to the machine_sequence
            for j in range(self.agv_num):
                population_AGVlist[i][j] = population_AGVlist[i][
                                            j] % self.A_num  # convert to job number format, every job appears m times
        return population_AGVlist

    def initJobSequence(self):
        population_joblist = []
        for i in range(self.population_size):
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

offspring_jobs = JSPAGV.initJobSequence()

print("交叉后的种群列表：", offspring_jobs)
init_agv = JSPAGV.initAGVSequence()
print("agv序列：",init_agv)

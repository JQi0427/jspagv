import pandas as pd
import numpy as np


class Encode:
    def __init__(self, pt, ms, J_num, M_num, A_num, population_size, num):
        self.pt = pt  # processing time
        self.ms = ms  # machine sequence
        self.J_num = J_num
        self.M_num = M_num
        self.A_num = A_num
        self.population_size = population_size
        self.num = num

    def initialization(self):
        population_list = []
        for i in range(self.population_size):
            nxm_random_num = list(
                np.random.permutation(self.num))  # generate a random permutation of 0 to num_job*num_mc-1
            population_list.append(nxm_random_num)  # add to the population_list
            for j in range(self.num):
                population_list[i][j] = population_list[i][
                                            j] % num_job  # convert to job number format, every job appears m times
        return population_list


pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0])
at_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="AGV Time", index_col=[0])

dfshape = pt_tmp.shape
num_mc = dfshape[1]  # number of machines
num_job = dfshape[0]  # number of jobs
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(num_job)]  # process time
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(num_job)]  # machine sequence
J_num = 10
M_num = 10
A_num = 2
population_size = 1
num = J_num * M_num

JSPAGV = Encode(pt, ms, J_num, M_num, A_num, population_size, num)

print(JSPAGV.initialization())

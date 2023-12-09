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
        self.agv_num = agv_num  # AGV iteration

    def initJobSequence(self):
        population_joblist = []
        for i in range(self.population_size):
            nxm_random_num = list(np.random.permutation(self.num))
            population_joblist.append(nxm_random_num)
            for j in range(self.num):
                population_joblist[i][j] = population_joblist[i][j] % self.J_num
        return population_joblist

    def initAGVSequence(self, initial_sequence=None, job_sequence_length=None):
        if initial_sequence is not None and job_sequence_length is None:
            raise ValueError("Please provide the length of the job sequence when initializing AGV sequence.")

        population_AGVlist = []
        for i in range(self.population_size):
            if initial_sequence is not None:
                nxm_random_num = initial_sequence[:]
                np.random.shuffle(nxm_random_num)
            else:
                nxm_random_num = list(np.random.permutation(self.agv_num))

            population_AGVlist.append(nxm_random_num)
            for j in range(len(nxm_random_num)):
                population_AGVlist[i][j] = population_AGVlist[i][j] % self.A_num

        return population_AGVlist


# Load data from Excel files
pt_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="Machines Sequence", index_col=[0])
at_tmp = pd.read_excel("JSP_dataset_ft06.xlsx", sheet_name="AGV Time", index_col=[0])

# Calculate necessary values
dfshape = pt_tmp.shape
J_num = dfshape[0]
M_num = dfshape[1]

A_num = 3
population_size = 1
num = J_num * M_num
agv_num = M_num * (M_num + 1)

# Create lists from loaded data
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(J_num)]  # process time
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(J_num)]  # machine sequence
agv = [list(map(int, at_tmp.iloc[i])) for i in range(M_num + 2)]  # AGV sequence

# Define the Encode class and create an instance
JSPAGV = Encode(pt, ms, agv, J_num, M_num, A_num, population_size, num, agv_num)

# 1. Fix the first set, generate a random permutation for the second set
init_jobs_fixed = JSPAGV.initJobSequence()
init_agv_random = JSPAGV.initAGVSequence()

print("1. Fixed first set, random second set:")
print(init_jobs_fixed)
print(init_agv_random[0])

# 2. Fix the second set, generate a random permutation for the first set
init_agv_fixed = JSPAGV.initAGVSequence()
init_jobs_random = JSPAGV.initJobSequence()

print("\n2. Fixed second set, random first set:")
print(init_jobs_random)
print(init_agv_fixed[0])


# 3. Ensure a one-to-one correspondence between the first and second sets
job_seq_length = len(init_jobs_fixed[0])
init_jobs_corresponding = JSPAGV.initJobSequence()
init_agv_corresponding = JSPAGV.initAGVSequence(init_jobs_corresponding[0], job_sequence_length=job_seq_length)

print("\n3. Corresponding first and second sets:")
print(init_jobs_corresponding)
print(init_agv_corresponding[0])

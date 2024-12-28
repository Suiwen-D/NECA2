import random
from typing import List, Tuple




class JobShopGA:
    def __init__(self, jobs_data: List[List[Tuple[int, int]]], 
                 population_size: int = 20,    
                 generations: int = 50,        
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 selection_method: str = "tournament",  
                 crossover_method: str = "order",    
                 mutation_method: str = "swap"):      
        self.jobs_data = jobs_data
        self.num_jobs = len(jobs_data)
        self.num_machines = len(jobs_data[0])
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method        
        self.best_fitness_history = []
        
    def create_chromosome(self) -> List[int]:
        """创建一个随机染色体"""
        chromosome = []
        for job in range(self.num_jobs):
            chromosome.extend([job] * self.num_machines)
        random.shuffle(chromosome)
        return chromosome
    
    def calculate_makespan(self, chromosome: List[int]) -> int:
        """计算完工时间"""
        # Record the current operation index for each job
        job_op_idx = [0] * self.num_jobs
        # Record the completion time of each machine
        machine_time = [0] * self.num_machines
        # Record the completion time of each job
        job_time = [0] * self.num_jobs
        
        for job_id in chromosome:
            # Skip if all operations for the job are scheduled
            if job_op_idx[job_id] >= self.num_machines:
                continue
                
            # Gets the machine and processing time for the current operation
            machine_id, proc_time = self.jobs_data[job_id][job_op_idx[job_id]]
            
            # Calculate Start Time
            start_time = max(machine_time[machine_id], job_time[job_id])
            
            # update time
            finish_time = start_time + proc_time
            machine_time[machine_id] = finish_time
            job_time[job_id] = finish_time
            
            job_op_idx[job_id] += 1
            
        return max(machine_time)

def read_job_shop_data(filename: str) -> List[List[Tuple[int, int]]]:
    """Read Job Scheduling Problem Data"""
    with open(filename, 'r') as f:
        f.readline()  # Skip description line
        n, m = map(int, f.readline().split())  # Read Scale
        
        jobs_data = []
        for _ in range(n):
            line = list(map(int, f.readline().split()))
            job = [(line[i], line[i+1]) for i in range(0, len(line), 2)]
            jobs_data.append(job)
            
        return jobs_data


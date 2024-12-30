import random
from typing import List, Tuple
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib to support Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei']  # For normal display of Chinese labels
plt.rcParams['axes.unicode_minus'] = False    # For normal display of minus sign

class JobShopGA:
    def __init__(self, jobs_data: List[List[Tuple[int, int]]], 
                 population_size: int = 20,    # Population size, number of chromosomes to keep in each generation
                 generations: int = 50,        # Maximum number of generations for genetic algorithm
                 mutation_rate: float = 0.2,   # Low mutation rate preserves good traits from parents, high stability but less exploration in early stages
                 crossover_rate: float = 0.8,  # High crossover rate may lead to lack of diversity when parent population lacks variety
                 selection_method: str = "tournament",
                 crossover_method: str = "uniform",
                 mutation_method: str = "swap",
                 stability_threshold: int = 20):     # Stability threshold, stop if no improvement for 20 generations
        # Initialize parameters
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
        self.stability_threshold = stability_threshold
        self.best_fitness_history = []
        
    def create_chromosome(self) -> List[int]:
        # Create a random chromosome
        chromosome = []
        for job in range(self.num_jobs):
            chromosome.extend([job] * self.num_machines)
        random.shuffle(chromosome)
        return chromosome
    
    def calculate_makespan(self, chromosome: List[int]) -> int:
        # Calculate completion time (fitness function)
        # Record current operation index for each job
        job_op_idx = [0] * self.num_jobs
        # Record completion time for each machine
        machine_time = [0] * self.num_machines
        # Record completion time for each job
        job_time = [0] * self.num_jobs
        
        for job_id in chromosome:
            # Skip if all operations for this job are scheduled
            if job_op_idx[job_id] >= self.num_machines:
                continue
                
            # Get machine and processing time for current operation
            machine_id, proc_time = self.jobs_data[job_id][job_op_idx[job_id]]
            
            # Calculate start time
            start_time = max(machine_time[machine_id], job_time[job_id])
            
            # Update times
            finish_time = start_time + proc_time
            machine_time[machine_id] = finish_time
            job_time[job_id] = finish_time
            
            # Current operation complete, move to next
            job_op_idx[job_id] += 1
            
        return max(machine_time)
    
    def tournament_selection(self, fitness_values: List[Tuple[int, List[int]]], tournament_size: int = 3) -> List[int]:
        """Tournament selection
        More flexible than other methods, can use different tournament sizes for different stages,
        or allow less fit individuals to win; prevents premature convergence"""
        tournament_indices = random.sample(range(len(fitness_values)), tournament_size)  # Default sample size of 3
        tournament = [fitness_values[i] for i in tournament_indices]
        winner = min(tournament, key=lambda x: x[0])  # Select individual with minimum fitness as winner
        return winner[1].copy()  # Return copy of winner's chromosome
    
    def random_sampling_selection(self, fitness_values: List[Tuple[int, List[int]]], sample_size: int = 5) -> List[int]:
        """Random sampling selection
        Randomly sample sample_size individuals from population, select the best one"""
        try:
            # Randomly sample sample_size individuals
            samples = random.sample(fitness_values, min(sample_size, len(fitness_values)))
            # Return the best individual from samples
            best_sample = min(samples, key=lambda x: x[0])
            return best_sample[1].copy()
        except Exception as e:
            print(f"Random sampling selection error: {str(e)}")
            return fitness_values[0][1].copy()  # Return best individual if error occurs
    
    def select_parent(self, fitness_values: List[Tuple[int, List[int]]]) -> List[int]:
        """Select parent based on selection method"""
        if self.selection_method == "tournament":
            return self.tournament_selection(fitness_values)
        elif self.selection_method == "random":  
            return self.random_sampling_selection(fitness_values)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def single_point_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Single point crossover: Cross at random position, ensure correct number of occurrences for each job"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
            
        try:
            size = len(parent1)
            point = random.randint(1, size-1)
            
            # Initialize child
            child = [-1] * size
            
            # Copy first half from parent1
            child[:point] = parent1[:point]
            
            # Count used jobs
            job_counts = [0] * self.num_jobs
            for gene in child[:point]:
                if gene != -1:
                    job_counts[gene] += 1
            
            # Get remaining genes from parent2, check job count constraints
            remaining_positions = list(range(point, size))
            for pos in remaining_positions:
                gene = parent2[pos]
                # Use gene if job hasn't reached maximum count
                if job_counts[gene] < self.num_machines:
                    child[pos] = gene
                    job_counts[gene] += 1
            
            # Fix unfilled positions
            unfilled = [i for i in range(size) if child[i] == -1]
            if unfilled:
                # Find jobs that need to be added and their required counts
                needed_jobs = []
                for job in range(self.num_jobs):
                    remaining_count = self.num_machines - job_counts[job]
                    needed_jobs.extend([job] * remaining_count)
                
                # Shuffle and fill
                random.shuffle(needed_jobs)
                for pos, job in zip(unfilled, needed_jobs):
                    child[pos] = job
            
            return child
            
        except Exception as e:
            print(f"Single point crossover error: {str(e)}")
            return parent1.copy()

    def uniform_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Uniform crossover: Randomly select genes from parents, ensure correct number of occurrences for each job"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
            
        try:
            size = len(parent1)
            child = [-1] * size
            job_counts = [0] * self.num_jobs
            
            # First pass: randomly select genes, check job count constraints
            for i in range(size):
                # Randomly decide which parent's gene to use
                if random.random() < 0.5:
                    gene = parent1[i]
                else:
                    gene = parent2[i]
                
                # Use gene if job hasn't reached maximum count
                if job_counts[gene] < self.num_machines:
                    child[i] = gene
                    job_counts[gene] += 1
            
            # Fix unfilled positions
            unfilled = [i for i in range(size) if child[i] == -1]
            if unfilled:
                # Find jobs that need to be added and their required counts
                needed_jobs = []
                for job in range(self.num_jobs):
                    remaining_count = self.num_machines - job_counts[job]
                    needed_jobs.extend([job] * remaining_count)
                
                # Shuffle remaining data and fill
                random.shuffle(needed_jobs)
                for pos, job in zip(unfilled, needed_jobs):
                    child[pos] = job
            
            return child
            
        except Exception as e:
            print(f"Uniform crossover error: {str(e)}")
            return parent1.copy()

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform crossover based on selected method"""
        if self.crossover_method == "single":
            return self.single_point_crossover(parent1, parent2)
        elif self.crossover_method == "uniform":
            return self.uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")

def read_job_shop_data(filename: str) -> List[List[Tuple[int, int]]]:
    """Read job shop scheduling problem data"""
    with open(filename, 'r') as f:
        f.readline()  # Skip description line
        n, m = map(int, f.readline().split())  # Read dimensions
        
        jobs_data = []
        for _ in range(n):
            line = list(map(int, f.readline().split()))
            job = [(line[i], line[i+1]) for i in range(0, len(line), 2)]
            jobs_data.append(job)
            
        return jobs_data

def test_with_small_data():
    """Test functionality using data_small.txt"""
    print("\n=== Testing with data_small.txt ===")
    
    # Read data from file
    try:
        jobs_data = read_job_shop_data('data_small.txt')
        print(f"\nSuccessfully read data_small.txt")
        print(f"Problem size: {len(jobs_data)} jobs × {len(jobs_data[0])} machines")
        print("\nJobs data:")
        for i, job in enumerate(jobs_data):
            print(f"Job {i}: {job}")
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        return
    
    # Initialize GA
    ga = JobShopGA(
        jobs_data=jobs_data,
        population_size=10,
        generations=5,
        selection_method="tournament",
        crossover_method="single"
    )
    
    print("\n1. Testing chromosome creation:")
    chromosome = ga.create_chromosome()
    print(f"Created chromosome: {chromosome}")
    print(f"Chromosome length: {len(chromosome)}")
    print(f"Job counts: {[chromosome.count(i) for i in range(ga.num_jobs)]}")
    
    print("\n2. Testing makespan calculation:")
    makespan = ga.calculate_makespan(chromosome)
    print(f"Makespan for random chromosome: {makespan}")
    
    print("\n3. Testing selection methods:")
    # Create some sample chromosomes with their fitness values
    sample_population = [(ga.calculate_makespan(ga.create_chromosome()), ga.create_chromosome()) 
                        for _ in range(5)]
    print("\nSample population fitness values:")
    for i, (fitness, chrom) in enumerate(sample_population):
        print(f"Individual {i}: Fitness = {fitness}")
    
    print("\nTesting tournament selection:")
    selected = ga.tournament_selection(sample_population)
    print(f"Selected chromosome: {selected}")
    
    print("\nTesting random sampling selection:")
    ga.selection_method = "random"
    selected = ga.random_sampling_selection(sample_population)
    print(f"Selected chromosome: {selected}")
    
    print("\n4. Testing crossover methods:")
    parent1 = ga.create_chromosome()
    parent2 = ga.create_chromosome()
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}")
    
    print("\nTesting single point crossover:")
    ga.crossover_method = "single"
    child = ga.single_point_crossover(parent1, parent2)
    print(f"Child (single point): {child}")
    print(f"Child job counts: {[child.count(i) for i in range(ga.num_jobs)]}")
    
    print("\nTesting uniform crossover:")
    ga.crossover_method = "uniform"
    child = ga.uniform_crossover(parent1, parent2)
    print(f"Child (uniform): {child}")
    print(f"Child job counts: {[child.count(i) for i in range(ga.num_jobs)]}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run tests with data_small.txt
    test_with_small_data()


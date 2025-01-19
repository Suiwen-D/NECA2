import random
from typing import List, Tuple
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib to support Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei']  # To properly display Chinese labels
plt.rcParams['axes.unicode_minus'] = False    # To properly display minus signs

class JobShopGA: 
    def __init__(self, jobs_data: List[List[Tuple[int, int]]],  
                 population_size: int = 20,    # Population size, i.e., how many chromosomes are retained each generation
                 generations: int = 50,        # Maximum number of generations the genetic algorithm will iterate
                 mutation_rate: float = 0.2,   # Low mutation rate retains good traits from parents, high stability, but insufficient exploration ability in early stages
                 crossover_rate: float = 0.8,  # Too high crossover rate can lead to many similar offspring when parental diversity is insufficient
                 selection_method: str = "tournament", 
                 crossover_method: str = "uniform", 
                 mutation_method: str = "swap", 
                 stability_threshold: int = 20): # Stability threshold, stop if no improvement after 20 generations
        # Assign values
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
        self.stability_threshold = stability_threshold  # New: Stability threshold
        self.best_fitness_history = [] 
         
    def create_chromosome(self) -> List[int]: 
        # Create a random chromosome
        chromosome = [] 
        for job in range(self.num_jobs): 
            chromosome.extend([job] * self.num_machines) 
        random.shuffle(chromosome) 
        return chromosome 
     
    def calculate_makespan(self, chromosome: List[int]) -> int: 
        # Calculate makespan (fitness function)
        # Record the current operation index for each job
        job_op_idx = [0] * self.num_jobs 
        # Record the completion time of each machine
        machine_time = [0] * self.num_machines 
        # Record the completion time of each job
        job_time = [0] * self.num_jobs 
         
        for job_id in chromosome: 
            # If all operations of this job have been scheduled, skip
            if job_op_idx[job_id] >= self.num_machines: 
                continue 
                 
            # Get the machine and processing time of the current operation
            machine_id, proc_time = self.jobs_data[job_id][job_op_idx[job_id]] 
             
            # Calculate start time
            start_time = max(machine_time[machine_id], job_time[job_id]) 
             
            # Update time
            finish_time = start_time + proc_time 
            machine_time[machine_id] = finish_time 
            job_time[job_id] = finish_time 
             
            # Current operation completed, move to next
            job_op_idx[job_id] += 1 
             
        return max(machine_time) 
     
    def tournament_selection(self, fitness_values: List[Tuple[int, List[int]]], tournament_size: int = 3) -> List[int]: 
        """Tournament Selection
        Tournament is more flexible, for example, using different tournament sizes in early and late stages, or allowing individuals with lower fitness to have a chance to win; thereby preventing premature convergence and insufficient exploration of the search space."""
        tournament_indices = random.sample(range(len(fitness_values)), tournament_size) # Default to sampling three for comparison
        tournament = [fitness_values[i] for i in tournament_indices] 
        winner = min(tournament, key=lambda x: x[0])  # Select the one with the smallest fitness as the winner
        return winner[1].copy() # Return a copy of its chromosome
     
    def random_sampling_selection(self, fitness_values: List[Tuple[int, List[int]]], sample_size: int = 5) -> List[int]: 
        #Random Sampling Selection
        #Randomly sample 'sample_size' individuals from the population and select the best one among them.
        try: 
            # Randomly sample 'sample_size' individuals
            samples = random.sample(fitness_values, min(sample_size, len(fitness_values))) 
            # Return the best individual in the samples
            best_sample = min(samples, key=lambda x: x[0]) 
            return best_sample[1].copy() 
        except Exception as e: 
            print(f"Random sampling selection error: {str(e)}") 
            return fitness_values[0][1].copy()  # Return the best individual in case of error
     
    def select_parent(self, fitness_values: List[Tuple[int, List[int]]]) -> List[int]: 
        #Select parent based on selection method
        if self.selection_method == "tournament": 
            return self.tournament_selection(fitness_values) 
        elif self.selection_method == "random":   
            return self.random_sampling_selection(fitness_values) 
        else: 
            raise ValueError(f"Unknown selection method: {self.selection_method}") 
     
    def single_point_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]: 
        #Single Point Crossover: Perform crossover at a random position and ensure each job appears the correct number of times
        if random.random() > self.crossover_rate: 
            return parent1.copy() 
             
        try: 
            size = len(parent1) 
            point = random.randint(1, size-1) 
             
            # Initialize offspring
            child = [-1] * size 
             
            # Copy the first part from parent1
            child[:point] = parent1[:point] 
             
            # Count the number of jobs already used
            job_counts = [0] * self.num_jobs 
            for gene in child[:point]: 
                if gene != -1:
                    job_counts[gene] += 1 
             
            # Get the remaining genes from parent2, ensuring job count constraints
            remaining_positions = list(range(point, size)) 
            for pos in remaining_positions: 
                gene = parent2[pos] 
                # If the job has not reached the maximum count, use it
                if job_counts[gene] < self.num_machines: 
                    child[pos] = gene 
                    job_counts[gene] += 1 
             
            # Repair unfilled positions
            unfilled = [i for i in range(size) if child[i] == -1] 
            if unfilled: 
                # Find the jobs that need to be filled and their required counts
                needed_jobs = [] 
                for job in range(self.num_jobs): 
                    remaining_count = self.num_machines - job_counts[job] 
                    needed_jobs.extend([job] * remaining_count) 
                 
                # Shuffle and fill the unfilled positions
                random.shuffle(needed_jobs) 
                for pos, job in zip(unfilled, needed_jobs): 
                    child[pos] = job 
             
            return child 
             
        except Exception as e: 
            print(f"Single point crossover error: {str(e)}") 
            return parent1.copy() 
     
    def uniform_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]: 
        #Uniform Crossover: Randomly select genes from parents and ensure each job appears the correct number of times
        if random.random() > self.crossover_rate: 
            return parent1.copy() 
             
        try: 
            # Initialize offspring chromosome
            size = len(parent1) 
            child = [-1] * size 
            job_counts = [0] * self.num_jobs # Record the number of times each job has appeared in the offspring
             
            # First pass: randomly select genes, ensuring job count constraints
            for i in range(size): 
                # Randomly decide which parent's gene to use with a 50% probability
                if random.random() < 0.5: 
                    gene = parent1[i] 
                else: 
                    gene = parent2[i] 
                 
                # If the job has not reached the maximum count, use it
                if job_counts[gene] < self.num_machines: 
                    child[i] = gene 
                    job_counts[gene] += 1 
             
            # Repair unfilled positions
            unfilled = [i for i in range(size) if child[i] == -1] 
            if unfilled: 
                # Find the jobs that need to be filled and their required counts
                needed_jobs = [] 
                for job in range(self.num_jobs): 
                    remaining_count = self.num_machines - job_counts[job] 
                    needed_jobs.extend([job] * remaining_count) 
                 
                # Shuffle the remaining jobs and fill the unfilled positions
                random.shuffle(needed_jobs) 
                for pos, job in zip(unfilled, needed_jobs): 
                    child[pos] = job 
             
            return child 
             
        except Exception as e: 
            print(f"Uniform crossover error: {str(e)}") 
            return parent1.copy() 
     
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]: 
        #Perform crossover based on the crossover method
        if self.crossover_method == "single": 
            return self.single_point_crossover(parent1, parent2) 
        elif self.crossover_method == "uniform": 
            return self.uniform_crossover(parent1, parent2) 
        else: 
            raise ValueError(f"Unknown crossover method: {self.crossover_method}") 
     
    #[0, 1, 2, 3, 4, 5, 6] choosing 2, 5  -> [0, 1, 5, 3, 4, 2, 6] 
    def swap_mutation(self, chromosome: List[int]) -> List[int]: 
        #Swap Mutation: Randomly swap two genes in the chromosome
        if random.random() > self.mutation_rate: 
            return chromosome 
             
        result = chromosome.copy() 
        idx1, idx2 = random.sample(range(len(chromosome)), 2) 
        result[idx1], result[idx2] = result[idx2], result[idx1] 
        return result 
     
    #[0, 1, 2, 3, 4, 5, 6] choosing insert 2 to 5  -> [0, 1, 3, 4, 5, 2, 6] 
    def insert_mutation(self, chromosome: List[int]) -> List[int]: 
        #Insert Mutation: Randomly select a gene and insert it into another random position
        if random.random() > self.mutation_rate: 
            return chromosome 
             
        result = chromosome.copy() 
        size = len(chromosome) 
        # Randomly select two different positions
        pos1, pos2 = random.sample(range(size), 2) 
         
        # Get the gene to insert
        gene = result[pos1] 
         
        # Remove the gene from the original position
        result.pop(pos1) 
         
        # Insert the gene at the new position
        result.insert(pos2, gene) 
         
        return result 
     
    #[0, 1, 2, 3, 4, 5, 6] choosing [2, 5]  -> [0, 1, 5, 4, 3, 2, 6] 
    def reverse_mutation(self, chromosome: List[int]) -> List[int]: 
        #Reverse Mutation: Randomly select a segment and reverse it
        if random.random() > self.mutation_rate: 
            return chromosome 
             
        result = chromosome.copy() 
        size = len(chromosome) 
        # Randomly select two positions as the reversal interval
        pos1, pos2 = sorted(random.sample(range(size), 2)) 
         
        # Reverse the genes within the interval
        result[pos1:pos2+1] = reversed(result[pos1:pos2+1]) 
         
        return result 
     
    def mutation(self, chromosome: List[int]) -> List[int]: 
        #Perform mutation based on the selected mutation method
        try: 
            if self.mutation_method == "swap": 
                return self.swap_mutation(chromosome) 
            elif self.mutation_method == "insert": 
                return self.insert_mutation(chromosome) 
            elif self.mutation_method == "reverse": 
                return self.reverse_mutation(chromosome) 
            else: 
                raise ValueError(f"Unknown mutation method: {self.mutation_method}") 
        except Exception as e: 
            print(f"Mutation operation error: {str(e)}") 
            return chromosome.copy() 
     
    # Calculate the average improvement over the last ten generations
    def calculate_improvement_rate(self, improvement_size: int = 10) -> float: 
        # If fewer than improvement_size, return inf 
        if len(self.best_fitness_history) < improvement_size: 
            return float('inf') 
             
        recent_values = self.best_fitness_history[-improvement_size:] 
        if len(recent_values) < 2: 
            return 0.0 
             
        improvements = [abs(recent_values[i] - recent_values[i-1])  
                       for i in range(1, len(recent_values))] 
        return sum(improvements) / len(improvements) 
 
    def is_system_stable(self) -> Tuple[bool, str]: 
        #Determine if the system has stabilized
        if len(self.best_fitness_history) < self.stability_threshold: 
            return False, "Insufficient generations, continue optimizing" 
             
        # Check if the best solutions have stagnated 
        recent_best = self.best_fitness_history[-self.stability_threshold:] 
        if len(set(recent_best)) == 1: 
            return True, f"The best solution has not improved for {self.stability_threshold} consecutive generations" 
             
        # Check improvement rate, stop if below 0.1 
        improvement_rate = self.calculate_improvement_rate() 
        if improvement_rate < 0.1:  # Improvement rate threshold 
            return True, f"Low improvement rate: {improvement_rate:.4f}" 
         
        return False, "The system is still optimizing"

    def run(self): 
        #Run the Genetic Algorithm
        print("\nInitializing population...")
        # Here, each type has 20 chromosomes
        population = [self.create_chromosome() for _ in range(self.population_size)]
        best_solution = None
        best_makespan = float('inf')
        
        print(f"\nUsing selection method: {self.selection_method}")
        print(f"Using crossover method: {self.crossover_method}")
        print(f"Using mutation method: {self.mutation_method}")
        print("\nStarting optimization...")
        
        try: # Iterate through generations
            for generation in tqdm(range(self.generations), desc="Optimization Progress"):
                # Calculate fitness: makespan for each chromosome
                fitness_values = [(self.calculate_makespan(chrom), chrom) 
                                for chrom in population]
                fitness_values.sort()
                
                # Update global best solution: if the current best individual's makespan is better, update the best solution
                if fitness_values[0][0] < best_makespan:
                    best_makespan = fitness_values[0][0]
                    best_solution = fitness_values[0][1].copy()
                    print(f"\nGeneration {generation+1} found a new best solution:")
                    print(f"Makespan: {best_makespan}")
                
                self.best_fitness_history.append(best_makespan)
                
                # Check system stability, if stable then stop.
                # Here are two methods: no change for 20 generations or improvement rate less than 0.1 for 10 generations
                is_stable, reason = self.is_system_stable()
                if is_stable:
                    print(f"\nSystem has stabilized, reason: {reason}")
                    print(f"Current generation: {generation + 1}")
                    break
                
                # Generate a new generation, retain elites. The chromosome with the least makespan is directly retained to the next generation, but it can still be selected as a parent and participate in crossover and mutation operations.
                new_population = [fitness_values[0][1].copy()]  # Retain the best individual
                
                # The remaining individuals are generated through selection -> crossover -> mutation. Here, specific methods need to be chosen
                while len(new_population) < self.population_size:
                    parent1 = self.select_parent(fitness_values)
                    parent2 = self.select_parent(fitness_values)
                    child = self.crossover(parent1, parent2)
                    child = self.mutation(child)
                    
                    # Ensure that the gene set is not lost after mutation, if there's a problem, use parent1 instead
                    if len(child) == len(parent1) and set(child) == set(parent1):
                        new_population.append(child)
                    else:
                        new_population.append(parent1.copy())
                
                population = new_population
                
                # Every 10 generations, record progress to ga_progress.txt.
                if (generation + 1) % 10 == 0:
                    with open('ga_progress.txt', 'a') as f:
                        f.write(f"Generation {generation + 1} Best Makespan: {best_makespan}\n")
                        f.write(f"Improvement Rate: {self.calculate_improvement_rate():.4f}\n")
            # Best solution and its corresponding makespan
            return best_solution, best_makespan
            
        except Exception as e:
            print(f"\nOptimization process error: {str(e)}")
            if best_solution is None:
                return population[0], float('inf')
            return best_solution, best_makespan
    
    def plot_progress(self):
        """Plot the optimization progress"""
        plt.figure(figsize=(10, 5))
        
        # Plot the change in best makespan
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history)
        plt.title('Change in Best Makespan', fontsize=12)
        plt.xlabel('Generation', fontsize=10)
        plt.ylabel('Makespan', fontsize=10)
        plt.grid(True)
        
        # Plot the change in improvement rate
        plt.subplot(1, 2, 2)
        # Calculate the difference between consecutive best solutions
        improvements = [abs(self.best_fitness_history[i] - self.best_fitness_history[i-1]) 
                       for i in range(1, len(self.best_fitness_history))]
        plt.plot(improvements)
        plt.title('Change in Improvement Rate', fontsize=12)
        plt.xlabel('Generation', fontsize=10)
        plt.ylabel('Improvement Amount', fontsize=10)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Save the best result to best_solution.txt
    def save_solution(self, solution: List[int], makespan: int, filename: str = "best_solution.txt"):
        #Save the scheduling plan of the best solution
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*50 + "\n")
                f.write(f"Problem Size: {self.num_jobs} Jobs x {self.num_machines} Machines\n")
                f.write(f"Best Makespan: {makespan}\n\n")
                
                # Convert one-dimensional encoding to dataset format
                f.write("Best Solution Encoding:\n")
                # Write problem size (consistent with dataset format)
                f.write(f"{self.num_jobs} {self.num_machines}\n")
                
                # Count the positions of each job
                job_positions = [[] for _ in range(self.num_jobs)]
                for pos, job in enumerate(solution):
                    job_positions[job].append(pos + 1)  # Positions start from 1
                
                # Output the positions of each job according to the dataset format
                for job_pos in job_positions:
                    f.write(" ".join(map(str, job_pos)) + "\n")
                
                f.write("="*50 + "\n")
                
        except Exception as e:
            print(f"Error saving the best solution: {str(e)}")

def read_job_shop_data(filename: str) -> List[List[Tuple[int, int]]]:
    #Read job shop scheduling problem data
    with open(filename, 'r') as f:
        f.readline()  # Skip the description line
        n, m = map(int, f.readline().split())  # Read the size
        
        jobs_data = []
        for _ in range(n):
            line = list(map(int, f.readline().split()))
            job = [(line[i], line[i+1]) for i in range(0, len(line), 2)]
            jobs_data.append(job)
            
        return jobs_data

def test_different_sizes(base_params: dict, test_cases: List[str], results_file: str = "test_results.txt"):
    #Test problems of different sizes
    print("\nStarting tests for problems of different sizes...")
    print("=" * 40)
    
    # Define 6 different parameter combinations
    parameter_combinations = [
        {   # Combination 1: Base Parameters
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'selection_method': "tournament",
            'crossover_method': "single",
            'mutation_method': "swap"
        },
        {   # Combination 2: High Mutation Rate
            'mutation_rate': 0.4,
            'crossover_rate': 0.8,
            'selection_method': "tournament",
            'crossover_method': "single",
            'mutation_method': "insert"
        },
        {   # Combination 3: High Crossover Rate
            'mutation_rate': 0.2,
            'crossover_rate': 0.9,
            'selection_method': "tournament",
            'crossover_method': "uniform",
            'mutation_method': "reverse"
        },
        {   # Combination 4: Random Selection
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'selection_method': "random",
            'crossover_method': "single",
            'mutation_method': "swap"
        },
        {   # Combination 5: Uniform Crossover Optimization
            'mutation_rate': 0.3,
            'crossover_rate': 0.85,
            'selection_method': "tournament",
            'crossover_method': "uniform",
            'mutation_method': "insert"
        },
        {   # Combination 6: Comprehensive Optimization
            'mutation_rate': 0.25,
            'crossover_rate': 0.8,
            'selection_method': "random",
            'crossover_method': "uniform",
            'mutation_method': "reverse"
        }
    ]
    
    # Clear or create the results file
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Job Shop Scheduling Problem Test Results\n")
        f.write("=" * 40 + "\n\n")
    
    for case_file in test_cases:
        print(f"\nTesting dataset: {case_file}")
        try:
            # Read data
            jobs_data = read_job_shop_data(case_file)
            n_jobs = len(jobs_data)
            n_machines = len(jobs_data[0])
            print(f"Problem Size: {n_jobs} Jobs x {n_machines} Machines")
            
            # Set base parameters for different problem sizes
            if n_machines <= 5:  # Small-scale problems: 30 chromosomes, 50 generations
                base_population = 30
                base_generations = 50
            elif n_machines <= 10:  # Medium-scale problems
                base_population = 50
                base_generations = 100
            else:  # Large-scale problems
                base_population = 100
                base_generations = 150
            
            # Test each parameter combination
            best_overall_makespan = float('inf')
            best_params = None
            best_overall_solution = None
            
            for i, params in enumerate(parameter_combinations, 1):
                print(f"\nTesting parameter combination {i}/6...")
                
                # Merge base parameters; without this, each test of different problem sizes would require manual parameter adjustments
                # marge population_size and generations into 6 different combinations
                current_params = params.copy()
                current_params.update({
                    'population_size': base_population,
                    'generations': base_generations
                })
                
                # Create GA instance
                ga = JobShopGA(jobs_data=jobs_data, **current_params)
                
                # Run the algorithm
                start_time = time.time()
                best_solution, best_makespan = ga.run()
                end_time = time.time()
                total_time = end_time - start_time
                
                # Update the best result
                if best_makespan < best_overall_makespan:
                    best_overall_makespan = best_makespan
                    best_params = current_params
                    best_overall_solution = best_solution
                
                # Save results
                with open(results_file, 'a', encoding='utf-8') as f:
                    f.write(f"\nDataset: {case_file} - Parameter Combination {i}\n")
                    f.write(f"Problem Size: {n_jobs} Jobs x {n_machines} Machines\n")
                    f.write(f"Parameter Settings:\n")
                    for key, value in current_params.items():
                        f.write(f"- {key}: {value}\n")
                    f.write(f"Run Results:\n")
                    f.write(f"- Best Makespan: {best_makespan}\n")
                    f.write(f"- Run Time: {total_time:.2f} seconds\n")
                    f.write("-" * 40 + "\n")
                
                # Save optimization progress plot
                plt.figure(figsize=(10, 5))
                plt.plot(ga.best_fitness_history)
                plt.title(f'Optimization Progress ({n_jobs}x{n_machines}) - Parameter Combination {i}')
                plt.xlabel('Generation')
                plt.ylabel('Best Makespan')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'optimization_progress_{n_jobs}x{n_machines}_params{i}.png')
                plt.close()
            
            # Record the best parameter combination
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{case_file} Best Overall Results:\n")
                f.write(f"Best Makespan: {best_overall_makespan}\n")
                f.write("Best Parameter Combination:\n")
                for key, value in best_params.items():
                    f.write(f"- {key}: {value}\n")
                f.write("=" * 40 + "\n")
            
            # Save the best solution's scheduling plan
            if best_overall_solution is not None:
                ga.save_solution(best_overall_solution, best_overall_makespan)
            
        except Exception as e:
            print(f"Testing error: {str(e)}")
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nDataset: {case_file}\n")
                f.write(f"Testing error: {str(e)}\n")
                f.write("=" * 40 + "\n")
                
# After the code, here is the translation of the remaining Chinese code blocks.

# Program startup, print prompt information.
def main():
    print("Job Shop Scheduling Optimization - Genetic Algorithm")
    print("=" * 40)
    
    # Base parameter settings
    base_params = {
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'selection_method': "tournament",
        'crossover_method': "order",
        'mutation_method': "swap"
    }
    
    # List of test cases
    test_cases = [
        'data_small.txt',    # Small-scale problem (5 machines)
        'data.txt',          # Medium-scale problem (10 machines)
        'data_large.txt'     # Large-scale problem (15 machines)
    ]
    
    try:
        
        test_different_sizes(base_params, test_cases)
        
        print("\nAll tests completed!")
        print("Detailed results have been saved to test_results.txt")
        print("Optimization progress plots have been saved as optimization_progress_*.png")
        
    except Exception as e:
        print(f"\nRun error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


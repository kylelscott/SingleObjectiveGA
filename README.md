# SingleObjectiveGA
Implementation of a simple Single Objective Evolutionary Algorithmn for sinusoidal optimization function. Created for fulfillment of CSE 598: Bio-Inspired AI and Optimization. Please reference if you use, and switch up some hyperparameters for your own use case :).


A common structure was chosen for the Genetic 
Algorithm implemented. Specifically, evolution was 
conducted by following the cyclic execution of 
Population Generation, Intercourse, and Mutation. 
The generation of the next population was executed 
through a series of sub-procedures: fitting the current 
population, selecting the most optimal candidates for 
breeding, and iteratively generating children until the 
next generation was of the size as the initial 
generation. 

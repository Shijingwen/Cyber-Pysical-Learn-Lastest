############################################################
# Generate 10 trails according to building case 1.
############################################################
include("example_HybridAutomataBuilding.jl")
# Configurations has been saved in project_dir/config/case*.
# Data will be saved project_dir/data/creat_time.
# The create time will be recorded into project_dir/config/case*/create_times.csv.
# simulation("case1", 10, 100000) # case_name, num_trails, sum_steps



############################################################
# Reconstruct the hybridAutomata.
############################################################
include("LearnHybridAutomata.jl")
using .HybridAutomataLearner
# Run selected cases.
analyze_results_only = false
learn_parts = ["control"] # ["physical", "control"]
for part in learn_parts
    HybridAutomataLearner.learn_case("case1", part, analyze_results_only)
end

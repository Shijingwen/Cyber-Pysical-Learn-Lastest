case_name = "case1"
mode = "non-linear-origin"
creat_times = readdlm("./config/" * case_name * "/create_times.csv", '\n')
parent_dir = string(dirname(@__DIR__))
slice_creat_times = creat_times
names = ["Â_T", "B̂_ext", "Ĉ_U", "D̂_HVAC_U", "Ê_T_U", "Q̃"]

# Prepare paras dictionary.
path = "./config/" * case_name * "/control_map.csv"
control_map = readdlm(path, ',', Float64, '\n')
n_c, n_r = size(control_map)
matrixs = [zeros(n_r, n_r), zeros(n_r, n_r), zeros(n_r, n_c),
zeros(n_r, n_c), zeros(n_r, n_c), zeros(n_r, 1)]
keys = ["A", "B", "C", "D", "E", "Q"]
paras = Dict((zip(keys, matrixs)))

# Load the trained parameters and save to dictionary.
for (index, trace) in enumerate(slice_creat_times)
    println("====================================")
    println(string(index) * ": " * trace)
    for (n, k) in zip(names, keys)
    # for (index, n) in enumerate(names)
        full_path = parent_dir * "/results/" * trace * "/para_" * n * "_32768_" * mode * ".csv"
        @show full_path
        paras[k] = readdlm(full_path, ',', Float64, '\n')
    end
end

# Check whether approximate.
for i in keys
    for j in keys
        if (i!= j) && (size(paras[i]) == size(paras[j]))
            @show i, j
            @show norm(paras[i] - paras[j])/norm(paras[i])
        end
    end
end

# Check whether approximate.
println("====================")
for i in keys
    # @show rank(paras[i])
    for j in keys
        if (i!= j) && (size(paras[i]) == size(paras[j]))
            @show i, j
            @show norm(paras[i] + paras[j])/norm(paras[i])
        end
    end
end

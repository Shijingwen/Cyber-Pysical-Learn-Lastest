include("HybridAutomataBuilding.jl")
using .HybridAutomata
using Dates, LinearAlgebra, DelimitedFiles

# Generate the parameters of the first case for building.
function simulation(case_name, num_trails, sum_steps=100000)
    # [1]Parameters about build structure.
    dir_config = string(dirname(@__DIR__)) * "/config/" * case_name

    system_build = HybridAutomata.System()
    system_build.room_size =
        readdlm(dir_config * "/room_size.csv", ',', Int, '\n')
    system_build.room_num = length(system_build.room_size)
    system_build.room_THVAC = [i < 2 ? 55 : 50 for i in system_build.room_size] # Small room 55F, big room 50F
    # In the diagonal location, we fill in the sum of neibors. It's useful when calculate the physical dynamics.
    system_build.room_adjecent =
        readdlm(dir_config * "/room_adjecent.csv", ',', Int, '\n')

    # [2]Parameters about controllers.
    system_build.control_group_num =
        readdlm(dir_config * "/control_group_num.csv", ',', Int, '\n')
    system_build.control_num = length(system_build.control_group_num)
    system_build.control_status = zeros(system_build.control_num)
    system_build.control_map =
        readdlm(dir_config * "/control_map.csv", ',', Int, '\n')
    system_build.control_rooms =
        system_build.control_map' * system_build.control_status
    # [3]Parameters about physical transition.
    system_build.A = Diagonal(0.1 * ones(system_build.room_num))  # α = 0.1
    system_build.B = Diagonal(0.1 * ones(system_build.room_num))  # β = 0.1
    # system_build.Q_internal = 1.36 * ones(system_build.room_num)  # ω = 1.36
    # Used different Q_internal
    system_build.Q_internal = [0.3 + i * 0.2 for i in 1:system_build.room_num]
    system_build.φ = 0.6
    system_build.noise_avg = 0
    system_build.noise_var = 0.1
    system_build.T_ext_avg = 85
    system_build.T_ext_var = 4.33
    system_build.THVAC_big = 50
    system_build.THVAC_small = 55

    # [4]Parameters about control transition.
    system_build.T_min = 70
    system_build.T_max = 80
    Δt = 0.1

    plot_room_indexs = [1, 2, 11, 12]

    # Automatically generate 10 new trails.
    create_time_all = Any[]
    for i = 1:num_trails
        # Prepare the directory.
        create_time = string(Dates.now())
        @show create_time
        push!(create_time_all, create_time)
        save_trace_path =
            string(dirname(@__DIR__)) * "/data/" * create_time * "/"
        save_fig_path = string(dirname(@__DIR__)) * "/fig/" * create_time * "_"
        mkdir(save_trace_path)

        # Change the initial values.
        system_build.room_Temp = rand(65:85, system_build.room_num)


        println( "[" * string(i) * "] Start simulation of system.")
        HybridAutomata.start(
            system_build,
            Δt,
            sum_steps,
            plot_room_indexs,
            save_fig_path,
            save_trace_path,
        )
    end
    # Save the create time into lists.
    writedlm(dir_config * "/create_times.csv", create_time_all, ',')
end

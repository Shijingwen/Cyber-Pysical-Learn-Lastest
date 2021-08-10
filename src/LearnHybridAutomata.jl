module HybridAutomataLearner
include("LearnPhysical.jl")
using .OLS
include("LearnControl.jl")
using .Control
using DelimitedFiles, LinearAlgebra, Plots, JLD

export Ground, learn_case

mutable struct Ground
    Δt::Float32             # 0.1 hour
    A::Array
    B::Array
    C::Array
    D::Array
    E::Array
    Q::Array
    room_THVAC::Array
    room_adjecent::Array
    control_map::Array
    A_origin::Array
    B_origin::Array
    Q_internal::Array
    φ::Float32              # 0.6
    beta::Array             # beta is ground matrix of weight for control.
    v::Array                # v is ground matrix of offset for control.
    Ground() = new()        # Allow imcomplete initialization.
end

function run_one_trace(part, creat_time, parent_dir, ground, sum_steps, p_modes, C, dim)
    Y_trace, U_trace, Ur_trace, T_ext, noise = load_files(creat_time, parent_dir)
    results = zeros(dim)

    for (index, sample_num) in enumerate(sum_steps)
        println("-------------------------------------------")
        @show sample_num
        index_end = sample_num + 1
        if part == "physical"
        # 1-Learn the physical part.
            trace_paras = OLS.format_ols(
                Y_trace[:, 1:index_end],
                U_trace[:, 1:index_end],
                Ur_trace[:, 1:index_end],
                T_ext[:, 1:index_end],
                noise[:, 1:index_end],
            )
            tmp_result = learn_physical(p_modes, trace_paras, creat_time, parent_dir,
            ground, index_end, sample_num, dim)
            results[index, :, :] = results[index, :, :] + tmp_result
        else
            # 2-Learn the control part.
            tmp_result = learn_control(sample_num, Y_trace, U_trace, index_end, C, ground.beta, ground.v)
            results[index, :, :, :, :] = results[index, :, :, :, :] + tmp_result
        end
    end
    results
end

function learn_control(sample_num, Y_trace, U_trace, index_end, C, beta_true, v_true)
    data = Control.load_control_format(sample_num, Y_trace[:, 1:index_end], U_trace[:, 1:index_end])
    results = Control.start_train_svm(data, C, beta_true, v_true)
    results
end

function learn_physical(p_modes, trace_paras, creat_time, parent_dir, ground, index_end, sample_num, dim)
    # check(trace_paras, ground)
    tmp_results = zeros(dim[2], dim[3]) # n_mode, n_paras

    for (index, mode) in enumerate(p_modes)
        @show mode
        paras, names = OLS.ols(mode, trace_paras, ground.Δt, ground.room_THVAC, ground.control_map')
        tmp_results[index, :] = evaluation(mode, paras, ground)
        # @show sample_num, tmp_results
        # save_paras(
        #     mode,
        #     sample_num,
        #     paras,
        #     names,
        #     creat_time,
        #     parent_dir * "/results/",
        # )
    end
    tmp_results
end

function load_files(creat_time, parent_dir)
    # Create paths
    Y_trace_path = parent_dir * "/data/" * creat_time * "/trace_Y.csv"
    U_trace_path = parent_dir * "/data/" * creat_time * "/trace_U.csv"
    Ur_trace_path = parent_dir * "/data/" * creat_time * "/trace_Ur.csv"
    T_ext_path = parent_dir * "/data/" * creat_time * "/trace_T_ext.csv"
    dif_air_path = parent_dir * "/data/" * creat_time * "/trace_dif_air.csv"
    dif_external_path =
        parent_dir * "/data/" * creat_time * "/trace_dif_external.csv"
    dif_neighbor_path =
        parent_dir * "/data/" * creat_time * "/trace_dif_neighbor.csv"
    noise_path = parent_dir * "/data/" * creat_time * "/trace_noise.csv"

    # Load files
    Y_trace = readdlm(Y_trace_path, ',', Float64, '\n')
    U_trace = readdlm(U_trace_path, ',', Float64, '\n')
    Ur_trace = readdlm(Ur_trace_path, ',', Float64, '\n')
    T_ext = readdlm(T_ext_path, ',', Float64, '\n')
    noise = readdlm(noise_path, ',', Float64, '\n')
    # We don't know the exact form of formula, so don't use the following files now.
    # dif_external = readdlm(dif_external_path, ',', Float64, '\n')
    # dif_neighbor = readdlm(dif_neighbor_path, ',', Float64, '\n')
    # dif_air = readdlm(dif_air_path, ',', Float64, '\n')
    Y_trace, U_trace, Ur_trace, T_ext, noise
end

# Calculate the Error
function evaluation(mode, paras, ground)
    compare = [ground.A, ground.B, ground.C, ground.D, ground.E, ground.Q]
    len_compare = length(compare)
    result = zeros(len_compare)
    for (index, (i, j)) in enumerate(zip(paras, compare))
        if length(i) == 1 # Float type
            D = i*ground.control_map'
            result[index] = abs(norm(j - D))
        else
            result[index] = abs(norm(j - i)) # Absolute
            # result[index] = norm(j - i)/norm(j)
        end
    end
    result
end

function check(trace_paras, ground, show_detail = true)
    for t in 1:3
        check_one_step(t, trace_paras, ground, show_detail)
    end
end

function check_one_step(n_t, trace_paras, ground, show_detail)
    Δt = ground.Δt
    Y_out = trace_paras[1][:, n_t]
    Y_in = trace_paras[2][:, n_t]
    U_in = trace_paras[3][:, n_t]
    Ext_in = ones(12, 1) * trace_paras[4][n_t]
    Noise_in = trace_paras[5][:, n_t]

    # Objective function LearnPhysical.jl
    # Y_out_estimate =
    #     ground.A * Y_in +
    #     ground.B * (Ext_in) +
    #     ground.C * U_in +
    #     ground.Q +
    #     + ground.room_THVAC .* (ground.D * U_in) +
    #     Y_in .* (ground.E * U_in) +
    #     Δt * Noise_in

    # Physical function at HybridAutomataBuilding
    # dif_external = (Ext_in - Y_in)
    # dif_sum_neighbor = (ground.room_adjecent * Y_in)
    # dif_air = (ground.room_THVAC - Y_in)
    # dif_Temp =
    #     Δt * (
    #         ground.A_origin * dif_external +
    #         ground.B_origin * dif_sum_neighbor +
    #         ground.Q_internal * ones(1, n_t) +
    #         ground.φ * (dif_air .* (ground.control_map' * U_in)) +
    #         Noise_in
    #     )

    Y_out_estimate =
        ground.A * Y_in +
        ground.B * (Ext_in) +
        ground.C * U_in +
        ground.Q +
        Δt * Noise_in

    dif_external = (Ext_in - Y_in)
    dif_sum_neighbor = (ground.room_adjecent * Y_in)
    dif_air = -25
    dif_Temp =
        Δt * (
            ground.A_origin * dif_external +
            ground.B_origin * dif_sum_neighbor +
            ground.Q_internal * ones(1, 1) +
            ground.φ * dif_air * ground.control_map' * U_in +
            Noise_in
        )
    Y_recalculate = Y_in + dif_Temp
    @show all(Y_recalculate .≈ Y_out_estimate)
    @show all(Y_out .≈ Y_out_estimate)
    if show_detail
        @show Y_in
        @show Y_out
        @show Y_recalculate
        @show Y_out_estimate
    end
end

function save_paras(mode, sample_num, paras, names, creat_time, parent_dir)
    write_dir = parent_dir * creat_time
    if isdir(write_dir) == false
        mkdir(write_dir)
    end
    for (i, j) in zip(names, paras)
        wpath = write_dir * "/para_" * i * "_" * string(sample_num) * "_" * mode * ".csv"
        writedlm(wpath, j, ',')
    end
end

function save_result(mode, sample_num, result, creat_time, parent_dir)
    write_dir = parent_dir * creat_time
    if isdir(write_dir) == false
        mkdir(write_dir)
    end
    wpath = write_dir * "/result_" * string(sample_num) * "_" * mode * "csv"
    writedlm(wpath, result, ',')
end

function draw_physical_results(p_modes, sum_steps, results, parent_dir, disp=false)
    @show size(results)
    fig_dir = parent_dir * "/results/figs/"
    # x = [log10(i) for i in sum_steps]
    # x = sum_steps
    x = [log2(i) for i in sum_steps]
    paras = ["Â_T", "B̂_ext", "Ĉ_U", "D̂_HVAC_U", "Ê_T_U", "Q̃"]
    labels = ["A", "B", "C", "D", "E", "Q"]
    for (index, para) in enumerate(paras)
        if index == 5
            tmp = zeros(size(results[:, :, index]))
            tmp[:, 1:1, 1] = results[:, 1:1, index]
            tmp[:, 2:2, 1] = results[:, 2:2, index-1] # Use D replace E (clip)
            tmp[:, 3:3, 1] = results[:, 3:3, index-1] # Use D replace E (clip-seive)
            y = tmp
        else
            y = results[:, :, index]
        end
        if index == 3
            loc_legend = :top
        else
            loc_legend = :topright
        end

        linestyles = [:dot :dash :solid]
        linecolors = ["#0571b0" "#404040" "#ca0020"]
        # linecolors = [:red, :grey, :blue]
        p = plot(x, y, title = "Error_" * labels[index],linecolor = linecolors,legendfontsize=13, legend = loc_legend,
        xguidefontsize=13, yguidefontsize=13, label = [p_modes[1] p_modes[2] p_modes[3]], linestyle=linestyles, lw = 2)
        xlabel!("Number of Training Samples (log2)")
        ylabel!("Absolute Error")
        if disp == true
            display(p)
        else
            fig_path = fig_dir * "/physical_"* para * ".pdf"
            savefig(p, fig_path)
        end
    end
end

function draw_control_dif_C(C, data, para, y_label, parent_dir, flag_combine=true)
    labels = ["controller1" "controller2" "controller3" "controller4"]
    x = [log10(i) for i in C]

    if flag_combine == false
        for i in 1:2
            y = data[:, i, :]
            linestyles = [:dot :dash :dashdot :solid]
            linecolors = ["#0571b0" "#404040" "#92c5de" "#ca0020"]
            p = plot(x, y', label = labels, lw = 2)
            p = plot(x, y',linecolor = linecolors,legendfontsize=13,
            xguidefontsize=13, yguidefontsize=13, label = labels, linestyle=linestyles, lw = 2)
            xlabel!("Value of C (log10)")
            ylabel!(y_label)
            fig_path = parent_dir * "/results/figs/" * "control_difc_" * para * "_" * string(i-1) * ".pdf"
            savefig(p, fig_path)
        end
    else
        n_controller = 4
        avg_controller = zeros(size(data)[2:end])
        for i in 1:n_controller
            avg_controller[:, :] = avg_controller[:, :] .+ data[i, :, :]
        end
        avg_controller = avg_controller/n_controller
        y = avg_controller'
        linestyles = [:dot :solid]
        linecolors = ["#0571b0" "#ca0020"]
        p = plot(x, y, label = ["Tmax" "Tmin"], linestyle=linestyles, linecolor=linecolors,
        lw = 2, legendfontsize=13, xguidefontsize=13, yguidefontsize=13)
        xlabel!("Value of C (log10)")
        ylabel!(y_label)
        fig_path = parent_dir * "/results/figs/" * "control_difc_" * para * "_avg.pdf"
        savefig(p, fig_path)
    end
end

function draw_control_dif_samples(sum_steps, data, para, y_label, parent_dir, flag_combine=true)
    labels = ["controller1" "controller2" "controller3" "controller4"]
    x = [log2(i) for i in sum_steps]

    if flag_combine == false
        for i in 1:2
            y = data[:, :, i]
            p = plot(x, y, label = labels, lw = 2)
            xlabel!("Number of Training Samples (2^x)")
            ylabel!(y_label)
            fig_path = parent_dir * "/results/figs/" * "control_difs_" * para * "_" * string(i-1) * ".pdf"
            savefig(p, fig_path)
        end
    else
        n_controller = 4
        avg_controller = zeros(size(data)[1], size(data)[3])
        for i in 1:n_controller
            avg_controller[:, :] = avg_controller[:, :] .+ data[:, i, :]
        end
        avg_controller = avg_controller/n_controller
        y = avg_controller
        linestyles = [:dot :solid]
        linecolors = ["#0571b0" "#ca0020"]
        p = plot(x, y, label = ["Tmax" "Tmin"], linestyle=linestyles, linecolor=linecolors,
        lw = 2, legendfontsize=11, xguidefontsize=13, yguidefontsize=13)
        xlabel!("Number of Training Samples (log2)")
        ylabel!(y_label)
        fig_path = parent_dir * "/results/figs/" * "control_difs_" * para * "_avg.pdf"
        savefig(p, fig_path)
    end
end

function draw_control_results(C, sum_steps, results, parent_dir, disp=false)
    # (n_step, n_controller, n_rules, n_C, n_paras)
    paras = ["v_error", "beta_error", "train_error", "test_error"]
    for i in 1:length(paras)
        if i > 2
            y_label = "Error (%)"
        else
            y_label = "Error"
        end
        draw_control_dif_C(C, results[end, :, :, :, 1], paras[i], y_label, parent_dir, false)
        draw_control_dif_samples(sum_steps, results[:, :, :, 2, i], paras[i], y_label, parent_dir)
    end

end

function analyze_results(part, p_modes, C, sum_steps,save_path_result, parent_dir)
    results = JLD.load(save_path_result)["data"]
    if part == "physical"
        draw_physical_results(p_modes, sum_steps, results, parent_dir)
    else
        draw_control_results(C, sum_steps, results, parent_dir)
    end
end

function run_multi_trails(part, case_name, all_results, parent_dir, ground, sum_steps, p_modes, C, save_dir)
    # Run 10 trails.
    creat_times = readdlm("./config/" * case_name * "/create_times.csv", '\n')
    slice_creat_times = creat_times[1:1]
    n_trail = length(slice_creat_times)
    for (index, trace) in enumerate(slice_creat_times)
        println("====================================")
        println(string(index) * ": " * trace)
        results = run_one_trace(part, trace, parent_dir, ground, sum_steps, p_modes, C, size(all_results))
        save_path = save_dir * trace * ".jld"
        save(save_path, "data", results)
        all_results = all_results .+ results
        println("Trail result has been save at " * save_path)
    end

    avg_results = all_results * (1.0 / n_trail)
    save_path = save_dir * "avg.jld"
    save(save_path, "data", avg_results)
    println("Averayge result has been save at " * save_path)
end

function load_groundtruth(dir_config, mode="non-linear")
    Δt = 0.1
    ground = Ground()
    ground.Δt = Δt
    ground.room_THVAC = readdlm(dir_config * "/room_THVAC.csv", ',', Int, '\n') # Small room 55F, big room 50F
    ground.room_adjecent =
        readdlm(dir_config * "/room_adjecent.csv", ',', Int, '\n')
    ground.control_map =
        readdlm(dir_config * "/control_map.csv", ',', Int, '\n')
    room_num = size(ground.control_map)[2]
    control_num = size(ground.control_map)[1]
    ground.A_origin = Diagonal(0.1 * ones(room_num))  # α = 0.1
    ground.B_origin = Diagonal(0.1 * ones(room_num))  # β = 0.1
    ground.Q_internal = [0.3 + i * 0.2 for i in 1:room_num]
    ground.φ = 0.6
    ground.A = I - Δt * ground.A_origin + Δt * ground.B_origin * ground.room_adjecent
    ground.B = Δt * ground.A_origin
    ground.Q = Δt * ground.Q_internal
    ground.beta = readdlm(dir_config * "/beta_true.csv", ',', Float32, '\n')
    ground.v = readdlm(dir_config * "/v_true.csv", ',', Float32, '\n')

    if mode == "linear"
        # For linear
        ground.C = Δt * ground.φ * -25 * ground.control_map'
    else
        # For non-linear
        ground.C = zeros(room_num, control_num)
    end
    ground.D = Δt * ground.φ * ground.control_map' # For non-linear
    ground.E = -Δt * ground.φ * ground.control_map' # For non-linear

    ground
end

function learn_case(case_name, part="physical", analyze_only=false)
    # Set paths.
    parent_dir = string(dirname(@__DIR__))
    dir_config = parent_dir * "/config/" * case_name

    # Set ground truth.
    ground = load_groundtruth(dir_config)

    sum_steps = [2^x for x in 6:12] # 6:14
    n_step = length(sum_steps)
    p_modes = ["Initial", "With-Clip", "With-Clip-Sieve"] # Choose the learn equation of physical part.
    C = [0.01, 0.1, 1, 10, 100] # Learn Control hyberparameter.
    dir_result = parent_dir * "/results/"
    if part == "physical"
        # Record results of physical parts.
        n_paras = 6
        all_results = zeros(n_step, length(p_modes), n_paras)
        save_dir = dir_result * "physical/"
    else
        # Record results of control parts.
        n_controller = 4
        n_rules = 2
        n_C = length(C)
        n_paras = 4
        all_results = zeros(n_step, n_controller, n_rules, n_C, n_paras)
        save_dir = dir_result * "control/"
    end
    save_path_result = save_dir * "avg.jld"
    if analyze_only == false
        run_multi_trails(part, case_name, all_results, parent_dir, ground, sum_steps, p_modes, C, save_dir)

        analyze_results(part, p_modes, C, sum_steps, save_path_result, parent_dir)
    else
        analyze_results(part, p_modes, C, sum_steps, save_path_result, parent_dir)
    end
end

end

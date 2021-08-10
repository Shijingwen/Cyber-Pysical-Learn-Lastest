module HybridAutomata
using Distributions, LinearAlgebra, Plots, DelimitedFiles

export System, update_control, update_physical, gen_system_trace
export plot_trace, start

# This mutable struct includes all of the simulation system parameters.
# To do: prevent access uninitiled parameters.
mutable struct System
    # Parameters about build structure.
    room_num::Int
    room_size::Array
    room_THVAC::Array
    room_Temp::Array
    room_adjecent::Array
    # Parameters about controllers.
    control_num::Int
    control_group_num::Array
    control_status::Array
    control_map::Array
    control_rooms::Array # control status of every room
    # The follow parameters have deafult value.
    noise_avg::Float32      # 0
    noise_var::Float32      # 0.1
    T_ext_avg::Float32      # 85
    T_ext_var::Float32      # 4.33
    THVAC_big::Float32      # 50
    THVAC_small::Float32    # 55
    T_min::Float32          # 70
    T_max::Float32          # 80
    A::Array                # Matrix of α, diagonal 0.1
    B::Array                # Matrix of β, diagonal 0.1
    Q_internal::Array       # Matrix of ω, ones 1.36
    φ::Float32              # 0.6
    # Φ::Array              # Matrix of φ, diagnal 0.6
    System() = new()        # Allow imcomplete initialization.

end

# Update control status according to the control automata.
function update_control(system_build)
    result = system_build.control_map * system_build.room_Temp
    group_avg = result./system_build.control_group_num
    if 0 in system_build.control_group_num
        replace!(group_avg, NaN=>0.0)
    end
    tmp_controls = system_build.control_status
    for (index, value) in enumerate(group_avg)
        if value >= system_build.T_max
            tmp_controls[index] = 1
        elseif value <= system_build.T_min
            tmp_controls[index] = 0
        else
            # Pass and keep the original value
        end
    end
    system_build.control_status = tmp_controls
    system_build.control_rooms = system_build.control_map' * system_build.control_status
    system_build.control_rooms, group_avg
end

# Update the physical dynamics.
# Ti(t) = α(Text(t-1) – Ti(t-1)) + β∑(Tj(t-1)−Ti(t-1)) + Q_internal + φ(THVAC(t-1)-Ti(t-1))ui(t-1)
function update_physical(system_build, t_ext, noise, Δt, sys="non-linear")
    # T_ext = t_ext * ones(system_build.room_num)
    T_ext = t_ext
    dif_external = (T_ext - system_build.room_Temp)
    dif_sum_neighbor = (system_build.room_adjecent * system_build.room_Temp)
    dif_air = (system_build.room_THVAC - system_build.room_Temp)
    # Non-linear system
    dif_Temp = Δt * (system_build.A * dif_external + system_build.B * dif_sum_neighbor + system_build.Q_internal + system_build.φ * (dif_air .* system_build.control_rooms) + noise)

    # Linear system
    #dif_air2 = -25
    #dif_Temp = Δt * (system_build.A * dif_external + system_build.B * dif_sum_neighbor + system_build.Q_internal + system_build.φ * dif_air2 * system_build.control_rooms + noise)
    # @show system_build.room_Temp + dif_Temp, system_build.room_Temp, dif_external, dif_sum_neighbor, dif_air, system_build.control_rooms
    system_build.room_Temp = system_build.room_Temp + dif_Temp
    dif_external, dif_sum_neighbor, dif_air
end

# Generate the dataset.
function gen_system_trace(system_build, Δt, sum_steps, save_trace_path)
    Y_trace = zeros(system_build.room_num, sum_steps+1)
    Ur_trace = zeros(system_build.room_num, sum_steps+1) # Control for every room
    U_trace = zeros(system_build.control_num, sum_steps+1)
    dif_external_trace = zeros(system_build.room_num, sum_steps+1)
    dif_sum_neighbor_trace = zeros(system_build.room_num, sum_steps+1)
    dif_air_u_trace  = zeros(system_build.room_num, sum_steps+1)

    Y_trace[:, 1] = system_build.room_Temp
    Ur_trace[:, 1] = system_build.control_rooms
    U_trace[:, 1] = system_build.control_status
    dif_external_trace[:, 1] = zeros(system_build.room_num, 1)
    dif_sum_neighbor_trace[:, 1] = zeros(system_build.room_num, 1)
    dif_air_u_trace[:, 1] = zeros(system_build.room_num, 1)
    T_ext_distribution = Normal(system_build.T_ext_avg, system_build.T_ext_var)
    T_ext_trace = rand(T_ext_distribution, (system_build.room_num, sum_steps+1))
    Noise_distribution = Normal(system_build.noise_avg, system_build.noise_var)
    Noise_trace = rand(Noise_distribution, (system_build.room_num, sum_steps+1))
    for i in 1:sum_steps
        t_ext = T_ext_trace[:, i]
        noise = Noise_trace[:, i]
        update_control(system_build)
        dif_external, dif_sum_neighbor, dif_air_u = update_physical(system_build, t_ext, noise, Δt)
        Ur_trace[:, i+1] = system_build.control_rooms
        Y_trace[:, i+1] = system_build.room_Temp
        U_trace[:, i+1] = system_build.control_status
        dif_external_trace[:, i+1] = dif_external
        dif_sum_neighbor_trace[:, i+1] = dif_sum_neighbor
        dif_air_u_trace[:, i+1] = dif_air_u
    end

    save_trace(Y_trace, Ur_trace, U_trace, T_ext_trace, dif_external_trace,
    dif_sum_neighbor_trace, dif_air_u_trace, Noise_trace, save_trace_path)
    Y_trace, Ur_trace
end

# Save the trace.
function save_trace(Y_trace, Ur_trace, U_trace, T_ext_trace, dif_external_trace,
    dif_sum_neighbor_trace, dif_air_trace, Noise_trace, save_trace_path)
    writedlm(save_trace_path * "trace_Y.csv",  Y_trace, ',')
    writedlm(save_trace_path * "trace_Ur.csv",  Ur_trace, ',')
    writedlm(save_trace_path * "trace_U.csv",  U_trace, ',')
    writedlm(save_trace_path * "trace_T_ext.csv",  T_ext_trace, ',')
    writedlm(save_trace_path * "trace_dif_external.csv",  dif_external_trace, ',')
    writedlm(save_trace_path * "trace_dif_neighbor.csv", dif_sum_neighbor_trace, ',')
    writedlm(save_trace_path * "trace_dif_air.csv", dif_air_trace, ',')
    writedlm(save_trace_path * "trace_noise.csv", Noise_trace, ',')
end

# Extract specific rooms from the trace.
function extract_plot_trace(Y_trace, Ur_trace, room_indexs)
    time_step = size(Y_trace)[2]
    plot_Y_trace = zeros(length(room_indexs), time_step)
    plot_U_trace = zeros(length(room_indexs), time_step)
    for (index, room_id) in enumerate(room_indexs)
        plot_Y_trace[index, :] =  Y_trace[room_id, :]
        plot_U_trace[index, :] = Ur_trace[room_id, :]
    end
    plot_Y_trace, plot_U_trace
end

# Generate the dataset.
function plot_trace(Y_trace, Ur_trace, Δt, sum_steps, room_indexs, save_fig_path)
    plot_Y_trace, plot_U_trace = extract_plot_trace(Y_trace, Ur_trace, room_indexs)
    p1 = plot(plot_Y_trace[:, end-100:end]', label=["Room1" "Room2" "Room11" "Room12"],
             xlabel="Time(Δt=0.1 hour)", ylabel="Temperature(F)", linestyle=[:dash :dash :solid :solid],
             color=["#6baed6" "#2171b5" "#74c476" "#238b45"])
    p2 = plot(plot_U_trace[:, end-100:end]', label=["Room1" "Room2" "Room11" "Room12"],
            xlabel="Time(Δt=0.1 hour)", ylabel="AC Controller", linestyle=[:dash :dash :solid :solid],
            color=["#6baed6" "#2171b5" "#74c476" "#238b45"])
    p = plot(p1, p2, layout = (2, 1), legend=true)
    if save_fig_path == ""
        display(p)
    else
        savefig(p, save_fig_path * "plot_temp_control.pdf")
    end
end


# Start from this function.
function start(system_build, Δt, sum_steps,
               plot_room_indexs, save_fig_path, save_trace_path)
    Y_trace, Ur_trace = gen_system_trace(system_build, Δt, sum_steps, save_trace_path)
    plot_trace(Y_trace, Ur_trace, Δt, sum_steps, plot_room_indexs, save_fig_path)
end

end # End module HybridAutomata

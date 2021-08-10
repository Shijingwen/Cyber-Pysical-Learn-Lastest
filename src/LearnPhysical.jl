module OLS

using JuMP, Ipopt, LinearAlgebra, Random, Distributions, Random
export format_ols, ols

# Generate correct format for OLS estimation.
function format_ols(Y, U, Ur, T_ext, noise=[])
    # Parameters at the right of formula.
    Y_in = Y[:, 1:end-1]
    T_ext_in = T_ext[:, 1:end-1]
    noise_in = noise[:, 1:end-1]
    U_in = U[:, 2:end]  # u(t) is first updated with y(t-1) and then used to calculate y(t)
    Ur_in = Ur[:, 2:end]
    # Parameters at the left of formula.
    Y_out = Y[:, 2:end]
    return_paras = [Y_out, Y_in, U_in, T_ext_in, noise_in]
    return_paras
end

# Conduct OLS estimation.
function ols(mode, trace_paras, Δt, room_THVAC, control_map, show_detail=false)
    Y_out = trace_paras[1]
    Y_in = trace_paras[2]
    U_in = trace_paras[3]
    # Ur_in = trace_paras[4]
    # Ext_in = ones(12, 1) * trace_paras[4]
    Ext_in = trace_paras[4]

    if mode == "split"
        println("Split")
        paras, names = train_split(Y_out, Y_in, U_in, Ext_in, room_THVAC, control_map, show_detail)
    else
        println("Combine")
        paras, names = train_one_round(mode, Y_out, Y_in, U_in, Ext_in, room_THVAC, control_map, show_detail)
    end
    paras, names
end

function train_split(Y_out, Y_in, U_in, Ext_in, room_THVAC, control_map, show_detail)
    n_y, n_t = size(Y_out)
    n_u = size(U_in)[1]
    combine_Â_T = zeros(n_y, n_y)
    combine_B̂_ext = zeros(n_y, n_y)
    combine_Ĉ_U = zeros(n_y, n_u)
    combine_D̂_U = zeros(n_y, n_u)
    combine_Ê_U = zeros(n_y, n_u)
    combine_Q = zeros(n_y)

    avg_train_len = 0 # The average value of .
    for i in 1:n_u
        tmp_u = U_in[i, :]
        target = 1 # Fix u as 1
        select_index = findall(x->x==target, tmp_u)
        avg_train_len += length(select_index)
        paras, names = train_one_round(Y_out[:, select_index], Y_in[:, select_index],
        U_in[i, select_index]', Ur_in[:, select_index], Ext_in[:, select_index], room_THVAC, show_detail)
        combine_Â_T = paras[1]
        combine_B̂_ext = paras[2]
        combine_Ĉ_U[:, i] =  paras[3]
        combine_D̂_U[:, i] =  paras[4]
        combine_Ê_U[:, i] =  paras[5]
        combine_Q = paras[6]
    end
    avg_train_len = avg_train_len/n_u
    paras = [combine_Â_T, combine_B̂_ext, combine_Ĉ_U, combine_D̂_U, combine_Ê_U, combine_Q]
paras, names
end

function train_one_round(mode, Y_out, Y_in, U_in, Ext_in, room_THVAC, control_map, show_detail)
    n_y, n_t = size(Y_out)
    n_u = size(U_in)[1]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0) # Comment out this line if you want to see tarining details.

    @variable(model, Â_T[1:n_y, 1:n_y], start=0, lower_bound=0) # rand(Float64, (1:n_y, 1:n_y) lower_bound=0, upper_bound=1
    @variable(model, B̂_ext[1:n_y, 1:n_y], start=0, lower_bound=0)
    @variable(model, Q[1:n_y])

    if mode == "linear"
        # Linear system
        @variable(model, Ĉ_U[1:n_y, 1:n_u], start=1, upper_bound=0)
        D̂_HVAC_U = zeros(n_y, n_u)
        Ê_T_U = zeros(n_y, n_u)
        @constraint(model, Z .== (Y_out - (Â_T * Y_in +  B̂_ext * Ext_in + Ĉ_U * U_in + Q*ones(1, n_t))))
        @objective(model, Min, sum(Z.^ 2))
    else
        # Non-linear system
        @variable(model, Ĉ_U[1:n_y, 1:n_u], start=0)
        @variable(model, Ê_T_U[1:n_y, 1:n_u], start = 0)
        @variable(model, Z[1:n_y, 1:n_t])

        if mode == "Initial"
            @variable(model, D̂_HVAC_U[1:n_y, 1:n_u], start = 0)
            @constraint(model, Z .== (Y_out - (Â_T * Y_in + B̂_ext * (Ext_in) + Ĉ_U * U_in + Q*ones(1, n_t) + (room_THVAC * ones(1, n_t)).* (D̂_HVAC_U * U_in) + Y_in.* (Ê_T_U * U_in))))
        elseif mode == "With-Clip"
            @variable(model, D̂_HVAC_U[1:n_y, 1:n_u], start = 0)
            @constraint(model, Z .== (Y_out - (Â_T * Y_in + B̂_ext * (Ext_in) + Q * ones(1, n_t) + (room_THVAC * ones(1, n_t) - Y_in).* (D̂_HVAC_U * U_in))))
        elseif mode == "With-Clip-Sieve"
            @variable(model, D̂_HVAC_U, start = 0) # know the controller-room map
            @constraint(model, Z .== (Y_out - (Â_T * Y_in + B̂_ext * (Ext_in) + Q * ones(1, n_t) + (room_THVAC * ones(1, n_t) - Y_in).* (D̂_HVAC_U * control_map * U_in))))
        else
            println("Not defined mode")
        end
    end
    @objective(model, Min, sum(Z.^ 2))
    optimize!(model)
    paras = [value.(Â_T), value.(B̂_ext), value.(Ĉ_U), value.(D̂_HVAC_U), value.(Ê_T_U), value.(Q)]
    names = ["Â_T", "B̂_ext", "Ĉ_U", "D̂_HVAC_U", "Ê_T_U", "Q̃"]
    paras, names
end

end

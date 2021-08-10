module Control

include("SVM2.jl")
using .SVM2
include("SVM.jl")
using .SVM

using StatsBase, DataFrames
import StatsBase: predict # Import to override

using Printf, LinearAlgebra, Random, Distributions
using JuMP, Ipopt
# using LIBSVM

export load_control_format, start_train_svm

function load_control_format(sample_num, Y, U)
    # names = [:u1_0; :u2_0; :u3_0; :u4_0; :u1_1; :u2_1; :u3_1; :u4_1; :T_0; :du0]
    m = size(Y)[1] # number of rooms
    n = size(U)[1] # number of controller
    data = zeros(m+n*2+1, sample_num)
    @show sample_num, size(Y), size(U), size(data)
    data[1, :] = 0.1 * collect(1:sample_num) #Time
    data[2:5, :] = U[:, 1:end-1] #U0
    data[6:9, :] = U[:, 2:end] #U1
    data[10:21, :] = Y[:, 1:end-1] #Y0

    df_data = DataFrame(data', :auto)
    df_data
end

function start_train_svm(data, C, beta_true, v_true)
    results = zeros(4, 2, length(C), 4)
    for controller in 1:4
        println("Learning Controller: ", controller)
        for u_0 in 0:1
                @show controller, u_0
                X, Y = choose_rules(deepcopy(data), controller, u_0)
                errors = train_svm(X, Y, beta_true[controller, :], v_true[controller, u_0+1], C, 0.8)
                results[controller, u_0+1, :, 1] = errors[1]
                results[controller, u_0+1, :, 2] = errors[2]
                results[controller, u_0+1, :, 3] = errors[3]
                results[controller, u_0+1, :, 4] = errors[4]
        end
    end
    results
end

function choose_rules(data, controller, u_0)
    index_u0 = controller +  1
    index_u1 = 4 + controller + 1
    println("Total: ")
    @show size(data)
    data = filter(row -> row[index_u0] == u_0, data)
    println("After: ")
    @show size(data)

    transit = 1 - u_0
    # Throw away 5% initial points
    index_start = max(round(Int, size(data, 1) * 0.05), 2)
    data = data[index_start:end, :]

    # Use all of the rooms
    X = data[!, 10:21]
    X = Matrix(X)
    if u_0 == 0
        Y = [u1 == transit ? 1.0 : -1.0 for u1 in data[!, index_u1]]
    else
        Y = [u1 == transit ? -1.0 : 1.0 for u1 in data[!, index_u1]]
    end

    X, Y
end

function train_svm(X, Y, beta_true, v_true, C, train_percentage=0.8)
    split_point = round(Int, size(X)[1] * train_percentage)
    X_train = X[1:split_point, :]
    Y_train = Y[1:split_point]
    X_test = X[split_point+1: end, :]
    Y_test = Y[split_point+1: end]
    n_sample, n_feature = size(X_train)

    v_error = SVM2.svmtrain2(X_train, Y_train, X_test, Y_test, beta_true, v_true, C)
    v_error
end

end

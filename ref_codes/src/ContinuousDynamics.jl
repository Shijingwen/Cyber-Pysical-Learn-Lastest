using Distributions, Random, LinearAlgebra

ẋ(t; x, u, A_d, ϵ) = A_d * [x; u] .+ ϵ
x̂(t; t_0, x_0, u_0, A_d, ϵ) = x_0 .+ ẋ(t_0; x=x_0, u=u_0, A_d=A_d, ϵ=ϵ) * (t - t_0)

function xhat(t; t_0, x_0, u_0, A_d, ϵ)
    return x_0 + xdot(t_0; x=x_0, u=u_0, A_d=A_d, ϵ=ϵ) * (t - t_0)
end


function xdot(t; x, u, A_d, ϵ)
    @show x, u, A_d, ϵ
    @show res = A_d * [x; u] .+ ϵ
    return A_d * [x; u] .+ ϵ
end


# u .* w_u is the switch-induced bias
# todo: natural bias (needed or not?)
function simulate(Δt, num_t; u=[1], w_u=[0.25])
    σ = 1
    rng = MersenneTwister(123)

    p_ϵ = Normal(0, σ^2)

    w_x = [-0.5]
    #w_u = [0.25]
    x_ini = [10]

    t_ini = 1

    t_end = t_ini + (num_t - 1)

    @show A_d = [w_x; w_u]' 

    m = num_t
    n_x = 1
    n_u = 1

    X = Matrix{Float32}(undef, m, n_x)
    U = Matrix{Float32}(undef, m, n_u)

    X[1, :] = x_ini
    U[1, :] = u
    for (i, t) in enumerate(t_ini : Δt : t_end - Δt)
        X[i + 1, :] = x̂(t + Δt; t_0=t, x_0=X[i, :], u_0=U[i, :], A_d=A_d, ϵ=rand(rng, p_ϵ, n_x))
        U[i + 1, :] = u
    end

    return X, U
end

function estimate_parameter(X, U, Δt)
    # (x_{t+1} - x_t) / Δt ≈ A_d * x_t + ϵ
    # x_{t+1} ≈ (A_d .* Δt + I) * x_t + Δt .* ϵ
    # x_{t+1} = Ax_t + Bϵ where A = A_d .* Δt + I and B = Δt
    m, n_x = size(X)
    n_u = size(U, 2)
    n = n_x + n_u
    Σ_0, Σ_1 = zeros(n, n), zeros(n, n)
    for i in 1:m-1
        z_0 = [X[i, :]; U[i, :]]
        z_1 = [X[i+1, :]; U[i+1, :]]
        Σ_0 += z_0 * z_0'
        Σ_1 += z_1 * z_0'
    end

    Σ_0 = Σ_0 ./ (m - 1)
    Σ_1 = Σ_1 ./ (m - 1)

    @show Σ_0, Σ_1

    @show Â = Σ_1 / Σ_0
    @show Â_d = (Â - I) ./ Δt
    B = Δt

    return (Â_d, B)
end


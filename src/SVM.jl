module SVM

using StatsBase
import StatsBase: predict # Import to override

using Printf, LinearAlgebra, Random, Distributions
using JuMP, Ipopt

export svm, predict, cost, accuracy, pegasos

struct SVMFit
    beta::Vector{Float64}
    v::Float64
    λ::Float32
    passes::Union{Int32, Nothing}
    converged::Union{Bool, Nothing}
    optimal::Union{Bool, Nothing}
end

function predict(fit::SVMFit, X::AbstractMatrix{<:Real})
    n, m = size(X)
    preds = fill(0, m)
    for i in 1:m
        dot_prod = 0.0
        for j in 1:n
            dot_prod += fit.w[j] * X[j, i]
        end
        preds[i] = sign(dot_prod)
    end
    return preds
end

@enum Optimizer begin
    PegasosBatch
    Pegasos
    CDDual
    InteriorPoint
end

function svm(X::AbstractMatrix{<:Real},
             Y::AbstractVector{<:Real};
             optimizer::Optimizer,
             ϵ::Real = 1e-6,
             max_passes::Integer = 1000)
    (n,m) = size(X)
    C = [10, 1, 0.1, 0.01, 0.0]
    for c in C
        println("~~~~~~~~~~~~~~~~~~~~~~~")
        @show λ = c/sqrt(m)
        svmFit = undef
        if optimizer == PegasosBatch
            (beta, v, t_converge) = pegasos_batch(X, Y, λ = λ, maxpasses = max_passes)
            svmFit = SVMFit(beta, v, Float32(λ), t_converge, t_converge < Inf, nothing)
        elseif optimizer == Pegasos
            (beta, v, t_converge) = pegasos(X, Y, λ = λ, maxpasses = max_passes)
            svmFit = SVMFit(beta, v, Float32(λ), t_converge, t_converge < Inf, nothing)
        elseif optimizer == CDDual
            (w, t_converge) = cddual(X, Y, norm = 1, maxpasses = max_passes)
            svmFit = SVMFit(w, Float32(λ), t_converge, t_converge < Inf, nothing)
        elseif optimizer == InteriorPoint
            (beta, v, optimal) = interiorPoint(X, Y; λ = λ)
            svmFit = SVMFit(beta, v, Float32(λ), nothing, nothing, optimal)
        else
            error("unsupported optimizer type: $(optimizer)")
        end
        @show svmFit.beta
        @show svmFit.v
    end
end

function cost(fit::SVMFit,
              X::AbstractMatrix{<:Real},
              Y::AbstractVector{<:Real})
    beta = fit.beta
    v = fit.v
    λ = fit.λ
    n, m = size(X)
    # risk = λ / 2 * norm(beta, 2)^2
    for i in 1:m
        p = 0.0
        l(beta, v, x, y) = max(0, 1 - y * (beta ⋅ x + v))
        risk = l(beta, v, X[:, i], Y[i])
    end
    return risk / m + λ * norm(beta,2)
end


function accuracy(fit::SVMFit,
                  X::AbstractMatrix{<:Real},
                  Y::AbstractVector{<:Real})
    n, m = size(X)
    return count(predict(fit, X) .== Y) / m
end


function interiorPoint(X::AbstractMatrix{<:Real},
                       Y::AbstractVector{<:Real};
                       λ::Real = 0.1)
    n, m = size(X)

    model = Model(with_optimizer(Ipopt.Optimizer))
    set_optimizer_attribute(model, "print_level", 0) # Comment out this line if you want to see tarining details.
    @variable(model, beta[1:n])
    @variable(model, v)
    @objective(model, Min, 1/2 * (beta ⋅ beta)) # l2 norm = beta ⋅ beta
    for i in 1:m
        @constraint(model, Y[i] * (beta ⋅ X[:, i] + v) ≥ 1)
    end

    optimize!(model)

    beta_sol, optimal = undef, undef # hack, indication of solution status
    if termination_status(model) == MOI.OPTIMAL
        beta_sol = value.(beta)
        v_sol = value.(v)
        optimal = true
    elseif has_values(model)
        beta_sol = value.(beta)
        v_sol = value.(v)
        optimal = false
    #else
        #warn("The model was not solved correctly.")
    end

    return (beta_sol, v_sol, optimal)
end



function pegasos(X::AbstractMatrix{<:Real},
                 Y::AbstractVector{<:Real};
                 λ::Real = 0.1,
                 ϵ::Real = 1e-3,
                 maxpasses ::Integer = 100,
                 seed::Integer = 123)
    rng = MersenneTwister(seed)

    n, m = size(X)

    # Initialize weights to 0
    beta = zeros(n)
    v = 0
    I = DiscreteUniform(1, m)

    t_converge = maxpasses

    # Iterations
    for t in 1:maxpasses
        i = rand(rng, I)
        η = 1 / (λ * t)
        (x_i, y_i) = (X[:, i], Y[i])
        p = y_i * (beta ⋅ x_i + v)
        ∇_beta = nothing
        if p ≥ 1
           ∇_beta = λ * beta
        else # p < 1
           ∇_beta = λ * beta - y_i * x_i
        end
        beta = beta - η * ∇_beta
        if t % 10 == 0
            @show norm(η * ∇_beta, 2)
            @show cost(SVMFit(beta, v, Float32(λ), t, t_converge < Inf, nothing), X, Y)
            @show beta
            @show v
        end

       if norm(η * ∇_beta, 2) < ϵ
           t_converge = t
           break
       end
    end

    return (beta, v, t_converge)
end

# S is X,Y
# T is maxpasses
# p: # of features
# n: # of data points
# k: size of minibatch
function pegasos_batch(X::AbstractMatrix{<:Real},
                          Y::AbstractVector{<:Real};
                          k::Integer = 5,
                          λ::Real = 0.1,
                          maxpasses::Integer = 100)
    # p features, n observations
    p, n = size(X)

    # Initialize weights so norm(w) <= 1 / sqrt(lambda)
    beta = randn(p)
    v = 0
    sqrtlambda = sqrt(lambda)
    normalizer = sqrtlambda * norm(beta)
    for j in 1:p
        beta[j] /= normalizer
    end

    # Allocate storage for repeated used arrays
    deltaw = Array{Float64,1}(undef, p)
    beta_tmp = Array{Float64,1}(undef, p)
    v_tmp = undef

    # Loop
    for t in 1:maxpasses
        # Calculate stepsize parameters
        alpha = 1.0 / t
        eta_t = 1.0 / (λ * t)

        # Calculate scaled sum over misclassified examples
        # Subgradient over minibatch of size k
        fill!(deltaw, 0.0)
        for i in 1:k
            # Select a random item from X
            # This is one element of At of S
            index = rand(1:n)

            # Test if prediction isn't sufficiently good
            # If so, current item is element of At+
            pred = Y[index] * dot(beta, X[:, index])
            if pred < 1.0
                # Update subgradient
                for j in 1:p
                    deltaw[j] += Y[index] * X[j, index]
                end
            end
        end

        # Rescale subgradient
        for j in 1:p
            deltaw[j] *= (eta_t / k)
        end

        # Calculate tentative weight-update
        for j in 1:p
            beta_tmp[j] = (1.0 - alpha) * beta[j] + deltaw[j]
        end

        # Find projection of weights into L2 ball
        proj = min(1.0, 1.0 / (sqrtlambda * norm(beta_tmp)))
        for j in 1:p
            beta[j] = proj * beta_tmp[j]
        end
    end

    return (beta, v, maxpasses)
end

# Randomization option slows down processing
# but improves quality of solution considerably
# Would be better to do randomization in place
function cddual(X::AbstractMatrix{<:Real},
                Y::AbstractVector{<:Real};
                C::Real = 1.0,
                norm::Integer = 2,
                randomized::Bool = true,
                maxpasses::Integer = 2)
    # l: # of samples
    # n: # of features
    n, l = size(X)
    alpha = zeros(Float64, l)
    w = zeros(Float64, n)

    # Set U and D
    #  * L1-SVM: U = C, D[i] = 0
    #  * L2-SVM: U = Inf, D[i] = 1 / (2C)
    U = 0.0
    if norm == 1
        U = C
        D = zeros(Float64, l)
    elseif norm == 2
        U = Inf
        D = fill(1.0 / (2.0 * C), l)
    else
        throw(ArgumentError("Only L1-SVM and L2-SVM are supported"))
    end

    # Set Qbar
    Qbar = Array{Float64,1}(undef, l)
    for i in 1:l
        Qbar[i] = D[i] + dot(X[:, i], X[:, i])
    end

    # Loop over examples
    converged = false
    pass = 0

    while !converged
        # Assess convergence
        pass += 1
        if pass == maxpasses
            converged = true
        end

        # Choose order of observations to process
        if randomized
            indices = randperm(l)
        else
            indices = 1:l
        end

        # Process all observations
        for i in indices
            g = Y[i] * dot(w, X[:, i]) - 1.0 + D[i] * alpha[i]

            if alpha[i] == 0.0
                pg = min(g, 0.0)
            elseif alpha[i] == U
                pg = max(g, 0.0)
            else
                pg = g
            end

            if abs(pg) > 0.0
                alphabar = alpha[i]
                alpha[i] = min(max(alpha[i] - g / Qbar[i], 0.0), U)
                for j in 1:n
                    w[j] = w[j] + (alpha[i] - alphabar) * Y[i] * X[j, i]
                end
            end
        end
    end

    return (w, pass)
end


end # module SVM

module SVM

using StatsBase
import StatsBase: predict # Import to override

using Printf, LinearAlgebra, Random, Distributions
using JuMP, Ipopt

export svm, predict, cost, accuracy, pegasos

struct SVMFit
    w::Vector{Float64}
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
    CDDual
    InteriorPoint
end

function svm(X::AbstractMatrix{<:Real},
             Y::AbstractVector{<:Real};
             optimizer::Optimizer,
             λ::Real = 0.1,
             ϵ::Real = 1e-6,
             max_passes::Integer = 100)
    svmFit = undef
    if optimizer == PegasosBatch
        (w, t_converge) = pegasos_batch(X, Y, lambda = λ, maxpasses = max_passes)
        svmFit = SVMFit(w, Float32(λ), t_converge, t_converge < Inf, nothing)
    elseif optimizer == CDDual
        (w, t_converge) = cddual(X, Y, norm = 1, maxpasses = max_passes)
        svmFit = SVMFit(w, Float32(λ), t_converge, t_converge < Inf, nothing)
    elseif optimizer == InteriorPoint
        (w, optimal) = interiorPoint(X, Y; λ = λ)
        svmFit = SVMFit(w, Float32(λ), nothing, nothing, optimal)
    else
        error("unsupported optimizer type: $(optimizer)")
    end

    return svmFit
end

function cost(fit::SVMFit,
              X::AbstractMatrix{<:Real},
              Y::AbstractVector{<:Real})
    w = fit.w
    λ = fit.λ
    n, m = size(X)
    risk = λ / 2 * norm(w, 2)^2
    for i in 1:m
        p = 0.0
        l(w, x, y) = max(0, 1 - y * (w ⋅ x))
        risk += l(w, X[:, i], Y[i])
    end
    return risk / m
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
    @variable(model, w[1:n])
    @objective(model, Min, w ⋅ w)
    for i in 1:m
        @constraint(model, Y[i] * (w ⋅ X[:, i]) ≥ 1)
    end

    optimize!(model)

    w_sol, optimal = undef, undef # hack, indication of solution status
    if termination_status(model) == MOI.OPTIMAL
        w_sol = value.(w)
        optimal = true
    elseif has_values(model)
        w_sol = value.(w)
        optimal = false
    #else
        #warn("The model was not solved correctly.")
    end

    return (w_sol, optimal)
end



function pegasos(X::AbstractMatrix{<:Real},
                 Y::AbstractVector{<:Real};
                 λ::Real = 0.1,
                 ϵ::Real = 1e-3,
                 T::Integer = 100,
                 seed::Integer = 123)
    rng = MersenneTwister(seed)

    n, m = size(X)

    # Initialize weights to 0
    w = zeros(n)
    I = DiscreteUniform(1, m)

    t_converge = Inf

    # Iterations
    for t in 1:T
        i = rand(rng, I)
        η = 1 / (λ * t)
        (x_i, y_i) = (X[:, i], Y[i])
        p = y_i * (w ⋅ x_i)
        ∇_w = nothing
        if p ≥ 1
           ∇_w = λ * w
        else # p < 1
           ∇_w = λ * w - y_i * x_i
        end
        w = w - η * ∇_w
        if t % 10 == 0
            @show norm(η * ∇_w, 2)
            @show cost(SVMFit(w, λ, T, t_converge < Inf), X, Y)
        end

#        if norm(η * ∇_w, 2) < ϵ
#            t_converge = t
#            break
#        end
    end

    return (w, t_converge)
end

# S is X,Y
# T is maxpasses
# p: # of features
# n: # of data points
# k: size of minibatch
function pegasos_batch(X::AbstractMatrix{<:Real},
                          Y::AbstractVector{<:Real};
                          k::Integer = 5,
                          lambda::Real = 0.1,
                          maxpasses::Integer = 100)
    # p features, n observations
    p, n = size(X)

    # Initialize weights so norm(w) <= 1 / sqrt(lambda)
    w = randn(p)
    sqrtlambda = sqrt(lambda)
    normalizer = sqrtlambda * norm(w)
    for j in 1:p
        w[j] /= normalizer
    end

    # Allocate storage for repeated used arrays
    deltaw = Array{Float64,1}(undef, p)
    w_tmp = Array{Float64,1}(undef, p)

    # Loop
    for t in 1:maxpasses
        # Calculate stepsize parameters
        alpha = 1.0 / t
        eta_t = 1.0 / (lambda * t)

        # Calculate scaled sum over misclassified examples
        # Subgradient over minibatch of size k
        fill!(deltaw, 0.0)
        for i in 1:k
            # Select a random item from X
            # This is one element of At of S
            index = rand(1:n)

            # Test if prediction isn't sufficiently good
            # If so, current item is element of At+
            pred = Y[index] * dot(w, X[:, index])
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
            w_tmp[j] = (1.0 - alpha) * w[j] + deltaw[j]
        end

        # Find projection of weights into L2 ball
        proj = min(1.0, 1.0 / (sqrtlambda * norm(w_tmp)))
        for j in 1:p
            w[j] = proj * w_tmp[j]
        end
    end

    return (w, maxpasses)
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

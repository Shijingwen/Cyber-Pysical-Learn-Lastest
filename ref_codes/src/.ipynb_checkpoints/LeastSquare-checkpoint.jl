module LeastSquare

using Distributions, Random, LinearAlgebra

export leastSquareEstimate

# model: y = w_1 * x_1 + b + ϵ where ϵ ∼ N(0, σ^2)

σ = 1

rng = MersenneTwister(1234)

m = 100
n = 2

p_x = Uniform(1, 10)
p_ϵ = Normal(0, σ^2)

b = 3
w = [2; b]
X = hcat(rand(rng, p_x, m, 1), ones(m, 1))
ϵ = rand(rng, p_ϵ, m)
Y = X * w + ϵ


# X'Y = X'Xw

function leastSquareEstimate(X, Y)
    return ŵ = (X' * X) \ (X' * Y)
end

leastSquareEstimate(X, Y)

end # LeastSquare

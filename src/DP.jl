module DP
using Distributions

function Guassian(ϵ, δ)
    σ = √(2*log(1.25/δ))/ϵ
    @show σ
    noise_distribution = Normal(0, σ)
    noise_trace = rand(noise_distribution, 10)
    @show noise_trace
    noise_trace
end

function BoundSensitive(value, C=1)
    clipped_value = value/max(1, abs(value))
    @show clipped_value
    clipped_value
end

# Guassian(1.2, 1e-5)
BoundSensitive(5, 1)
BoundSensitive(0.2, 1)
end

module RoomTemperatureLearn

using StatsBase, Random, LinearAlgebra, Gadfly, Distributions

include("RoomTemperature.jl")
using .RoomTemperature

include("SVM.jl")
using .SVM

t_max = 100 # seconds
data = RoomTemperature.simulate(t_max)

# learn rules to turn on heater
u_0_condition = 0
if u_0_condition != undef
    filter!(row -> row[:u_0] == u_0_condition, data)
end

# plot a window of data
#plot(data[data[!, :u_0] .== 0, :], x=:t, y=:T_0, color=:du)

# Learn the joint of heater state and room temperature
rng = MersenneTwister(1234)

du_on = 1.0

# make the points separable
data_transit = filter(row -> row[:du] == du_on, data)
@show size(data_transit)
@show T_transit_max, T_transit_min = maximum(data_transit[!, :T]), minimum(data_transit[!, :T])
within_transit_noise = (T_transit_min - 0.5 .≤ data[!, :T_0]) .& (data[!, :T_0] .≤ T_transit_max + 0.5)
# todo: softmargin.  stop-gap: make points separable
data[within_transit_noise, :du] .= du_on


# get rid of initial points
data = data[round(Int, size(data, 1) * 0.1):end, :]

X = transpose(convert(Matrix, data[!, [:T]]))
Y = [du == du_on ? 1.0 : -1.0 for du in data[!, :du]]
n, m = size(X)

# What if we add some other unrelated measurements?
T_ind = rand(rng, Normal(10, 5), m)
X = vcat(X, T_ind')

# What if we add some other related measurements?
T = X[1, :]
T_dep = 2T .- 1
X = vcat(X, T_dep')

# Add bias term 1 for each example
X = vcat(X, ones(m)')
n, m = size(X)

# bad: this probably leaves out those rare transition points!
#train = bitrand(rng, m)
#test = .!train
train = ones(m) .== 1
test = train

#λ = 0.01
#max_passes = 10000
#@show model = svm(X[:,train], Y[train]; optimizer=SVM.CDDual, max_passes = max_passes, λ = λ)
@show model = svm(X[:,train], Y[train]; optimizer=SVM.InteriorPoint)

@show train_cost = cost(model, X[:,train], Y[train])

@show train_accuracy = accuracy(model, X[:,train], Y[train])
@show test_accuracy = accuracy(model, X[:,test], Y[test])

# more info
# show an example of a training example...
@show u_0_condition
@show X[:,train][:, 1]
@show size(X)

@show model.w
@show model.λ

# wx + b = 0 ⟹ [w; b] ⋅ [x; 1] = 0 is the boundary.  Here we make w := [w; b]
@show -model.w[2] / model.w[1]

# show size of filtered data
@show size(data)

end # RoomTemperaturerLearn

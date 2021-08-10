module RoomTemperature

using DataFrames, Gadfly

# Euler's method for 1st order approximation
L(x; dy, x_0, y_0) = y_0 + dy(x_0, y_0) * (x - x_0)

# Room temperature dynamics with heater
# t: Time
# T: Temerature
# a: decay factor
# T_Δ: heater capacity
# u: heater switch; 0 for OFF and 1 for ON
# OFF: dT = a * (T - T_ext)
# ON: dT = a * (T - T_ext + T_Δ / a)
dT(t, T; u, T_Δ, T_ext, a) = a * T + T_Δ * u - a * T_ext

# Transition function for heater thermostat automaton
function δ(u, T; T_min, T_max)
    if T < T_min
        return 1
    elseif T > T_max
        return 0
    else
        return u
    end
end


# Approxiamtion for room temperature given initial condition
# Outputs temperature T and control u at time t, given previous state.
function system(t; u_0, t_0, T_0, T_min, T_max, T_Δ, T_ext, a)
    # physical at time t
    T_t = L(t; dy = (t_0, T_0) -> dT(t_0, T_0; u = u_0, T_Δ=T_Δ, T_ext=T_ext, a=a), x_0 = t_0, y_0 = T_0)
    # cyber at time t
    u_t = δ(u_0, T_t; T_min=T_min, T_max=T_max)
    return (u_0, T_0, u_t, T_t)
end


function simulate(t_max; Δt = 1e-1, T_min=19.0, T_max=21.0, T_Δ=5, T_ext=17, a=-1)
    u_ini, t_ini, T_ini = 0, 0.0, 20.0

    u = u_ini
    T = T_ini

    names = [:t; :u_0; :T_0; :u; :T; :du]
    n = size(names, 1)
    m = size(t_ini:Δt:t_max-Δt, 1)
    R = Matrix{Float32}(undef, m, n)

    # initial condition
    #R[1, :] = [t_ini, u_ini, T_ini, 0]

    # simulate from t_ini to t_max for each time step Δt
    for (i, t) in enumerate(t_ini:Δt:t_max-Δt)
        (u_0, T_0, u, T) = system(t + Δt; u_0=u, t_0=t, T_0=T, T_min=T_min, T_max=T_max, T_Δ=T_Δ, T_ext=T_ext, a=a)
        R[i, :] = [t u_0 T_0 u T u - u_0]
    end

    return DataFrame(R, names)
end

function plotSample(t_max; Δt=1e-1, T_min=19.0, T_max=21.0, T_Δ=5, T_ext=17, a=-1)
    df = simulate(t_max; Δt=Δt, T_min=T_min, T_max=T_max, T_Δ=T_Δ, T_ext=T_ext, a=a)
    df[!, :u] = df[!, :u] .== 0
    plot(df, x=:t, y=:T, color=:u,
         Geom.point,
         Guide.title("Temperature Trajectory of Thermostat"),
         Guide.xlabel("Time"),
         Guide.ylabel("Temperature"),
         Guide.colorkey(title="Heater", labels=["Off", "On"]))
end


end # module RoomTemperature

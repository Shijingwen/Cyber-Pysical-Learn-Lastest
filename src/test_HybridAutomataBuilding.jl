using Test, LinearAlgebra

include("HybridAutomataBuilding.jl")
using .HybridAutomata

# Test update_control - two rooms, two controls.
function test_update_control_case1()
    system_build = HybridAutomata.System()
    system_build.room_Temp = [10, 30]
    system_build.control_map = [1 0; 0 1]
    system_build.control_group_num = [1, 1]
    system_build.control_status = [1, 0]
    system_build.T_min = 15
    system_build.T_max = 20
    @testset verbose = true "test_update_control_case1" begin
        @testset "Independent control" begin
            control_rooms, group_avg =
                HybridAutomata.update_control(system_build)
            @test control_rooms == [0, 1]
            @test group_avg == [10, 30]
            system_build.room_Temp = [16, 17]
            system_build.control_status = [1, 0]
            control_rooms, group_avg =
                HybridAutomata.update_control(system_build)
            @test control_rooms == [1, 0]
            @test group_avg == [16, 17]
        end
        @testset "Dependent control" begin
            system_build.room_Temp = [15, 30]
            system_build.control_map = [1 1; 0 0]
            system_build.control_group_num = [2, 0]
            system_build.control_status = [0, 1]
            control_rooms, group_avg =
                HybridAutomata.update_control(system_build)
            @test control_rooms == [1, 1]
            @test group_avg == [22.5, 0]
            system_build.room_Temp = [16, 17]
            system_build.control_status = [0, 1]
            control_rooms, group_avg =
                HybridAutomata.update_control(system_build)
            @test control_rooms == [0, 0]
            @test group_avg == [16.5, 0]
        end
    end
end

# Test update_control - two rooms, two groups, two controls.
function test_update_physical_case1()
    system_build = HybridAutomata.System()
    system_build.room_size = [1, 1]
    system_build.room_THVAC = [i < 2 ? 55 : 50 for i in system_build.room_size]
    system_build.room_Temp = [70, 85]
    system_build.room_adjecent = [-1 1; 1 -1]
    system_build.control_map = [1 0; 0 1]
    system_build.control_group_num = [1, 1]
    system_build.control_status = [0, 1]
    system_build.control_rooms = [0, 1]
    system_build.T_min = 70
    system_build.T_max = 80
    system_build.room_num = length(system_build.room_size)
    system_build.A = Diagonal(0.1 * ones(system_build.room_num))  # α = 0.1
    system_build.B = Diagonal(0.1 * ones(system_build.room_num))  # β = 0.1
    system_build.Ω = 1.36 * ones(system_build.room_num)           # ω = 1.36
    system_build.Φ = Diagonal(0.6 * ones(system_build.room_num))  # φ = 0.6

    @testset verbose = true "test_update_physical_case1" begin
        # return: dif_external, dif_sum_neighbor, dif_air_u, room_Temp
        t_ext = 86
        noise = 0
        Δt  = 10 / 200
        dif_external, dif_sum_neighbor, dif_air_u, room_Temp =
            HybridAutomata.update_physical(system_build, t_ext, noise, Δt)
        @testset "Independent control" begin
            @test dif_external == [1.6, 0.1]
            @test dif_sum_neighbor == [1.5, -1.5]
            @test dif_air_u == [0, -18]
            @test room_Temp == [70.223, 84.098]
        end
    end
end

test_update_control_case1()
test_update_physical_case1()

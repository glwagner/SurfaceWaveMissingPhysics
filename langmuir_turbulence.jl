using Oceananigans
using Oceananigans.Units: minute, minutes, hours
using Printf

arch = CPU()

Nx = 64
Ny = Nx
Nz = 64
Lx = 256
Ly = Lx
Lz = 128

Jᵘ = -4e-5 # m² s⁻², surface kinematic momentum flux
Jᵇ = 2e-8  # m² s⁻³, surface buoyancy flux
N² = 2e-5  # s⁻², initial and bottom buoyancy gradient
f = 1e-4
g = 9.81
a = 0.8 # m
λ = 60  # m
k = 2π / λ
σ = sqrt(g * k) # s⁻¹

# Some initial condition parameters
initial_mixed_layer_depth = 10 # m
noise_amplitude = 1e-2 # non-dimensional

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       extent = (Lx, Ly, Lz),
                       halo = (5, 5, 5))

const ℓ = λ / 4π
const Uˢ = a^2 * k * σ

@inline uˢ(z) = Uˢ * exp(z / ℓ)
@inline ∂z_uˢ(z, t) = uˢ(z) / ℓ

u_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵘ))
b_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇ),
                                                bottom = GradientBoundaryCondition(N²))

coriolis = FPlane(; f) # s⁻¹

ν = CenterField(grid)
κ = CenterField(grid)
closure = ScalarDiffusivity(; ν, κ)

model = NonhydrostaticModel(; grid, coriolis, closure,
                            advection = WENO(order=5),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_boundary_conditions, b=b_boundary_conditions))

Ξ(z) = randn() * exp(z / 4)

stratification(z) = z < - initial_mixed_layer_depth ? N² * z : N² * (-initial_mixed_layer_depth)

bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * N² * model.grid.Lz

u★ = sqrt(abs(Jᵘ))
uᵢ(x, y, z) = u★ * noise_amplitude * Ξ(z)
wᵢ(x, y, z) = u★ * noise_amplitude * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

simulation = Simulation(model, Δt=30.0, stop_time=4hours)

wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))


function progress(simulation)
    u, v, w = simulation.model.velocities

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

# Output
fields_filename   = "langmuir_turbulence_fields.jld2"
slices_filename   = "langmuir_turbulence_slices.jld2"
averages_filename = "langmuir_turbulence_averages.jld2"

fields_output_interval = 5minutes
slices_output_interval = 1minute
averages_output_interval = slices_output_interval

fields_outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields_outputs,
                                                      schedule = TimeInterval(fields_output_interval),
                                                      filename = fields_filename,
                                                      overwrite_existing = true)

simulation.output_writers[:slices] = JLD2OutputWriter(model, fields_outputs,
                                                      schedule = TimeInterval(slices_output_interval),
                                                      indices = (:, 1, :),
                                                      filename = slices_filename,
                                                      overwrite_existing = true)


u, v, w = model.velocities
b = model.tracers.b

U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
B = Average(b, dims=(1, 2))
averaged_outputs = (; U, V, B)

simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_outputs,
                                                        schedule = TimeInterval(averages_output_interval),
                                                        filename = averages_filename,
                                                        overwrite_existing = true)

run!(simulation)

using GLMakie

wt = FieldTimeSeries(slices_filename, "w")
Bt = FieldTimeSeries(averages_filename, "B")
Ut = FieldTimeSeries(averages_filename, "U")
Vt = FieldTimeSeries(averages_filename, "V")

wmax = maximum(abs, wt)
wlim = wmax / 2
ulim = 0.06

t = wt.times
Nt = length(t)
n = Observable(Nt)

wn = @lift interior(wt[$n], :, 1, :)
Bn = @lift interior(Bt[$n], 1, 1, :)
Un = @lift interior(Ut[$n], 1, 1, :)
Vn = @lift interior(Vt[$n], 1, 1, :)

z = znodes(Bt)

fig = Figure(resolution=(1500, 500))

axw = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axB = Axis(fig[1, 2], xlabel="Buoyancy (m s⁻²)", ylabel="z (m)")
axU = Axis(fig[1, 3], xlabel="Velocities (m s⁻¹)", ylabel="z (m)")

heatmap!(axw, wn, colorrange=(-wlim, wlim), colormap=:balance)

lines!(axB, Bn, z)
lines!(axU, Un, z, label="u")
lines!(axU, Vn, z, label="v")

xlims!(axU, -ulim, ulim)

axislegend(axU)

display(fig)

record(fig, "langmuir_turbulence.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end

display(fig)


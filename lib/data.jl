module Data

using HDF5

export DEFAULT_PATH, advection_data_path, advection_data_paths, advection_parameters, load_trajectory

"""
    load_trajectory(path, idx=1) -> (u, dx, dt)

Load a single trajectory from an HDF5 file.

Returns:
- `u`  — Float32 matrix of shape (n_space, n_time)
- `dx` — spatial step size
- `dt` — temporal step size
"""
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const ADVECTION_TRAIN_RELATIVE_DIR = joinpath("data", "1D", "Advection", "Train")
const DEFAULT_RELATIVE_PATH = joinpath(
    ADVECTION_TRAIN_RELATIVE_DIR, "1D_Advection_Sols_beta1.0.hdf5"
)
const DEFAULT_PATH = joinpath(PROJECT_ROOT, DEFAULT_RELATIVE_PATH)

function advection_data_path(path::AbstractString = get(ENV, "PDE_DATA_PATH", DEFAULT_PATH))
    candidates = isabspath(path) ? [String(path)] : [
        String(path),
        joinpath(PROJECT_ROOT, path),
        joinpath(PROJECT_ROOT, ADVECTION_TRAIN_RELATIVE_DIR, path),
    ]

    for candidate in candidates
        if isfile(candidate)
            return normpath(candidate)
        end
    end

    tried = join(candidates, "\n  ")
    error("""
    HDF5 data file not found.

    Tried:
      $tried

    Download the data with pde/download_pdebench_subset.py, pass an explicit path to
    load_trajectory(path, idx), or set PDE_DATA_PATH.
    """)
end

function advection_data_paths()
    dirs = [joinpath(PROJECT_ROOT, ADVECTION_TRAIN_RELATIVE_DIR)]
    paths = String[]
    for dir in dirs
        isdir(dir) || continue
        append!(paths, joinpath.(dir, filter(name -> endswith(name, ".hdf5"), readdir(dir))))
    end
    unique_paths = unique(normpath.(paths))
    sort(unique_paths; by = path -> advection_parameters(path).beta)
end

function load_trajectory(path::AbstractString = get(ENV, "PDE_DATA_PATH", DEFAULT_PATH), idx::Int = 1)
    resolved_path = advection_data_path(path)
    u, x, t = h5open(resolved_path, "r") do file
        read(file, "tensor")[:, :, idx],
        read(file, "x-coordinate"),
        read(file, "t-coordinate")
    end
    dx = x[2] - x[1]
    dt = t[2] - t[1]
    # t-coordinate has one extra trailing entry vs. the stored solution snapshots
    t = t[1:size(u, 2)]
    return u, dx, dt, x, t
end

function advection_parameters(path::AbstractString = get(ENV, "PDE_DATA_PATH", DEFAULT_PATH))
    resolved_path = advection_data_path(path)
    beta, x, t = h5open(resolved_path, "r") do file
        file_attrs = attrs(file)
        beta = haskey(file_attrs, "beta") ? file_attrs["beta"] : nothing
        if beta === nothing
            beta_match = match(r"beta([\d.]+)\.hdf5", basename(resolved_path))
            beta_match === nothing && error("Could not infer beta from data file: $(basename(resolved_path))")
            beta = parse(Float64, beta_match.captures[1])
        end
        Float64(beta), read(file, "x-coordinate"), read(file, "t-coordinate")
    end
    dx = Float64(x[2] - x[1])
    dt = Float64(t[2] - t[1])
    return (; beta, dx, dt)
end

end # module Data

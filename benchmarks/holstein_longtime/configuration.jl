using TOML, SHA

function grid_count(duration, step)
    duration isa Real && isfinite(duration) && duration >= 0 ||
        throw(ArgumentError("duration must be finite and nonnegative"))
    step isa Real && isfinite(step) && step > 0 ||
        throw(ArgumentError("step must be positive and finite"))
    ratio = duration/step
    isfinite(ratio) && ratio < typemax(Int) || throw(ArgumentError("too many time steps"))
    n = round(Int,ratio)
    isapprox(ratio,n;atol=1e-10,rtol=1e-12) ||
        throw(ArgumentError("$duration must be an integer multiple of $step"))
    return n
end

canonical_hash(data) = bytes2hex(sha256(sprint(io -> TOML.print(io,data;sorted=true))))

function case_config(cfg, name, seen=String[])
    name in seen && throw(ArgumentError("cyclic run inheritance: $name"))
    haskey(cfg["runs"],name) || throw(ArgumentError("unknown run: $name"))
    own = copy(cfg["runs"][name])
    parent = pop!(own,"extends",nothing)
    result = isnothing(parent) ? own : merge(case_config(cfg,parent,[seen;name]),own)
    # Record the effective default in run identities and checkpoint metadata.
    if get(result,"method",nothing) in ("augmented","parallel")
        get!(result,"truncation","weighted")
    end
    return result
end

function checked_keys(table, expected, context)
    table isa AbstractDict || throw(ArgumentError("$context must be a table"))
    Set(keys(table)) == Set(expected) || throw(ArgumentError("invalid keys in $context: expected $expected"))
end

positive(x) = x isa Real && !(x isa Bool) && isfinite(x) && x > 0
integer_at_least(x,n) = x isa Integer && !(x isa Bool) && x >= n

function validate_config(cfg)
    checked_keys(cfg,["model","time","execution","comparison","runs"],"configuration")
    m,t,e,c = (cfg[k] for k in ("model","time","execution","comparison"))
    checked_keys(m,["sites","initial_site","hopping_cm","frequency_cm","damping_cm",
        "reorganization_cm","temperature_K","pade_order","hierarchy_local_size","operator_tolerance"],"model")
    integer_at_least(m["sites"],2) && integer_at_least(m["initial_site"],1) &&
        m["initial_site"] <= m["sites"] || throw(ArgumentError("invalid site count/initial site"))
    integer_at_least(m["pade_order"],0) && integer_at_least(m["hierarchy_local_size"],2) ||
        throw(ArgumentError("invalid bath expansion/hierarchy size"))
    all(positive(m[k]) for k in ("frequency_cm","damping_cm","reorganization_cm","temperature_K","operator_tolerance")) ||
        throw(ArgumentError("bath parameters/tolerance must be positive and finite"))
    m["hopping_cm"] isa Real && isfinite(m["hopping_cm"]) || throw(ArgumentError("invalid hopping"))
    checked_keys(t,["end_fs","observation_fs","checkpoint_fs"],"time")
    all(positive,values(t)) || throw(ArgumentError("time intervals must be positive and finite"))
    grid_count(t["end_fs"],t["observation_fs"]) >= 1 || throw(ArgumentError("empty trajectory"))
    grid_count(t["checkpoint_fs"],t["observation_fs"]) >= 1 || throw(ArgumentError("checkpoint interval too short"))
    checked_keys(e,["output_directory","seed"],"execution")
    e["output_directory"] isa String && !isempty(e["output_directory"]) || throw(ArgumentError("invalid output directory"))
    integer_at_least(e["seed"],0) || throw(ArgumentError("invalid seed"))
    checked_keys(c,["reference","reference_check","runs","state_target","density_target"],"comparison")
    positive(c["state_target"]) && positive(c["density_target"]) || throw(ArgumentError("invalid accuracy targets"))
    c["runs"] isa Vector && !isempty(c["runs"]) && all(x->x isa String,c["runs"]) ||
        throw(ArgumentError("comparison runs must be nonempty names"))
    for name in keys(cfg["runs"])
        occursin(r"^[A-Za-z0-9_-]+$",name) || throw(ArgumentError("invalid run name"))
        s = case_config(cfg,name)
        method = get(s,"method",nothing)
        method in ("tamen","augmented","parallel") || throw(ArgumentError("unknown method in $name"))
        expected = method=="tamen" ? ["method","dt_fs","tolerance","points","sweeps","local_iters","kick_rank","time_error_damp","blas_threads"] :
            ["method","dt_fs","truncation","truncation_rate","maxrank","krylov_tolerance","blas_threads"]
        checked_keys(s,expected,"run $name")
        integer_at_least(s["blas_threads"],1) || throw(ArgumentError("invalid BLAS threads"))
        grid_count(t["observation_fs"],s["dt_fs"]) >= 1 || throw(ArgumentError("step exceeds observation interval"))
        steps = grid_count(t["end_fs"],s["dt_fs"])
        e["seed"] <= typemax(Int)-steps || throw(ArgumentError("per-step seed would overflow"))
        if method=="tamen"
            positive(s["tolerance"]) && positive(s["time_error_damp"]) &&
                integer_at_least(s["points"],2) && integer_at_least(s["sweeps"],1) &&
                integer_at_least(s["local_iters"],1) && integer_at_least(s["kick_rank"],0) ||
                throw(ArgumentError("invalid tAMEn controls"))
        else
            s["truncation"] in ("weighted","local") ||
                throw(ArgumentError("truncation must be weighted or local"))
            positive(s["truncation_rate"]) && positive(s["krylov_tolerance"]) &&
                integer_at_least(s["maxrank"],m["sites"]) ||
                throw(ArgumentError("invalid BUG controls; split trace protection needs maxrank ≥ sites"))
        end
    end
    for name in [c["reference"],c["reference_check"],c["runs"]...]
        case_config(cfg,name)
    end
    for name in (c["reference"],c["reference_check"])
        case_config(cfg,name)["method"] == "tamen" || throw(ArgumentError("reference must use tAMEn"))
    end
    c["reference"] != c["reference_check"] || throw(ArgumentError("reference refinement must be a distinct run"))
    base,fine=(case_config(cfg,c[key]) for key in ("reference","reference_check"))
    fine["dt_fs"]<=base["dt_fs"] && fine["tolerance"]<=base["tolerance"] && fine["points"]>=base["points"] &&
        (fine["dt_fs"]<base["dt_fs"] || fine["tolerance"]<base["tolerance"] || fine["points"]>base["points"]) ||
        throw(ArgumentError("reference_check must tighten timestep, tolerance or time basis without loosening the others"))
    return cfg
end

load_config(path) = validate_config(TOML.parsefile(path))
run_spec(cfg,name) = Dict("model"=>cfg["model"],"time"=>cfg["time"],"solver"=>case_config(cfg,name),"seed"=>cfg["execution"]["seed"])
run_identity(cfg,name) = canonical_hash(run_spec(cfg,name))
output_root(cfg,path) = abspath(joinpath(dirname(abspath(path)),cfg["execution"]["output_directory"]))

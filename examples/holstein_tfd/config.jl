# One undamped physical oscillator per electronic site. Energies in cm^-1,
# time in fs, temperature in K. Couplings multiply x=(a+a†)/sqrt(2).
holstein_config = (
    site_energies_cm = [0.0, 0.0],
    hopping_cm = 200.0,
    frequencies_cm = [160.0, 160.0],
    couplings_cm = [80.0, 80.0],
    temperature_K = 300.0,
    boson_size = 3,               # each of four transformed modes: n=0,1,2
    periodic = true,              # the two-site bond is counted only once
    initial = :localized,         # :localized or :electronic_gibbs (bare H_e)
    initial_site = 1,
    tmax_fs = 50.0,
    method = :fixed,             # :fixed (dt and dt/2) or :adaptive
    fixed_dt_fs = 1.0,
    fixed_atol = 1e-10,          # rank-truncation budget per fixed step
    save_interval_fs = 1.0,      # adaptive output interval
    initial_dt_fs = 2.0,         # adaptive initial proposal
    adaptive_atol = 1e-9,        # adaptive local-error tolerance
    tolerances = (("loose",1e-3), ("tight",1e-5)),
)

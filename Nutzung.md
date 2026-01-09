# Nutzung der Micro-Swarm DLL

Die DLL stellt eine vollstaendige C-API bereit (FFI-sicher) und kann aus Python, Rust oder anderen Sprachen aufgerufen werden.

Dateien:
- `micro_swarm.dll` (Windows) bzw. `libmicro_swarm.so` (Linux)
- `src/micro_swarm_api.h` (C-API Header)

## Wichtiges Grundprinzip

- Alle Funktionen sind synchron, kein internes Threading.
- Alle Structs sind POD und `repr(C)` kompatibel.
- Felder werden als `float*` im Row-Major-Format genutzt (`width * height`).
- Ownership: `ms_create()` liefert einen Handle, der mit `ms_destroy()` freigegeben wird.

---

## Python (ctypes) Beispiel

```python
import ctypes as ct

lib = ct.CDLL(r".\build\Release\micro_swarm.dll")

class ms_params_t(ct.Structure):
    _fields_ = [
        ("width", ct.c_int),
        ("height", ct.c_int),
        ("agent_count", ct.c_int),
        ("steps", ct.c_int),
        ("pheromone_evaporation", ct.c_float),
        ("pheromone_diffusion", ct.c_float),
        ("molecule_evaporation", ct.c_float),
        ("molecule_diffusion", ct.c_float),
        ("resource_regen", ct.c_float),
        ("resource_max", ct.c_float),
        ("mycel_decay", ct.c_float),
        ("mycel_growth", ct.c_float),
        ("mycel_transport", ct.c_float),
        ("mycel_drive_threshold", ct.c_float),
        ("mycel_drive_p", ct.c_float),
        ("mycel_drive_r", ct.c_float),
        ("agent_move_cost", ct.c_float),
        ("agent_harvest", ct.c_float),
        ("agent_deposit_scale", ct.c_float),
        ("agent_sense_radius", ct.c_float),
        ("agent_random_turn", ct.c_float),
        ("dna_capacity", ct.c_int),
        ("dna_global_capacity", ct.c_int),
        ("dna_survival_bias", ct.c_float),
        ("phero_food_deposit_scale", ct.c_float),
        ("phero_danger_deposit_scale", ct.c_float),
        ("danger_delta_threshold", ct.c_float),
        ("danger_bounce_deposit", ct.c_float),
        ("evo_enable", ct.c_int),
        ("evo_elite_frac", ct.c_float),
        ("evo_min_energy_to_store", ct.c_float),
        ("evo_mutation_sigma", ct.c_float),
        ("evo_exploration_delta", ct.c_float),
        ("evo_fitness_window", ct.c_int),
        ("evo_age_decay", ct.c_float),
        ("global_spawn_frac", ct.c_float),
    ]

class ms_config_t(ct.Structure):
    _fields_ = [("params", ms_params_t), ("seed", ct.c_uint32)]

lib.ms_create.restype = ct.c_void_p
lib.ms_destroy.argtypes = [ct.c_void_p]
lib.ms_step.argtypes = [ct.c_void_p, ct.c_int]
lib.ms_get_field_info.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
lib.ms_copy_field_out.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_float), ct.c_int]

cfg = ms_config_t()
cfg.params.width = 128
cfg.params.height = 128
cfg.params.agent_count = 512
cfg.params.steps = 100
cfg.seed = 42

h = lib.ms_create(ct.byref(cfg))
lib.ms_step(h, 10)

w = ct.c_int()
hgt = ct.c_int()
MS_FIELD_PHEROMONE_FOOD = 1
lib.ms_get_field_info(h, MS_FIELD_PHEROMONE_FOOD, ct.byref(w), ct.byref(hgt))
count = w.value * hgt.value
buffer = (ct.c_float * count)()
lib.ms_copy_field_out(h, MS_FIELD_PHEROMONE_FOOD, buffer, count)

lib.ms_destroy(h)
```

---

## Rust (FFI) Beispiel

```rust
#[repr(C)]
pub struct ms_params_t {
    width: i32,
    height: i32,
    agent_count: i32,
    steps: i32,
    pheromone_evaporation: f32,
    pheromone_diffusion: f32,
    molecule_evaporation: f32,
    molecule_diffusion: f32,
    resource_regen: f32,
    resource_max: f32,
    mycel_decay: f32,
    mycel_growth: f32,
    mycel_transport: f32,
    mycel_drive_threshold: f32,
    mycel_drive_p: f32,
    mycel_drive_r: f32,
    agent_move_cost: f32,
    agent_harvest: f32,
    agent_deposit_scale: f32,
    agent_sense_radius: f32,
    agent_random_turn: f32,
    dna_capacity: i32,
    dna_global_capacity: i32,
    dna_survival_bias: f32,
    phero_food_deposit_scale: f32,
    phero_danger_deposit_scale: f32,
    danger_delta_threshold: f32,
    danger_bounce_deposit: f32,
    evo_enable: i32,
    evo_elite_frac: f32,
    evo_min_energy_to_store: f32,
    evo_mutation_sigma: f32,
    evo_exploration_delta: f32,
    evo_fitness_window: i32,
    evo_age_decay: f32,
    global_spawn_frac: f32,
}

#[repr(C)]
pub struct ms_config_t {
    params: ms_params_t,
    seed: u32,
}

extern "C" {
    fn ms_create(cfg: *const ms_config_t) -> *mut std::ffi::c_void;
    fn ms_destroy(h: *mut std::ffi::c_void);
    fn ms_step(h: *mut std::ffi::c_void, steps: i32) -> i32;
}

fn main() {
    let cfg = ms_config_t { params: unsafe { std::mem::zeroed() }, seed: 42 };
    unsafe {
        let h = ms_create(&cfg);
        ms_step(h, 10);
        ms_destroy(h);
    }
}
```

---

## Ownership / Hinweise

- `ms_create()` erzeugt den Kontext und initialisiert Felder + Agenten.
- `ms_destroy()` MUSS immer aufgerufen werden.
- `ms_copy_field_in/out` arbeitet mit rohen Float-Arrays.
- `ms_ocl_enable()` kann die GPU-Diffusion aktivieren (falls OpenCL vorhanden).

from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
sample_scheduler.set_timesteps(
    40, shift=1)
timesteps = sample_scheduler.timesteps
print(timesteps)
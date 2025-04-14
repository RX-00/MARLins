from gymnasium.envs.registration import register

register(
    id='Kilobots-QuadAssembly-v0',
    entry_point='gym_kilobots.envs:QuadAssemblyKilobotsEnv',  # Ensure the entry point is correct
)

register(
    id='Kilobots-Yaml-v0',
    entry_point='gym_kilobots.envs:YamlKilobotsEnv',
)

register(
    id='Kilobots-DirectControl-v0',
    entry_point='gym_kilobots.envs:DirectControlKilobotsEnv',
)


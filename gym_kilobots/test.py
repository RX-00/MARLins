import os
os.environ["PYGAME_DETECT_AVX2"] = "1"
import numpy as np
import gymnasium as gym
import gym_kilobots
import pygame  # Initialize pygame for keyboard input

if __name__ == "__main__":
    pygame.init()  # Initialize pygame for keyboard input

    # Define four objects in a centered square
    square_objects = [
        ((-0.25,  0.25), 0.0),   # top-left
        (( 0.25,  0.25), 0.0),   # top-right
        (( 0.25, -0.25), 0.0),   # bottom-right
        ((-0.25, -0.25), 0.0),   # bottom-left
    ]

    # Define 1 object in the center of the environment
    single_center_object = [
        ((0.0, 0.0), 0.0)        # center,
    ]

    # Choose a fixed light position
    light_pos = (0.5, 0.5)

    # Choose four explicit kilobot positions
    kb_positions = [
        (-0.50,  0.60),
        ( 0.50,  0.60),
        ( 0.60,  0.50),
        ( 0.60, -0.50),
    ]

    # NOTE: if you don't do this, the environment will randomly place the objects
    env = gym.make(
        'Kilobots-QuadAssembly-v0',
        render_mode='human',
        num_kilobots=len(kb_positions),
        object_config=square_objects,
        light_position=light_pos,
        kilobot_positions=kb_positions
    )
    #env = gym.make('Kilobots-QuadAssembly-v0', render_mode='human', num_kilobots=4, num_objects=2)

    obs, info = env.reset()

    running = True
    while running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Control the light source with the keyboard
        # Get keyboard state and set action accordingly
        # NOTE: if you want to control the light source with a random input, just give env.step() the argument: env.action_space.sample()
        keys = pygame.key.get_pressed()
        action = [0.0, 0.0]
        if keys[pygame.K_UP]:
            action[1] = 1.0
        if keys[pygame.K_DOWN]:
            action[1] = -1.0
        if keys[pygame.K_LEFT]:
            action[0] = -1.0
        if keys[pygame.K_RIGHT]:
            action[0] = 1.0

        env.render()
        obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))

        # Print the orientation of the objects using the unwrapped environment
        print(env.unwrapped.get_objects_status())

        # End episode
        if terminated or truncated:
            break

    env.close()  # Ensure the environment is properly closed
    pygame.quit()  # Quit pygame
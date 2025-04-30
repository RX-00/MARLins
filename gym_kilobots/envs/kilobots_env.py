import time

import gymnasium as gym
from gymnasium import spaces

import numpy as np

from Box2D import b2World, b2ChainShape

from ..lib.body import Body, _world_scale
from ..lib.kilobot import Kilobot
from ..lib.light import Light

import abc


class KilobotsEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}  # Updated metadata

    world_size = world_width, world_height = 2., 1.5
    screen_size = screen_width, screen_height = 1200, 900

    _observe_objects = True
    _observe_light = True

    __sim_steps_per_second = 10
    __sim_velocity_iterations = 10
    __sim_position_iterations = 10
    __steps_per_action = 10

    def __new__(cls, **kwargs):
        cls.sim_steps_per_second = cls.__sim_steps_per_second
        cls.sim_step = 1. / cls.__sim_steps_per_second
        cls.world_x_range = -cls.world_width / 2, cls.world_width / 2
        cls.world_y_range = -cls.world_height / 2, cls.world_height / 2
        cls.world_bounds = (np.array([-cls.world_width / 2, -cls.world_height / 2]),
                            np.array([cls.world_width / 2, cls.world_height / 2]))

        return super(KilobotsEnv, cls).__new__(cls)

    def __init__(self, render_mode=None, **kwargs):
        self.__sim_steps = 0
        self.__reset_counter = 0

        # create the Kilobots world in Box2D
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.table = self.world.CreateStaticBody(position=(.0, .0))
        self.table.CreateFixture(
            shape=b2ChainShape(vertices=[(_world_scale * self.world_x_range[0], _world_scale * self.world_y_range[1]),
                                         (_world_scale * self.world_x_range[0], _world_scale * self.world_y_range[0]),
                                         (_world_scale * self.world_x_range[1], _world_scale * self.world_y_range[0]),
                                         (_world_scale * self.world_x_range[1], _world_scale * self.world_y_range[1])]))
        self._real_time = False

        # add kilobots
        self._kilobots: [Kilobot] = []
        # add objects
        self._objects: [Body] = []
        # add light
        self._light: Light = None

        self.__seed = 0

        self._screen = None
        self.render_mode = 'human'
        self.video_path = None

        # NOTE: we are going to set the environment and kilobots in the actual environment implementation so that we can pass arguments to it.
        #self._configure_environment()
        #self._kilobots = []

        # Define default observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )  # Placeholder, should be overridden by subclasses
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )  # Placeholder, should be overridden by subclasses

        self._step_world()

        self.max_steps = 500
        self.n_steps = 0

    @property
    def _sim_steps(self):
        return self.__sim_steps

    @property
    def kilobots(self):
        return tuple(self._kilobots)

    @property
    def num_kilobots(self):
        return len(self._kilobots)

    @property
    def objects(self):
        return tuple(self._objects)

    @property
    def _steps_per_action(self):
        return self.__steps_per_action

    def _add_kilobot(self, kilobot: Kilobot):
        self._kilobots.append(kilobot)

    def _add_object(self, body: Body):
        self._objects.append(body)

    @abc.abstractmethod
    def _configure_environment(self):
        raise NotImplementedError

    def get_state(self):
        return {'kilobots': np.array([k.get_state() for k in self._kilobots]),
                'objects': np.array([o.get_state() for o in self._objects]),
                'light': self._light.get_state()}

    def get_observation(self):
        # Flatten the state dictionary into a single NumPy array
        kilobots_state = np.array([k.get_state() for k in self._kilobots], dtype=np.float32).flatten()
        objects_state = np.array([o.get_state() for o in self._objects], dtype=np.float32).flatten()
        light_state = np.array(self._light.get_state(), dtype=np.float32).flatten() if self._light else np.array([], dtype=np.float32)

        # Concatenate all components into a single observation array
        return np.concatenate([kilobots_state, objects_state, light_state]).astype(np.float32)
        #return objects_state.astype(np.float32)

    @abc.abstractmethod
    def get_reward(self, state, action):
        raise NotImplementedError

    def has_finished(self, state, action):
        return False

    def get_info(self, state, action):
        # Ensure the method always returns a dictionary
        return {}  # Explicitly return an empty dictionary

    def destroy(self):
        del self._objects[:]
        del self._kilobots[:]
        del self._light
        self._light = None
        if self._screen is not None:
            del self._screen
            self._screen = None

    def close(self):
        self.destroy()

    def seed(self, seed=None):
        if seed is not None:
            self.__seed = seed
        return [self.__seed]

    def reset(self, *, seed=None, options=None):
        # Handle the seed for random number generation
        if seed is not None:
            self.seed(seed)

        # Handle options (if any)
        if options is not None:
            # Process options if needed
            pass

        self.__reset_counter += 1
        self.destroy()
        self._configure_environment()
        self.__sim_steps = 0

        # Step to resolve initial state
        self._step_world()

        # Get observation
        observation = self.get_observation()

        # Reset number of steps
        self.n_steps = 0

        # Return observation and an empty info dictionary
        info = {}
        return observation, info

    def step(self, action: np.ndarray):
        # if self.action_space and action is not None:
        #     assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # state before action is applied
        state = self.get_state()

        for i in range(self.__steps_per_action):
            _t_step_start = time.time()
            # step light
            if action is not None and self._light:
                self._light.step(action, self.sim_step)

            if self._light:
                # compute light values and gradients
                sensor_positions = np.array([kb.light_sensor_pos() for kb in self._kilobots])
                values, gradients = self._light.value_and_gradients(sensor_positions)

                for kb, v, g in zip(self._kilobots, values, gradients):
                    kb.set_light_value_and_gradient(v, g)

            # step kilobots
            for k in self._kilobots:
                k.step(self.sim_step)

            # step world
            self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
            self.world.ClearForces()

            self.__sim_steps += 1

            if self._screen is not None:
                self.render(self.render_mode)

            _t_step_end = time.time()

            if self._real_time:
                time.sleep(max(self.sim_step - (_t_step_end - _t_step_start), .0))

        # state
        next_state = self.get_state()

        # observation
        observation = self.get_observation()

        # reward
        reward = self.get_reward(state, action)

        # keep track of number of steps
        self.n_steps +=1

        # done (split into terminated and truncated)
        terminated = bool(self.has_finished(next_state, action))  # Ensure Python bool
        truncated = bool(self.n_steps >= self.max_steps)  # Add logic for truncation if needed

        # info
        info = self.get_info(next_state, action)
        if not isinstance(info, dict):  # Ensure info is always a dictionary
            info = {}

        return observation, reward, terminated, truncated, info

    def _step_world(self):
        self.world.Step(self.sim_step, self.__sim_velocity_iterations, self.__sim_position_iterations)
        self.world.ClearForces()

    def render(self, mode=None):
        # if close:
        #     if self._screen is not None:
        #         self._screen.close()
        #         self._screen = None
        #     return
        if mode is None:
            mode = self.render_mode

        from gym_kilobots import kb_rendering
        if self._screen is None:
            caption = self.spec.id if self.spec else ""
            if self.video_path:
                import os
                os.makedirs(self.video_path, exist_ok=True)
                _video_path = os.path.join(self.video_path, str(self.__reset_counter) + '.mp4')
            else:
                _video_path = None

            self._screen = kb_rendering.KilobotsViewer(self.screen_width, self.screen_height, caption=caption,
                                                       display=mode == 'human', record_to=_video_path)
            world_min, world_max = self.world_bounds
            self._screen.set_bounds(world_min[0], world_max[0], world_min[1], world_max[1])
        elif self._screen.close_requested():
            self._screen.close()
            self._screen = None
            # TODO how to handle this event?

        # render table
        x_min, x_max = self.world_x_range
        y_min, y_max = self.world_y_range
        self._screen.draw_polygon([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)],
                                  color=(255, 255, 255))
        self._screen.draw_polyline([(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)],
                                   width=.003)

        # allow to draw on table
        self._draw_on_table(self._screen)

        # render objects
        for o in self._objects:
            o.draw(self._screen)

        # render kilobots
        for kb in self._kilobots:
            kb.draw(self._screen)

        # render light
        if self._light is not None:
            self._light.draw(self._screen)

        # allow to draw on top
        self._draw_on_top(self._screen)

        self._screen.render()

    def get_objects(self) -> [Body]:
        return self._objects

    def get_kilobots(self) -> [Kilobot]:
        return self._kilobots

    def get_light(self) -> Light:
        return self._light

    def _draw_on_table(self, screen):
        pass

    def _draw_on_top(self, screen):
        pass


class UnknownObjectException(Exception):
    pass


class UnknownLightTypeException(Exception):
    pass

from .kilobots_env import KilobotsEnv

import numpy as np
from scipy import stats
from gymnasium import spaces  # Import spaces from gymnasium

from ..lib.body import CornerQuad, Triangle, LForm, CForm, TForm
from ..lib.kilobot import PhototaxisKilobot, SimplePhototaxisKilobot
from ..lib.light import CircularGradientLight


class QuadPushingEnv(KilobotsEnv):
    world_size = world_width, world_height = 1., .5

    def __init__(self):
        # distribution for sampling swarm position
        self._swarm_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(.9, 1.4))
        # distribution for sampling the pushing object
        self._obj_spawn_distribution = stats.uniform(loc=(.05, -.7), scale=(.9, .65))

        super().__init__()


class QuadAssemblyKilobotsEnv(KilobotsEnv):
    def __init__(self,
                 render_mode=None,
                 num_kilobots=5,
                 num_objects=1,
                 object_config=None,
                 light_position=None,
                 kilobot_positions=None,
                 ):
        # Initialize attributes before calling super().__init__()
        self._swarm_spawn_distribution = stats.uniform(loc=(-.95, -.7), scale=(.9, 1.4))
        self._obj_spawn_distribution = stats.uniform(loc=(.05, -.7), scale=(.9, .65))
        self._light_position = light_position
        self._kilobot_positions = kilobot_positions
        
        super().__init__(render_mode=render_mode)  # Pass render_mode to the parent class

        # Rename attributes to avoid conflict with read-only properties
        self._num_kilobots = num_kilobots  # Number of kilobots
        self._num_objects = num_objects    # Number of objects
        self._object_config = object_config  # Optional list of (position, orientation)
        self.kilobot_obs_dim = 2  # x, y positions
        self.object_obs_dim = 3  # x, y, orientation

        print(self._num_objects)

        obs, _ = self.reset()   # sample the true reset() output

        # Define the observation space
        # The observation space includes:
        # - Kilobot observations: positions (x, y) of each kilobot
        # - Object observations: positions (x, y) and orientations of each object
        # - Additional environment-specific information (if applicable)
        # Refer to KiloBotsEnv for the exact structure of the observation space in get_observation()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,  # Shape depends on the number of kilobots and objects
            dtype=obs.dtype,
        )

        # Define the action space
        # Example: 2D continuous control for light source movement
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )  # Example action space for light control

    def _configure_environment(self):
        swarm_spawn_location = self._swarm_spawn_distribution.rvs()
        obj_spawn_location = self._obj_spawn_distribution.rvs()

        # Light spawn
        if self._light_position is not None:
            swarm_spawn_location = tuple(self._light_position)
        else:
            swarm_spawn_location = self._swarm_spawn_distribution.rvs()
        # Object spawn (unchanged)
        obj_spawn_location   = self._obj_spawn_distribution.rvs()

        # Spawn objects: use specific config if provided
        self._objects = []
        if self._object_config is not None:
            for config in self._object_config:
                # Each config is assumed to be a tuple: (position, orientation)
                pos, orientation = config
                self._objects.append(CornerQuad(world=self.world, width=0.15, height=0.15, position=pos, orientation=orientation))
        else:
            for i in range(self._num_objects):
                offset = (0.02 * i, 0.02 * i)
                pos = (obj_spawn_location[0] + offset[0], obj_spawn_location[1] + offset[1])
                self._objects.append(CornerQuad(world=self.world, width=0.15, height=0.15, position=pos, orientation=-np.pi/2))

        self._light = CircularGradientLight(position=swarm_spawn_location)
        
        # Spawn kilobots at fixed positions or with the usual symmetric offsets:
        self._kilobots = []
        if self._kilobot_positions is not None:
            for pos in self._kilobot_positions:
                self._kilobots.append(PhototaxisKilobot(self.world, position=tuple(pos), light=self._light))
        else:
            for i in range(self._num_kilobots):
                angle = 2 * np.pi * i / self._num_kilobots
                offset = (0.03 * np.cos(angle), 0.03 * np.sin(angle))
                pos = (swarm_spawn_location[0] + offset[0], swarm_spawn_location[1] + offset[1])
                self._kilobots.append(PhototaxisKilobot(self.world, position=pos, light=self._light))

    def has_finished(self, state, action):
        return False

    def get_reward(self, state, action, new_state):
        return 1.0

    def get_info(self, state, action):
        return None

    def get_objects_status(self):
        objects_info = []

        # Iterate through the spawned objects and print their status
        for i, obj in enumerate(self._objects):
            objects_info.append(f"object {i}: position: {obj.get_position()}, orientation: {obj.get_orientation()}")

        return objects_info

class TriangleTestEnv(KilobotsEnv):
    def _configure_environment(self):
        self._objects = [Triangle(world=self.world, width=.15, height=.15, position=(.0, .0)),
                         LForm(world=self.world, width=.15, height=.15, position=(.0, .3)),
                         TForm(world=self.world, width=.15, height=.15, position=(.0, -.3)),
                         CForm(world=self.world, width=.15, height=.15, position=(.3, .0))]

    # TODO implement has_finished function
    def has_finished(self, state, action):
        return False

    # TODO implement reward function
    def get_reward(self, state, action, new_state):
        # compute reward based on task and swarm state
        return 1.

    # info function
    def get_info(self, state, action):
        return None
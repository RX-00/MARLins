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

        #print(self._num_objects)

        obs, _ = self.reset()   # sample the true reset() output

        # Define the observation space
        # The observation space includes:
        # - Kilobot observations: positions (x, y) of each kilobot
        # - Object observations: positions (x, y) and orientations of each object
        # - Additional environment-specific information (if applicable)
        # Refer to KiloBotsEnv for the exact structure of the observation space in get_observation()
        self.observation_space = spaces.Box(
            low=-100,
            high=100,
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
        # Parse the observation to get kilobots, object, and light information
        #kilobots_info, _, light_info = self.parse_observation(state)

        kilobots_info = state['kilobots']
        light_info = state['light']

        # Terminate if any kilobot leaves the radius of the light
        light_radius = 0.3  # Define the radius of the light
        kilobot_positions = kilobots_info[:, :2]  # Get x, y positions of kilobots
        distances_to_light = np.linalg.norm(kilobot_positions - light_info, axis=1)
        
        # Check if any kilobot is outside the light radius
        if np.any(distances_to_light > light_radius):
            #print("Termination: A kilobot left the radius of the light.")
            return True

        # Terminate if the light's position exits the size of the displayed environment
        environment_bounds = np.array([[-1.0, -0.5], [1.0, 0.5]])  # Define environment bounds as [[x_min, y_min], [x_max, y_max]]
        if not (environment_bounds[0, 0] <= light_info[0] <= environment_bounds[1, 0] and
                environment_bounds[0, 1] <= light_info[1] <= environment_bounds[1, 1]):
            #print("Termination: The light exited the environment bounds.")
            return True

        return False

    def get_reward(self, state, action):
        kilobots_info = state["kilobots"]
        object_info = state["objects"].squeeze()
        light_info = state["light"]

        goal = np.array([0.0, 0.0, 0.0])

        # 1) Encourage object to goal
        distance_obj_goal = np.linalg.norm(object_info[:2] - goal[:2])
        orientation_error = np.abs(object_info[2] - goal[2])
        r_obj_dist = np.exp(-distance_obj_goal)
        r_obj_orient = np.exp(-orientation_error)

        # 2) Encourage light near object
        dist_light_obj = np.linalg.norm(light_info - object_info[:2])
        r_light_obj = np.exp(-dist_light_obj)

        # 3) Encourage swarm near object
        #swarm_center = np.mean(kilobots_info[:, :2], axis=0)
        #dist_swarm_obj = np.linalg.norm(swarm_center - object_info[:2])
        #r_swarm_obj = np.exp(-dist_swarm_obj)

        # 4) Combine rewards
        reward = 10 * r_obj_dist + 0.5 * r_obj_orient # + 0.5 * r_light_obj # + 0.3 * r_swarm_obj

        # Apply a large penalty if the object is far from the goal
        if distance_obj_goal > 0.0005:
            reward -= 10.0

        return reward

    def get_info(self, state, action):
        return None

    def get_objects_status(self):
        objects_info = []

        # Iterate through the spawned objects and print their status
        for i, obj in enumerate(self._objects):
            objects_info.append(f"object {i}: position: {obj.get_position()}, orientation: {obj.get_orientation()}")

        return objects_info
    
    # Function to parse observation array
    def parse_observation(self, obs_array):
        # Each kilobot has x, y, orientation (3 values)
        kilobot_dim = 3
        # Object has x, y, orientation (3 values)
        object_dim = 3
        # Light has x, y (2 values)
        light_dim = 2

        # Check if the observation is a scalar or a 1-element array
        obs_array = np.array(obs_array).flatten()
        if obs_array.size <= 1:
            # Return default empty values when observation is a scalar or too small
            empty_kilobots = np.zeros((self._num_kilobots, kilobot_dim))
            empty_object = np.zeros(object_dim)
            empty_light = np.zeros(light_dim)
            return empty_kilobots, empty_object, empty_light
        
        # Make sure the array has the expected minimum size
        expected_size = self._num_kilobots * kilobot_dim + object_dim + light_dim
        if obs_array.size < expected_size:
            # Pad with zeros if the array is smaller than expected
            padded_array = np.zeros(expected_size)
            padded_array[:obs_array.size] = obs_array
            obs_array = padded_array
        
        # Calculate array indices
        kilobots_end = self._num_kilobots * kilobot_dim
        object_end = kilobots_end + object_dim
        
        # Extract data with safety checks
        kilobots = obs_array[:kilobots_end].reshape(self._num_kilobots, kilobot_dim)
        object_data = obs_array[kilobots_end:object_end]
        light_data = obs_array[object_end:object_end+light_dim]  # Limit to expected light dimensions
        
        return kilobots, object_data, light_data



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
import numpy as np
from Box2D import b2Vec2
from gymnasium import spaces

from .body import Circle, _world_scale


class Kilobot(Circle):
    _radius = 0.0165

    _leg_front = np.array([.0, _radius])
    _leg_left = np.array([-0.013, -.009])
    _leg_right = np.array([+0.013, -.009])
    _light_sensor = np.array([.0, -_radius+.001])
    _led = np.array([.011, .01])

    # _impulse_right_dir = _leg_front - _leg_right
    # _impulse_left_dir = _leg_front - _leg_left
    # _impulse_right_point_body = (_leg_front + _leg_right) / 2
    # _impulse_left_point_body = (_leg_front + _leg_left) / 2

    _max_linear_velocity = 0.01  # meters / s
    _max_angular_velocity = 0.5 * np.pi  # radians / s

    _density = 1.0
    _friction = 0.0
    _restitution = 0.0

    _linear_damping = .8  #* _world_scale
    _angular_damping = .8  #* _world_scale

    def __init__(self, world, position=None, orientation=None, light=None):
        # all parameters in real world units
        super().__init__(world=world, position=position, orientation=orientation, radius=self._radius)

        # 0 .. 255
        self._motor_left = 0
        self._motor_right = 0
        self.__light_measurement = 0
        self.__turn_direction = None

        self._body_color = (150, 150, 150)
        self._highlight_color = (255, 255, 255)

        self._light_value = None
        self._light_gradient = None

        self._setup()

    def set_light_value_and_gradient(self, value, gradient):
        self._light_value = value
        self._light_gradient = gradient

    def light_sensor_pos(self):
        return self.get_world_point((0.0, -self._radius))

    def get_ambientlight(self):
        if self._light_value:
            return self._light_value
        else:
            return 0

    def set_motors(self, left, right):
        self._motor_left = left
        self._motor_right = right

    def switch_directions(self, motor_pwr=255):
        if self.__turn_direction == 'left':
            self.turn_right(motor_pwr)
        else:
            self.turn_left(motor_pwr)

    def turn_right(self, motor_pwr=255):
        self.__turn_direction = 'right'
        self.set_motors(0, motor_pwr)
        self.set_color((255, 0, 0))

    def turn_left(self, motor_pwr=255):
        self.__turn_direction = 'left'
        self.set_motors(motor_pwr, 0)
        self.set_color((0, 255, 0))

    def set_color(self, color):
        self._highlight_color = color

    def step(self, time_step):
        # loop kilobot logic
        self._loop()

        cos_dir = np.cos(self.get_orientation())
        sin_dir = np.sin(self.get_orientation())

        linear_velocity = [.0, .0]
        angular_velocity = .0

        # compute new kilobot position or kilobot velocity
        if self._motor_left and self._motor_right:
            linear_velocity = (self._motor_right + self._motor_left) / 510. * self._max_linear_velocity
            linear_velocity = [sin_dir * linear_velocity, cos_dir * linear_velocity]

            angular_velocity = (self._motor_right - self._motor_left) / 510. * self._max_angular_velocity

        elif self._motor_right:
            angular_velocity = self._motor_right / 255. * self._max_angular_velocity
            angular_displacement = angular_velocity * time_step

            c, s = np.cos(angular_displacement), np.sin(angular_displacement)
            R = [[c, -s], [s, c]]

            translation = self._leg_left - np.dot(R, self._leg_left)
            linear_velocity = self._body.GetWorldVector(translation * _world_scale) / _world_scale / time_step

        elif self._motor_left:
            angular_velocity = -self._motor_left / 255. * self._max_angular_velocity
            angular_displacement = angular_velocity * time_step

            c, s = np.cos(angular_displacement), np.sin(angular_displacement)
            R = [[c, -s], [s, c]]

            translation = self._leg_right - np.dot(R, self._leg_right)
            linear_velocity = self._body.GetWorldVector(translation * _world_scale) / _world_scale / time_step

        self._body.angularVelocity = angular_velocity
        if type(linear_velocity) == np.ndarray:
            self._body.linearVelocity = b2Vec2(*linear_velocity.astype(float)) * _world_scale
        else:
            self._body.linearVelocity = linear_velocity * _world_scale

    def draw(self, viewer):
        # super(Kilobot, self).draw(viewer)
        # viewer.draw_circle(position=self._body.position, radius=self._radius, color=(50,) * 3, filled=False)
        viewer.draw_aacircle(position=self.get_position(), radius=self._radius + .002, color=self._body_color)
        viewer.draw_aacircle(position=self.get_position(), radius=self._radius + .002, color=(100, 100, 100),
                             filled=False, width=.005)

        # draw direction as triangle with color set by function
        front = self.get_world_point((self._radius - .005, 0.0))
        # w = 0.1 * self._radius
        # h = np.cos(np.arcsin(w)) - self._radius
        # bottom_left = self._body.GetWorldPoint((-0.006, -0.009))
        # bottom_right = self._body.GetWorldPoint((0.006, -0.009))
        middle = self.get_position()

        # viewer.draw_polygon(vertices=(top, bottom_left, bottom_right), color=self._highlight_color)
        viewer.draw_polyline(vertices=(front, middle), color=self._highlight_color, closed=False, width=.005)

        # t = rendering.Transform(translation=self._body.GetWorldPoint(self._led))
        # viewer.draw_circle(.003, res=20, color=self._highlight_color).add_attr(t)
        # viewer.draw_circle(.003, res=20, color=(0, 0, 0), filled=False).add_attr(t)

        # light sensor
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._light_sensor), radius=.005, color=(0, 0, 0))
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._light_sensor), radius=.0035, color=(255, 255, 0))

        # draw legs
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._leg_front), radius=.001, color=(0, 0, 0))
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._leg_left), radius=.001, color=(0, 0, 0))
        # viewer.draw_circle(position=self._body.GetWorldPoint(self._leg_right), radius=.001, color=(0, 0, 0))

    @classmethod
    def get_radius(cls):
        return cls._radius

    def _setup(self):
        raise NotImplementedError('Kilobot subclass needs to implement _setup')

    def _loop(self):
        raise NotImplementedError('Kilobot subclass needs to implement _loop')


class SimplePhototaxisKilobot(Kilobot):
    def __init__(self, world, position=None, orientation=None, light=None):
        super().__init__(world=world, position=position, orientation=orientation, light=light)

        self.last_light = 0
        self.turn_cw = 1
        self.counter = 0

        self.env = world

    def _setup(self):
        self.turn_left()

    def _loop(self):
        # we override step
        pass

    def light_sensor_pos(self):
        return self.get_position()

    def step(self, time_step):
        movement_direction = self._light_gradient

        n = np.sqrt(np.dot(movement_direction, movement_direction))
        # n = np.linalg.norm(movement_direction)
        if n > self._max_linear_velocity:
            movement_direction = movement_direction / n * self._max_linear_velocity

        movement_direction *= _world_scale

        self._body.linearVelocity = b2Vec2(*movement_direction.astype(float))
        # self._body.angle = np.arctan2(movement_direction[1], movement_direction[0])
        self._body.linearDamping = .0

    def draw(self, viewer):
        # super(Kilobot, self).draw(viewer)
        # viewer.draw_circle(position=self._body.position, radius=self._radius, color=(50,) * 3, filled=False)
        viewer.draw_aacircle(position=self.get_position(), radius=self._radius + .002, color=self._body_color)
        viewer.draw_aacircle(position=self.get_position(), radius=self._radius + .002, color=(100, 100, 100),
                             filled=False, width=.005)


class SimpleVelocityControlKilobot(Kilobot):
    _density = 2.0

    action_space = spaces.Box(np.array([.0, -Kilobot._max_angular_velocity]),
                              np.array([Kilobot._max_linear_velocity, Kilobot._max_angular_velocity]),
                              dtype=np.float64)
    state_space = spaces.Box(np.array([-np.inf, -np.inf, -np.inf]),
                             np.array([np.inf, np.inf, np.inf, ]), dtype=np.float64)

    def __init__(self, world, *, velocity=None, **kwargs):
        super().__init__(world=world, light=None, **kwargs)

        if velocity:
            self._velocity = velocity
        else:
            self._velocity = np.random.rand(2) * np.array([self._max_linear_velocity, 2 * self._max_angular_velocity])
            self._velocity[1] -= self._max_angular_velocity

    # def get_state(self):
    #     pose = super(SimpleVelocityControlKilobot, self).get_state()
    #     return pose + tuple(self._velocity)

    def set_action(self, action):
        if action is not None:
            action = np.minimum(action, self.action_space.high)
            action = np.maximum(action, self.action_space.low)
            self._velocity = action
        else:
            self._velocity = np.array([.0, .0])

    def get_action(self):
        return self._velocity

    def _setup(self):
        pass

    def _loop(self):
        # we override step
        pass

    def step(self, time_step):
        linear_velocity = np.array([np.cos(self.get_orientation()), np.sin(self.get_orientation())])
        linear_velocity *= self._velocity[0] * _world_scale

        self._body.linearVelocity = b2Vec2(*linear_velocity.astype(float))
        self._body.angularVelocity = self._velocity[1]
        # self._body.linearDamping = .0
        # self._body.angularDamping = .0

    def set_color(self, color):
        self._body_color = color


class SimpleAccelerationControlKilobot(SimpleVelocityControlKilobot):
    _density = 2.0

    action_space = spaces.Box(np.array([-.005, -.2 * np.pi]), np.array([.005, .2 * np.pi]), dtype=np.float64)
    state_space = spaces.Box(np.array([-np.inf, -np.inf, -np.inf, .0, -Kilobot._max_angular_velocity]),
                                        np.array([np.inf, np.inf, np.inf, Kilobot._max_linear_velocity,
                                            Kilobot._max_angular_velocity]), dtype=np.float64)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._acceleration = np.array([.0, .0])

    def get_state(self):
        pose = super(SimpleAccelerationControlKilobot, self).get_state()
        return pose + tuple(self._velocity)

    def set_action(self, action):
        if action is not None:
            action = np.minimum(action, self.action_space.high)
            action = np.maximum(action, self.action_space.low)
            self._acceleration = action
        else:
            self._acceleration = np.array([.0, .0])

    def get_action(self):
        return self._acceleration

    def step(self, time_step):
        self._velocity += self._acceleration * time_step

        self._velocity = np.maximum(self._velocity, [.0, -self._max_angular_velocity])
        self._velocity = np.minimum(self._velocity, [self._max_linear_velocity, self._max_angular_velocity])

        super(SimpleAccelerationControlKilobot, self).step(time_step)


class PhototaxisKilobot(Kilobot):
    def __init__(self, world, position=None, orientation=None, light=None, damping=0.9):
        super(PhototaxisKilobot, self).__init__(world=world, position=position, orientation=orientation, light=light)

        self.__light_measurement = 0
        self.__threshold = -np.inf
        self.__last_update = .0
        self.__update_interval = 2
        self.__update_counter = 0
        self.__no_change_counter = 0
        self.__no_change_threshold = 40

        # The "damping" factor scales the control signal derived from the light gradient.
        # When damping is closer to 1, the kilobot uses almost the full strength of the sensed gradient,
        # leading to more vigorous and rapid movements toward or away from the light.
        # When damping is closer to 0, the control signal is significantly reduced,
        # resulting in slower, less responsive adjustments.
        self.damping = damping

    def _setup(self):
        self.turn_left()
        

    def step(self, time_step):
        """
        Executes one simulation step, updating the kilobot's position and orientation.
        
        This method handles the physics simulation of the kilobot's movement based on
        motor signals. It calculates the resulting linear and angular velocities
        and applies them to the Box2D physics body.
        
        Parameters:
            time_step (float): The duration of the simulation step in seconds.
        """
        # Execute the kilobot's decision-making logic
        self._loop()

        # Calculate the orientation vectors for movement direction
        cos_dir = np.cos(self.get_orientation())  # Cosine of orientation angle
        sin_dir = np.sin(self.get_orientation())  # Sine of orientation angle

        # Initialize velocity variables
        linear_velocity = [.0, .0]  # Initial linear velocity is zero
        angular_velocity = .0  # Initial angular velocity is zero

        # These factors scale the motor signals to appropriate velocities
        # Higher values reduce the effect of motor signals (slower movement)
        both_motors_fudge_factor = 900.  # Scaling factor when both motors are active og 500
        single_motor_fudge_factor = 400.  # Scaling factor when only one motor is active og 250

        # Compute kilobot movement based on motor states
        if self._motor_left and self._motor_right:
            # BOTH MOTORS ON: Move forward with possible rotation
            
            # Calculate linear velocity magnitude based on sum of motor signals
            linear_velocity = ((self._motor_right + self._motor_left) / both_motors_fudge_factor *
                               self._max_linear_velocity)
            
            # Convert scalar velocity to vector in kilobot's forward direction
            linear_velocity = [sin_dir * linear_velocity, cos_dir * linear_velocity]

            # Calculate angular velocity based on difference between motors
            # Positive difference (right > left) causes clockwise rotation
            angular_velocity = ((self._motor_right - self._motor_left) / both_motors_fudge_factor *
                                self._max_angular_velocity)

        elif self._motor_right:
            # ONLY RIGHT MOTOR ON: Rotate clockwise around left leg
            
            # Calculate angular velocity based on right motor signal
            angular_velocity = (self._motor_right / single_motor_fudge_factor *
                                self._max_angular_velocity)
                                
            # Calculate total angular displacement for this time step
            angular_displacement = angular_velocity * time_step

            # Create rotation matrix for the angular displacement
            c, s = np.cos(angular_displacement), np.sin(angular_displacement)
            R = [[c, -s], [s, c]]

            # Calculate linear movement resulting from rotation around left leg
            # The body rotates around the left leg, causing the body to translate
            translation = self._leg_left - np.dot(R, self._leg_left)
            
            # Convert local translation to world coordinates and normalize by time
            linear_velocity = self._body.GetWorldVector(translation * _world_scale) / _world_scale / time_step

        elif self._motor_left:
            # ONLY LEFT MOTOR ON: Rotate counter-clockwise around right leg
            
            # Calculate angular velocity (negative for counter-clockwise)
            angular_velocity = -(self._motor_left / single_motor_fudge_factor *
                                 self._max_angular_velocity)
                                 
            # Calculate total angular displacement for this time step
            angular_displacement = angular_velocity * time_step

            # Create rotation matrix for the angular displacement
            c, s = np.cos(angular_displacement), np.sin(angular_displacement)
            R = [[c, -s], [s, c]]

            # Calculate linear movement resulting from rotation around right leg
            # The body rotates around the right leg, causing the body to translate
            translation = self._leg_right - np.dot(R, self._leg_right)
            
            # Convert local translation to world coordinates and normalize by time
            linear_velocity = self._body.GetWorldVector(translation * _world_scale) / _world_scale / time_step

        # Apply calculated velocities to the Box2D physics body
        self._body.angularVelocity = angular_velocity  # Set the angular velocity
        
        # Set the linear velocity, handling both array and scalar cases
        if type(linear_velocity) == np.ndarray:
            # Convert numpy array to Box2D vector and scale to world units
            self._body.linearVelocity = b2Vec2(*linear_velocity.astype(float)) * _world_scale
        else:
            # Handle scalar case (shouldn't typically happen)
            self._body.linearVelocity = linear_velocity * _world_scale

    def _loop(self):
        """
        Implements the kilobot's light-following behavior logic.
        
        This method implements a simple threshold-based phototaxis algorithm:
        1. Only executes at specific intervals to simulate the kilobot's limited processing speed
        2. Measures ambient light intensity at the kilobot's current position
        3. Switches direction either when:
           a. A higher light intensity is detected (moving towards brighter areas)
           b. No improvement is detected for too long (escaping local minima)
        
        This creates a zigzagging pattern that generally moves the kilobot towards light sources.
        """
        # Skip updates until the update interval is reached
        # This creates a slower update rate than the physics simulation
        # NOTE: Let's get rid of this for our training purposes
        if self.__update_counter % self.__update_interval:
            self.__update_counter += 1
            return

        # Increment the update counter
        self.__update_counter += 1

        # Measure the ambient light intensity at the kilobot's position
        self.__light_measurement = self.get_ambientlight()

        # Check if the light measurement exceeds the threshold 
        # OR if no change has occurred for too long (stuck in local minimum)
        if self.__light_measurement > self.__threshold or self.__no_change_counter >= self.__no_change_threshold:
            # Update the threshold to a slightly higher value
            # This creates a "ratcheting" effect where the robot seeks increasing light values
            self.__threshold = self.__light_measurement + .01

            # Switch the direction of movement (left or right)
            # This creates the zigzagging behavior for exploring and climbing light gradients
            self.switch_directions(motor_pwr=255)

            # Reset the no-change counter since we're trying a new direction
            self.__no_change_counter = 0
        else:
            # Increment the no-change counter if no significant change in light is detected
            # This helps detect when the robot is stuck and needs to try a new direction
            self.__no_change_counter += 1

    def update(self):
        gradient = self.sense_light()
        control_signal = np.array(gradient) * self.damping
        self.position += control_signal

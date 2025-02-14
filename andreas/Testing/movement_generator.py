# movement_generator.py
import math

# Screen dimensions (used by both generator and main)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class MovementGenerator:
    def __init__(self, speed=200, sine_amplitude=100, sine_frequency=1):
        """
        speed: pixels per second
        sine_amplitude: amplitude (in pixels) of the sine wave.
        sine_frequency: number of full sine cycles over the horizontal span.
        """
        self.speed = speed
        self.sine_amplitude = sine_amplitude
        self.sine_frequency = sine_frequency
        # List of segment methods. Each segment method accepts a mode:
        #   'length'  => returns the length of the segment
        #   'eval', u => returns (pos, vel) at parameter u in [0,1]
        self.segments = [self._straight_segment, self._circle_segment, self._sine_segment]
        self.current_segment_index = 0
        self.u = 0.0
        self.position = (0, 0)
        self.velocity = (0, 0)
        self.segment_length = 0.0
        self._init_segment()

    def _init_segment(self):
        seg = self.segments[self.current_segment_index]
        self.segment_length = seg('length')
        self.u = 0.0
        self._update_position_and_velocity()

    def update(self, dt):
        """
        Update the projectile state by dt seconds.
        Returns: (position, velocity, segment_changed)
          - position: (x,y) tuple.
          - velocity: (vx,vy) tuple.
          - segment_changed: True if the movement type just changed.
        """
        segment_changed = False
        # Increment u by distance traveled / segment_length.
        du = (self.speed * dt) / self.segment_length
        self.u += du

        # When a segment is complete, move to the next.
        if self.u >= 1.0:
            segment_changed = True
            self.u -= 1.0
            self.current_segment_index = (self.current_segment_index + 1) % len(self.segments)
            self._init_segment()
        else:
            self._update_position_and_velocity()
        return self.position, self.velocity, segment_changed

    def _update_position_and_velocity(self):
        seg = self.segments[self.current_segment_index]
        self.position, self.velocity = seg('eval', self.u)

    def _straight_segment(self, mode, u=0):
        """
        Straight segment: moves from left to right along the horizontal center.
        """
        start = (0, SCREEN_HEIGHT / 2)
        end = (SCREEN_WIDTH, SCREEN_HEIGHT / 2)
        if mode == 'length':
            return SCREEN_WIDTH
        elif mode == 'eval':
            x = start[0] + (end[0] - start[0]) * u
            y = start[1]  # constant
            pos = (x, y)
            vel = (self.speed, 0)
            return pos, vel

    def _circle_segment(self, mode, u=0):
        """
        Circle segment: a full circle centered in the screen.
        """
        center = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        r = 250  # radius in pixels
        if mode == 'length':
            return 2 * math.pi * r
        elif mode == 'eval':
            angle = 2 * math.pi * u
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            pos = (x, y)
            dx_du = -2 * math.pi * r * math.sin(angle)
            dy_du =  2 * math.pi * r * math.cos(angle)
            factor = self.speed / (2 * math.pi * r)
            vel = (dx_du * factor, dy_du * factor)
            return pos, vel

    def _sine_segment(self, mode, u=0):
        """
        Sine segment: x moves from left to right, y oscillates around the vertical center.
        The sine parameters are now configurable.
        """
        amplitude = self.sine_amplitude
        frequency = self.sine_frequency  # number of full cycles in [0,1]
        if mode == 'length':
            # Numerically approximate the length of the sine curve.
            steps = 100
            length = 0.0
            prev_x = SCREEN_WIDTH * 0
            prev_y = SCREEN_HEIGHT / 2 + amplitude * math.sin(2 * math.pi * frequency * 0)
            for i in range(1, steps + 1):
                u_i = i / steps
                x = SCREEN_WIDTH * u_i
                y = SCREEN_HEIGHT / 2 + amplitude * math.sin(2 * math.pi * frequency * u_i)
                dx = x - prev_x
                dy = y - prev_y
                length += math.hypot(dx, dy)
                prev_x, prev_y = x, y
            return length
        elif mode == 'eval':
            x = SCREEN_WIDTH * u
            y = SCREEN_HEIGHT / 2 + amplitude * math.sin(2 * math.pi * frequency * u)
            pos = (x, y)
            dx_du = SCREEN_WIDTH
            dy_du = amplitude * 2 * math.pi * frequency * math.cos(2 * math.pi * frequency * u)
            seg_length = self._sine_segment('length')
            factor = self.speed / seg_length
            vel = (dx_du * factor, dy_du * factor)
            return pos, vel

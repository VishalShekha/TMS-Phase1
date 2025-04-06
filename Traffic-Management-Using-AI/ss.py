import random
import math
import time
import threading
import pygame
import sys
import os

# Default signal times
DEFAULT_RED = 150
DEFAULT_YELLOW = 5
DEFAULT_GREEN = 20
DEFAULT_MINIMUM = 10
DEFAULT_MAXIMUM = 60

# Simulation parameters
SIMULATION_TIME = 300  # Simulation duration in seconds
DETECTION_TIME = 5     # Time at which vehicles will be detected
GAP_STOPPED = 7        # Gap between stopped vehicles
GAP_MOVING = 7         # Gap between moving vehicles
ROTATION_ANGLE = 3     # Angle for turning vehicles

# Vehicle passing times
VEHICLE_TIMES = {
    'car': 2,
    'bike': 1,
    'rickshaw': 2.25,
    'bus': 2.5,
    'truck': 2.5
}

# Vehicle speeds
VEHICLE_SPEEDS = {
    'car': 2.25,
    'bus': 1.8,
    'truck': 1.8,
    'rickshaw': 2,
    'bike': 2.5
}

# Direction mappings
DIRECTION_NUMBERS = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
VEHICLE_TYPES = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike'}

# Screen settings
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Coordinates of start positions
START_X = {
    'right': [0, 0, 0],
    'down': [271, 254, 240],
    'left': [1400, 1400, 1400],
    'up': [200, 210, 225]
}

START_Y = {
    'right': [223, 232, 250],
    'down': [0, 0, 0],
    'left': [300, 285, 268],
    'up': [800, 800, 800]
}

# Coordinates for UI elements
SIGNAL_COORDS = [(590, 340), (675, 260), (770, 430), (675, 510)]
SIGNAL_TIMER_COORDS = [(530, 210), (810, 210), (810, 550), (530, 550)]
VEHICLE_COUNT_COORDS = [(480, 210), (880, 210), (880, 550), (480, 550)]

# Stop lines coordinates
STOP_LINES = {'right': 210, 'down': 220, 'left': 270, 'up': 307}
DEFAULT_STOP = {'right': 200, 'down': 210, 'left': 280, 'up': 317}
STOPS = {'right': [200, 200, 200], 'down': [320, 320, 320], 'left': [810, 810, 810], 'up': [545, 545, 545]}

# Midpoint coordinates for turns
MID_POINTS = {
    'right': {'x': 705, 'y': 445},
    'down': {'x': 695, 'y': 450},
    'left': {'x': 695, 'y': 425},
    'up': {'x': 695, 'y': 400}
}

class TrafficSignal:
    """Class representing a traffic signal with its timing parameters."""
    
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signal_text = "30"
        self.total_green_time = 0

class Vehicle(pygame.sprite.Sprite):
    """Class representing a vehicle in the simulation."""
    
    def __init__(self, lane, vehicle_class, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicle_class = vehicle_class
        self.speed = VEHICLE_SPEEDS[vehicle_class]
        self.direction_number = direction_number
        self.direction = direction
        self.x = START_X[direction][lane]
        self.y = START_Y[direction][lane]
        self.crossed = 0
        self.will_turn = will_turn
        self.turned = 0
        self.rotate_angle = 0
        self.index = len(vehicles[direction][lane])
        vehicles[direction][lane].append(self)
        
        # Load vehicle image
        image_path = f"images/{direction}/{vehicle_class}.png"
        self.original_image = pygame.image.load(image_path)
        self.current_image = pygame.image.load(image_path)
        
        # Set stop position based on direction and lane
        self._set_stop_position()
        
        # Add to simulation group
        simulation.add(self)
    
    def _set_stop_position(self):
        """Set the stopping position for the vehicle based on its direction and lane."""
        direction = self.direction
        lane = self.lane
        index = self.index
        
        if direction == 'right':
            if index > 0 and vehicles[direction][lane][index-1].crossed == 0:
                self.stop = (vehicles[direction][lane][index-1].stop - 
                            vehicles[direction][lane][index-1].current_image.get_rect().width - 
                            GAP_STOPPED)
            else:
                self.stop = DEFAULT_STOP[direction]
            
            # Update starting positions for next vehicle
            temp = self.current_image.get_rect().width + GAP_STOPPED
            START_X[direction][lane] -= temp
            STOPS[direction][lane] -= temp
            
        elif direction == 'left':
            if index > 0 and vehicles[direction][lane][index-1].crossed == 0:
                self.stop = (vehicles[direction][lane][index-1].stop + 
                            vehicles[direction][lane][index-1].current_image.get_rect().width + 
                            GAP_STOPPED)
            else:
                self.stop = DEFAULT_STOP[direction]
                
            temp = self.current_image.get_rect().width + GAP_STOPPED
            START_X[direction][lane] += temp
            STOPS[direction][lane] += temp
            
        elif direction == 'down':
            if index > 0 and vehicles[direction][lane][index-1].crossed == 0:
                self.stop = (vehicles[direction][lane][index-1].stop - 
                            vehicles[direction][lane][index-1].current_image.get_rect().height - 
                            GAP_STOPPED)
            else:
                self.stop = DEFAULT_STOP[direction]
                
            temp = self.current_image.get_rect().height + GAP_STOPPED
            START_Y[direction][lane] -= temp
            STOPS[direction][lane] -= temp
            
        elif direction == 'up':
            if index > 0 and vehicles[direction][lane][index-1].crossed == 0:
                self.stop = (vehicles[direction][lane][index-1].stop + 
                            vehicles[direction][lane][index-1].current_image.get_rect().height + 
                            GAP_STOPPED)
            else:
                self.stop = DEFAULT_STOP[direction]
                
            temp = self.current_image.get_rect().height + GAP_STOPPED
            START_Y[direction][lane] += temp
            STOPS[direction][lane] += temp
    
    def render(self, screen):
        """Render the vehicle on the screen."""
        screen.blit(self.current_image, (self.x, self.y))
    
    def move(self):
        """Handle vehicle movement based on direction, traffic signals, and turning."""
        direction = self.direction
        
        # Check if vehicle has crossed the stop line
        if self.crossed == 0:
            if ((direction == 'right' and self.x + self.current_image.get_rect().width > STOP_LINES[direction]) or
                (direction == 'down' and self.y + self.current_image.get_rect().height > STOP_LINES[direction]) or
                (direction == 'left' and self.x < STOP_LINES[direction]) or
                (direction == 'up' and self.y < STOP_LINES[direction])):
                self.crossed = 1
                vehicles[direction]['crossed'] += 1
        
        # Handle movement based on direction
        if direction == 'right':
            self._move_right()
        elif direction == 'down':
            self._move_down()
        elif direction == 'left':
            self._move_left()
        elif direction == 'up':
            self._move_up()
    
    def _can_move_forward(self, direction, green_signal):
        """Check if vehicle can move forward based on signals and position."""
        if self.crossed == 1:
            return True
        if green_signal:
            return True
        if ((direction == 'right' and self.x + self.current_image.get_rect().width <= self.stop) or
            (direction == 'down' and self.y + self.current_image.get_rect().height <= self.stop) or
            (direction == 'left' and self.x >= self.stop) or
            (direction == 'up' and self.y >= self.stop)):
            return True
        return False
    
    def _has_gap_to_next_vehicle(self, direction):
        """Check if there's enough gap to the vehicle ahead."""
        if self.index == 0:
            return True
        
        prev_vehicle = vehicles[direction][self.lane][self.index-1]
        
        if prev_vehicle.turned == 1:
            return True
        
        if direction == 'right':
            return self.x + self.current_image.get_rect().width < prev_vehicle.x - GAP_MOVING
        elif direction == 'down':
            return self.y + self.current_image.get_rect().height < prev_vehicle.y - GAP_MOVING
        elif direction == 'left':
            return self.x > prev_vehicle.x + prev_vehicle.current_image.get_rect().width + GAP_MOVING
        elif direction == 'up':
            return self.y > prev_vehicle.y + prev_vehicle.current_image.get_rect().height + GAP_MOVING
    
    def _move_right(self):
        """Handle movement for vehicles going right."""
        if self.will_turn == 1:
            if self.crossed == 0 or self.x + self.current_image.get_rect().width < MID_POINTS[self.direction]['x']:
                # Move forward until mid-point if not crossed or not reached mid-point
                if (self._can_move_forward('right', current_green == 0 and current_yellow == 0) and 
                    self._has_gap_to_next_vehicle('right')):
                    self.x += self.speed
            else:
                # Handle turning
                if self.turned == 0:
                    self.rotate_angle += ROTATION_ANGLE
                    self.current_image = pygame.transform.rotate(self.original_image, -self.rotate_angle)
                    self.x += 2
                    self.y += 1.8
                    if self.rotate_angle == 90:
                        self.turned = 1
                else:
                    # Continue moving in new direction after turn
                    if self.index == 0 or self.y + self.current_image.get_rect().height < (
                            vehicles[self.direction][self.lane][self.index-1].y - GAP_MOVING) or (
                            self.x + self.current_image.get_rect().width < (
                            vehicles[self.direction][self.lane][self.index-1].x - GAP_MOVING)):
                        self.y += self.speed
        else:
            # Straight movement
            if (self._can_move_forward('right', current_green == 0 and current_yellow == 0) and 
                self._has_gap_to_next_vehicle('right')):
                self.x += self.speed
    
    def _move_down(self):
        """Handle movement for vehicles going down."""
        if self.will_turn == 1:
            if self.crossed == 0 or self.y + self.current_image.get_rect().height < MID_POINTS[self.direction]['y']:
                if (self._can_move_forward('down', current_green == 1 and current_yellow == 0) and 
                    self._has_gap_to_next_vehicle('down')):
                    self.y += self.speed
            else:
                if self.turned == 0:
                    self.rotate_angle += ROTATION_ANGLE
                    self.current_image = pygame.transform.rotate(self.original_image, -self.rotate_angle)
                    self.x -= 2.5
                    self.y += 2
                    if self.rotate_angle == 90:
                        self.turned = 1
                else:
                    if (self.index == 0 or 
                        self.x > (vehicles[self.direction][self.lane][self.index-1].x + 
                                vehicles[self.direction][self.lane][self.index-1].current_image.get_rect().width + 
                                GAP_MOVING) or 
                        self.y < (vehicles[self.direction][self.lane][self.index-1].y - GAP_MOVING)):
                        self.x -= self.speed
        else:
            if (self._can_move_forward('down', current_green == 1 and current_yellow == 0) and 
                self._has_gap_to_next_vehicle('down')):
                self.y += self.speed
    
    def _move_left(self):
        """Handle movement for vehicles going left."""
        if self.will_turn == 1:
            if self.crossed == 0 or self.x > MID_POINTS[self.direction]['x']:
                if (self._can_move_forward('left', current_green == 2 and current_yellow == 0) and 
                    self._has_gap_to_next_vehicle('left')):
                    self.x -= self.speed
            else:
                if self.turned == 0:
                    self.rotate_angle += ROTATION_ANGLE
                    self.current_image = pygame.transform.rotate(self.original_image, -self.rotate_angle)
                    self.x -= 1.8
                    self.y -= 2.5
                    if self.rotate_angle == 90:
                        self.turned = 1
                else:
                    if (self.index == 0 or 
                        self.y > (vehicles[self.direction][self.lane][self.index-1].y + 
                                vehicles[self.direction][self.lane][self.index-1].current_image.get_rect().height + 
                                GAP_MOVING) or 
                        self.x > (vehicles[self.direction][self.lane][self.index-1].x + GAP_MOVING)):
                        self.y -= self.speed
        else:
            if (self._can_move_forward('left', current_green == 2 and current_yellow == 0) and 
                self._has_gap_to_next_vehicle('left')):
                self.x -= self.speed
    
    def _move_up(self):
        """Handle movement for vehicles going up."""
        if self.will_turn == 1:
            if self.crossed == 0 or self.y > MID_POINTS[self.direction]['y']:
                if (self._can_move_forward('up', current_green == 3 and current_yellow == 0) and 
                    self._has_gap_to_next_vehicle('up')):
                    self.y -= self.speed
            else:
                if self.turned == 0:
                    self.rotate_angle += ROTATION_ANGLE
                    self.current_image = pygame.transform.rotate(self.original_image, -self.rotate_angle)
                    self.x += 1
                    self.y -= 1
                    if self.rotate_angle == 90:
                        self.turned = 1
                else:
                    if (self.index == 0 or 
                        self.x < (vehicles[self.direction][self.lane][self.index-1].x - 
                                vehicles[self.direction][self.lane][self.index-1].current_image.get_rect().width - 
                                GAP_MOVING) or 
                        self.y > (vehicles[self.direction][self.lane][self.index-1].y + GAP_MOVING)):
                        self.x += self.speed
        else:
            if (self._can_move_forward('up', current_green == 3 and current_yellow == 0) and 
                self._has_gap_to_next_vehicle('up')):
                self.y -= self.speed

def initialize():
    """Initialize traffic signals with default values."""
    # Signal 1
    ts1 = TrafficSignal(0, DEFAULT_YELLOW, DEFAULT_GREEN, DEFAULT_MINIMUM, DEFAULT_MAXIMUM)
    signals.append(ts1)
    
    # Signal 2
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, DEFAULT_YELLOW, DEFAULT_GREEN, 
                        DEFAULT_MINIMUM, DEFAULT_MAXIMUM)
    signals.append(ts2)
    
    # Signal 3
    ts3 = TrafficSignal(DEFAULT_RED, DEFAULT_YELLOW, DEFAULT_GREEN, DEFAULT_MINIMUM, DEFAULT_MAXIMUM)
    signals.append(ts3)
    
    # Signal 4
    ts4 = TrafficSignal(DEFAULT_RED, DEFAULT_YELLOW, DEFAULT_GREEN, DEFAULT_MINIMUM, DEFAULT_MAXIMUM)
    signals.append(ts4)
    
    # Start the main signal cycle
    repeat()

def count_vehicles():
    """Count vehicles for the next green signal to calculate timing."""
    counts = {
        'car': 0,
        'bus': 0,
        'truck': 0,
        'rickshaw': 0,
        'bike': 0
    }
    
    next_direction = DIRECTION_NUMBERS[next_green]
    
    # Count bikes in lane 0
    for vehicle in vehicles[next_direction][0]:
        if vehicle.crossed == 0:
            counts['bike'] += 1
    
    # Count other vehicles in lanes 1 and 2
    for lane in range(1, 3):
        for vehicle in vehicles[next_direction][lane]:
            if vehicle.crossed == 0:
                counts[vehicle.vehicle_class] += 1
    
    return counts

def set_time():
    """Set the time for the next green signal based on vehicle counts."""
    global next_green
    
    # Audio announcement (commented out since it might not work in all environments)
    # os.system(f"say detecting vehicles, {DIRECTION_NUMBERS[(current_green+1)%len(signals)]}")
    
    # Count vehicles and calculate green time
    vehicle_counts = count_vehicles()
    
    # Calculate green time based on vehicle types and their passing times
    green_time = math.ceil(
        (
            (vehicle_counts['car'] * VEHICLE_TIMES['car']) + 
            (vehicle_counts['rickshaw'] * VEHICLE_TIMES['rickshaw']) + 
            (vehicle_counts['bus'] * VEHICLE_TIMES['bus']) + 
            (vehicle_counts['truck'] * VEHICLE_TIMES['truck']) + 
            (vehicle_counts['bike'] * VEHICLE_TIMES['bike'])
        ) / (NO_OF_LANES + 1)
    )
    
    print('Green Time:', green_time)
    
    # Apply minimum and maximum constraints
    green_time = max(DEFAULT_MINIMUM, min(green_time, DEFAULT_MAXIMUM))
    
    # Set the green time for the next signal
    signals[next_green].green = green_time

def print_status():
    """Print the current status of all signals."""
    for i in range(NO_OF_SIGNALS):
        if i == current_green:
            if current_yellow == 0:
                print(f" GREEN TS {i+1} -> r: {signals[i].red} y: {signals[i].yellow} g: {signals[i].green}")
            else:
                print(f"YELLOW TS {i+1} -> r: {signals[i].red} y: {signals[i].yellow} g: {signals[i].green}")
        else:
            print(f"   RED TS {i+1} -> r: {signals[i].red} y: {signals[i].yellow} g: {signals[i].green}")
    print()

def update_values():
    """Update signal timers after each second."""
    for i in range(NO_OF_SIGNALS):
        if i == current_green:
            if current_yellow == 0:
                signals[i].green -= 1
                signals[i].total_green_time += 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

def reset_stops_and_vehicles(direction):
    """Reset stop coordinates for lanes and vehicles."""
    direction_name = DIRECTION_NUMBERS[direction]
    for lane in range(3):
        STOPS[direction_name][lane] = DEFAULT_STOP[direction_name]
        for vehicle in vehicles[direction_name][lane]:
            vehicle.stop = DEFAULT_STOP[direction_name]

def repeat():
    """Main loop for traffic signal rotation."""
    global current_green, current_yellow, next_green
    
    # Green signal phase
    while signals[current_green].green > 0:
        print_status()
        update_values()
        
        # Start vehicle detection when next signal's red time reaches detection time
        if signals[next_green].red == DETECTION_TIME:
            thread = threading.Thread(name="detection", target=set_time, daemon=True)
            thread.start()
        
        time.sleep(1)
    
    # Yellow signal phase
    current_yellow = 1
    vehicle_count_texts[current_green] = "0"
    
    # Reset stops for current direction
    reset_stops_and_vehicles(current_green)
    
    while signals[current_green].yellow > 0:
        print_status()
        update_values()
        time.sleep(1)
    
    # End of cycle, reset signals
    current_yellow = 0
    
    # Reset signal times to default
    signals[current_green].green = DEFAULT_GREEN
    signals[current_green].yellow = DEFAULT_YELLOW
    signals[current_green].red = DEFAULT_RED
    
    # Move to next signal
    current_green = next_green
    next_green = (current_green + 1) % NO_OF_SIGNALS
    
    # Set red time for the signal after next
    signals[next_green].red = signals[current_green].yellow + signals[current_green].green
    
    # Continue cycle
    repeat()

def generate_vehicles():
    """Generate vehicles randomly throughout the simulation."""
    while True:
        # Determine vehicle type, lane, and turning behavior
        vehicle_type = random.randint(0, 4)
        
        # Bikes go in lane 0, others in lanes 1 or 2
        lane_number = 0 if vehicle_type == 4 else random.randint(1, 2)
        
        # Determine if vehicle will turn (only vehicles in lane 2 can turn)
        will_turn = 0
        if lane_number == 2:
            will_turn = 1 if random.randint(0, 4) <= 2 else 0
        
        # Determine direction (weighted probabilities)
        direction_probs = [400, 400, 100, 100]  # Right, down, left, up
        temp = random.randint(0, sum(direction_probs) - 1)
        
        direction_number = 0
        cumulative = 0
        for i, prob in enumerate(direction_probs):
            cumulative += prob
            if temp < cumulative:
                direction_number = i
                break
        
        # Create vehicle
        Vehicle(
            lane_number,
            VEHICLE_TYPES[vehicle_type],
            direction_number,
            DIRECTION_NUMBERS[direction_number],
            will_turn
        )
        
        # Wait before generating next vehicle
        time.sleep(0.25)

def track_simulation_time():
    """Track elapsed time and output statistics at the end."""
    global time_elapsed
    
    while True:
        time_elapsed += 1
        time.sleep(1)
        
        if time_elapsed == SIMULATION_TIME:
            # Calculate and print final statistics
            total_vehicles = 0
            print('Lane-wise Vehicle Counts')
            
            for i in range(NO_OF_SIGNALS):
                direction = DIRECTION_NUMBERS[i]
                count = vehicles[direction]['crossed']
                print(f'Lane {i+1}: {count}')
                total_vehicles += count
            
            print(f'Total vehicles passed: {total_vehicles}')
            print(f'Total time passed: {time_elapsed}')
            print(f'No. of vehicles passed per unit time: {float(total_vehicles)/float(time_elapsed):.2f}')
            
            os._exit(0)

class Simulation:
    """Main simulation class that handles pygame initialization and rendering."""
    
    def __init__(self):
        pygame.init()
        
        # Set up display
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("TRAFFIC SIMULATION")
        
        # Load images
        self.background = pygame.image.load('first.png')
        self.red_signal = pygame.image.load('images/signals/red.png')
        self.yellow_signal = pygame.image.load('images/signals/yellow.png')
        self.green_signal = pygame.image.load('images/signals/green.png')
        
        # Set up font
        self.font = pygame.font.Font(None, 30)
        
        # Start threads
        self._start_threads()
        
        # Main simulation loop
        self._main_loop()
    
    def _start_threads(self):
        """Start all simulation threads."""
        # Thread to track simulation time
        time_thread = threading.Thread(name="simulationTime", target=track_simulation_time, daemon=True)
        time_thread.start()
        
        # Thread to initialize signals
        init_thread = threading.Thread(name="initialization", target=initialize, daemon=True)
        init_thread.start()
        
        # Thread to generate vehicles
        vehicle_thread = threading.Thread(name="generateVehicles", target=generate_vehicles, daemon=True)
        vehicle_thread.start()
    
    def _main_loop(self):
        """Main rendering loop."""
        while True:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            
            # Draw background
            self.screen.blit(self.background, (0, 0))
            
            # Draw signals
            self._draw_signals()
            
            # Draw time elapsed
            time_text = self.font.render(f"Time Elapsed: {time_elapsed}", True, BLACK, WHITE)
            self.screen.blit(time_text, (1100, 50))
            
            # Draw vehicles
            for vehicle in simulation:
                self.screen.blit(vehicle.current_image, [vehicle.x, vehicle.y])
                vehicle.move()
            
            pygame.display.update()
    
    def _draw_signals(self):
        """Draw traffic signals and their timers."""
        for i in range(NO_OF_SIGNALS):
            # Determine signal color and text
            if i == current_green:
                if current_yellow == 1:
                    signal_text = signals[i].yellow if signals[i].yellow > 0 else "STOP"
                    self.screen.blit(self.yellow_signal, SIGNAL_COORDS[i])
                else:
                    signal_text = signals[i].green if signals[i].green > 0 else "SLOW"
                    self.screen.blit(self.green_signal, SIGNAL_COORDS[i])
            else:
                if signals[i].red <= 10:
                    signal_text = signals[i].red if signals[i].red > 0 else "GO"
                else:
                    signal_text = "---"
                self.screen.blit(self.red_signal, SIGNAL_COORDS[i])
            
            # Draw signal timer text
            timer_text = self.font.render(str(signal_text), True, WHITE, BLACK)
            self.screen.blit(timer_text, SIGNAL_TIMER_COORDS[i])
            
            # Draw vehicle count
            count_text = self.font.render(str(vehicles[DIRECTION_NUMBERS[i]]['crossed']), True, BLACK, WHITE)
            self.screen.blit(count_text, VEHICLE_COUNT_COORDS[i])

# Global variables
signals = []
NO_OF_SIGNALS = 4
NO_OF_LANES = 2

current_green = 0  # Index of the signal that is currently green
next_green = 1     # Index of the next signal to turn green
current_yellow = 0  # Flag indicating if current signal is yellow (1) or not (0)

time_elapsed = 0   # Time elapsed since simulation start

# Initialize vehicle data structure
vehicles = {
    'right': {0: [], 1: [], 2: [], 'crossed': 0},
    'down': {0: [], 1: [], 2: [], 'crossed': 0},
    'left': {0: [], 1: [], 2: [], 'crossed': 0},
    'up': {0: [], 1: [], 2: [], 'crossed': 0}
}

# Vehicle count texts
vehicle_count_texts = ["0", "0", "0", "0"]

# Initialize pygame sprite group
pygame.init()
simulation = pygame.sprite.Group()

# Start simulation
if __name__ == "__main__":
    Simulation()
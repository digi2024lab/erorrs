from abc import ABC, abstractmethod 
import math
import threading
import carla
import cv2
import numpy as np
import torch
import pygame
import random
from params_pool2 import ParamsPool
from replay_buffer import ReplayBuffer, Transition

# Training parameters
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
BATCH_SIZE = 64
LOAD_MODEL = False  # Set to True if you want to load a saved model

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sensor(ABC):
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.sensor = None
        self.history = []

    @abstractmethod
    def listen(self):
        pass

    def clear_history(self):
        self.history.clear()

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except RuntimeError as e:
                print(f"Error destroying sensor: {e}")
    
    def get_history(self):
        return self.history

class CollisionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        
    def _on_collision(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_collision)

class LaneInvasionSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)

    def _on_lane_invasion(self, event):
        self.history.append(event)

    def listen(self):
        self.sensor.listen(self._on_lane_invasion)        

class GnssSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world):
        super().__init__(vehicle)
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        self.current_gnss = None

    def _on_gnss_event(self, event):
        self.current_gnss = event

    def listen(self):
        self.sensor.listen(self._on_gnss_event)
    
    def get_current_gnss(self):
        return self.current_gnss

class CameraSensor(Sensor):
    def __init__(self, vehicle, blueprint_library, world, image_processor_callback):
        super().__init__(vehicle)
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.image_processor_callback = image_processor_callback

    def listen(self):
        self.sensor.listen(self.image_processor_callback)

class CarlaEnv:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.client.load_world('Town01')
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None

        self.image_lock = threading.Lock()
        self.running = True

        # Initialize Pygame and set up display
        pygame.init()
        self.display = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("CARLA Semantic Segmentation")

        # Synchronous mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        self.latest_image = None
        self.agent_image = None
        self.display_surface = None

        self.reset_environment()

    def reset_environment(self):
        self._clear_sensors()

        # Destroy vehicle if it exists
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_sensor = None

        # Spawn vehicle at random spawn point
        vehicle_bp = self.blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
        self.spawn_point = self.spawn_points[0]
        self.spawn_rotation = self.spawn_point.rotation
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        # Attach sensors
        self.setup_sensors()

        # Wait for sensors to initialize
        for _ in range(10):
            self.world.tick()

    def setup_sensors(self):
        # Camera sensor
        self.camera_sensor = CameraSensor(self.vehicle, self.blueprint_library, self.world, self.process_image)
        self.camera_sensor.listen()

        # Collision sensor
        self.collision_sensor = CollisionSensor(self.vehicle, self.blueprint_library, self.world)
        self.collision_sensor.listen()

        # Lane invasion sensor
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.blueprint_library, self.world)
        self.lane_invasion_sensor.listen()

        # GNSS sensor
        self.gnss_sensor = GnssSensor(self.vehicle, self.blueprint_library, self.world)
        self.gnss_sensor.listen()

    def _clear_sensors(self):
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor is not None:
            self.gnss_sensor.destroy()
        self.latest_image = None
        self.agent_image = None

    def process_image(self, image):
        # Convert image for display
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb_array = array[:, :, :3]  # Extract RGB channels

        with self.image_lock:
            self.latest_image = rgb_array.copy()

            # Get labels for the agent
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            labels = array[:, :, 2]  # Extract labels from the red channel

            # Normalize labels
            self.agent_image = labels / 22.0  # Normalize to [0, 1]

            # Create Pygame surface for display
            self.display_surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))

    def get_vehicle_speed(self):
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def process_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def render_display(self):
        with self.image_lock:
            if self.display_surface is not None:
                self.display.blit(self.display_surface, (0, 0))
                pygame.display.flip()

    def destroy(self):
        # Restore original settings
        self.world.apply_settings(self.original_settings)

        # Clean up actors
        if self.camera_sensor is not None:
            self.camera_sensor.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()
        if self.gnss_sensor is not None:
            self.gnss_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        pygame.quit()  # Quit Pygame

def train_agent(env, agent, replay_buffer, num_episodes=1000, max_steps_per_episode=1000, batch_size=64):
    try:
        for episode in range(num_episodes):
            if not env.running:
                break
            env.reset_environment()

            direction_vector = env.spawn_rotation.get_forward_vector()

            destination = env.spawn_point.location + direction_vector * 40

            previous_distance = None
            episode_reward = 0
            termination_reason = None

            for step in range(max_steps_per_episode):
                if not env.running:
                    break

                with env.image_lock:
                    current_agent_image = env.agent_image.copy() if env.agent_image is not None else None
                current_gnss = env.gnss_sensor.get_current_gnss()
                if current_agent_image is None or current_gnss is None:
                    env.world.tick()
                    continue

                transform = env.vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                yaw = math.radians(rotation.yaw)

                map = env.world.get_map()
                waypoint = map.get_waypoint(location)
                next_waypoints = waypoint.next(2.0)
                if next_waypoints:
                    next_waypoint = next_waypoints[0]
                else:
                    next_waypoint = waypoint

                wp_location = next_waypoint.transform.location
                dx = wp_location.x - location.x
                dy = wp_location.y - location.y
                desired_yaw = math.atan2(dy, dx)
                epsilon = desired_yaw - yaw
                epsilon = (epsilon + math.pi) % (2 * math.pi) - math.pi

                lateral_deviation = -math.sin(yaw) * dx + math.cos(yaw) * dy

                speed = env.get_vehicle_speed()
                distance_to_destination = location.distance(destination)

                if previous_distance is None:
                    previous_distance = distance_to_destination

                scalars = np.array([distance_to_destination, lateral_deviation])

                action = agent.act(current_agent_image, scalars)

                steer = float(action[0])
                throttle = float(action[1])
                control = carla.VehicleControl(
                    steer=np.clip(steer, -1.0, 1.0),
                    throttle=np.clip(throttle, 0.0, 1.0)
                )
                env.vehicle.apply_control(control)

                env.world.tick()

                with env.image_lock:
                    next_agent_image = env.agent_image.copy() if env.agent_image is not None else None
                transform = env.vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                yaw = math.radians(rotation.yaw)

                speed = env.get_vehicle_speed()

                deviation_threshold = 0.7
                deviation_penalty_scale = 4.0

                if len(env.collision_sensor.get_history()) > 0:
                    reward = -30
                    done = True
                    termination_reason = 'collision'
                elif step >= max_steps_per_episode - 1:
                    reward = -1
                    done = True
                    termination_reason = 'timeout'
                else:
                    if abs(lateral_deviation) <= deviation_threshold:
                        r_lane_centering = 1.0 / (abs(lateral_deviation) + 0.1)
                    else:
                        r_lane_centering = -deviation_penalty_scale * (abs(lateral_deviation) - deviation_threshold)

                    v = speed * 3.6
                    v_target = 20
                    r_speed = 1 - min(1, abs(v - v_target) / 5)

                    r_heading = - (abs(epsilon) / 3) ** 2

                    if distance_to_destination < previous_distance:
                        r_traveled = 1
                    else:
                        r_traveled = -0.1

                    r_overspeed = -5 if v > 25 else 0

                    reward = r_lane_centering + r_speed + r_heading + r_traveled + r_overspeed
                    done = False

                episode_reward += reward

                print(f"Episode {episode+1}, Step {step}, Reward: {reward:.2f}, Total Reward: {episode_reward:.2f}")
                print(f"r_travel {r_traveled}")

                transition = Transition(
                    img=current_agent_image,
                    scalars=scalars,
                    a=action,
                    r=reward,
                    n_img=next_agent_image,
                    n_scalars=scalars,
                    d=done
                )
                replay_buffer.push(transition)

                if step % 20 == 0 and replay_buffer.ready_for(batch_size):
                    batch = replay_buffer.sample(batch_size)
                    agent.update_networks(batch)

                env.process_pygame_events()
                env.render_display()

                if done:
                    break

                previous_distance = distance_to_destination

            print(f'Episode {episode+1}, Total Reward: {episode_reward:.2f}')
            if (episode + 1) % 50 == 0:
                agent.save_model('model_params.pth')
                print(f'Model parameters saved after episode {episode+1}.')

        print('Training completed.')

    finally:
        env.destroy()

def main():
    env = CarlaEnv()
    input_height = 480
    input_width = 640
    feature_dim = 64
    scalar_dim = 2
    action_dim = 2

    agent = ParamsPool(feature_dim=feature_dim,
                       scalar_dim=scalar_dim,
                       action_dim=action_dim,
                       input_height=input_height,
                       input_width=input_width,
                       device=device)
    replay_buffer = ReplayBuffer(capacity=25000, device=device)

    load_model = input("Do you want to load saved model parameters? (y/n): ")
    if load_model.lower() == 'y':
        agent.load_model('model_params.pth')
        print('Model parameters loaded.')

    train_agent(env, agent, replay_buffer,
                num_episodes=NUM_EPISODES,
                max_steps_per_episode=MAX_STEPS_PER_EPISODE,
                batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()

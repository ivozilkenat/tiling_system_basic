import pygame
import sys
import math
import time
import queue
from abc import abstractmethod, ABC
from typing import List, Set, Tuple

# TODO: Math class

# Tiles are typically 16x16 oder 32x32
# Todo: draw tiles in a single call

def print_matrix(matrix):
    print("[")
    for i in matrix:
        print(i)
    print("]")

# Todo: to be tested
# matrix is here defined as a nxn list array
def get_sub_matrix(matrix, start_x, start_y, width, height):
    assert len(matrix[0]) >= start_x + width
    assert len(matrix) >= start_y + height

    sub_matrix = [list() for _ in range(height)]
    for y in range(start_y, start_y + height):
        for x in range(start_x, start_x + width):
            sub_matrix[y-start_y].append(matrix[y][x])

    return sub_matrix


class Vec2:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vec2(self.x * other, self.y * other)
        else:
            raise ValueError("Invalid operand")

    def __str__(self):
        return f"Vec2({self.x}, {self.y})"

    def norm(self):
        absolute = self.abs()
        if absolute == 0:
            return Vec2(0, 0) # TODO: return None?
        return self * (1 / self.abs())

    def abs(self):
        return abs(math.sqrt(self.x**2 + self.y**2))

    def to_tuple(self):
        return self.x, self.y


class Vec3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vec3(self.x * other, self.y * other, self.z - other)
        else:
            raise ValueError("Invalid operand")

    def __str__(self):
        return f"Vec2({self.x}, {self.y}, {self.z})"

    def to_tuple(self):
        return self.x, self.y, self.z

# Take any height and width and make an addressable coordinate system out of that allowing (x, y) selection from the center
# Todo: assert anything about divisibility
class CoordinateSystem:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.origin = Vec2(self.width // 2, self.height // 2)
        self.definition_area = (-self.origin.x, self.width - self.origin.x) # TODO: test this
        self.value_area = (-self.origin.y, self.height - self.origin.y)

    def virtual_to_pos(self, virtual_pos: Vec2):
        res = self.origin + virtual_pos
        assert self.contains_pos(res)
        return res

    # Todo: off by 1 errors might be present in "contains" checking here
    def contains_virtual(self, virtual_pos: Vec2):
        return self.definition_area[0] <= virtual_pos.x <= self.definition_area[1] and self.value_area[0] <= virtual_pos.y <= self.value_area[1]

    def contains_pos(self, pos: Vec2):
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

# TODO: desired: Exact width of playing grid should be provided
# TODO: is auto resizing allowed? => changes grid size

# Base class to represent objects that can be drawn to the screen
class Drawable(ABC):
    def __init__(self, display):
        self.display = display

    @abstractmethod
    def draw(self):
        raise NotImplementedError

# Tiles
class Tile(Drawable):
    def __init__(self, display, x, y, size, color = Vec3(0, 0, 0)):
        super().__init__(display) # If hashing is used, this must be put to the bottom of the init

        self.x: int = x
        self.y: int = y
        self.size: int = size

        self.pos = Vec2(self.x, self.y)
        self.center = Vec2(self.x + self.size // 2, self.y + self.size // 2)

        #self.sprite = None
        self.color: Vec3 = color

        self.draw_border = True

    def draw(self):
        pygame.draw.rect(self.display, self.color.to_tuple(), (self.x, self.y, self.size, self.size))
        if self.draw_border:
            self.draw_borders()

    def draw_borders(self, width = 1):
        pygame.draw.rect(self.display, (0, 0, 0), (self.x, self.y, self.size, self.size), width)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.size == other.size and self.pos == other.pos and self.color == other.color

    def __hash__(self):
        return hash((1/2) * ( self.x + self.y ) * ( self.x + self.y + 1 ) + self.y + self.color.x + self.color.y + self.color.z)

# Collection of Tiles
class TileSet:
    def __init__(self, *args, **kwargs):
        self.tiles = set()

    # TODO: single source of bad performance -> only redraw if update necessary
    def draw(self, border = True):
        for t in self.tiles:
            t.draw()
            if border:
                t.draw_border()

    @staticmethod
    def from_set(tile_set: set):
        ts = TileSet()
        ts.tiles = tile_set
        return ts

# Grid of tiles in matrix
class TileGrid(TileSet):
    def __init__(self, width, height, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height

        self.tile_grid = None

        self.coordinate_system = CoordinateSystem(self.width, self.height)

    def set_grid(self, tile_matrix):
        assert len(tile_matrix) == self.height and len(tile_matrix[0]) == self.width
        self.tile_grid = tile_matrix

    def get_tile(self, pos: Vec2):
        x, y = self.coordinate_system.virtual_to_pos(pos).to_tuple()
        return self.tile_grid[y][x]

    @staticmethod
    def from_matrix(tile_matrix):
        tg = TileGrid(len(tile_matrix[0]), len(tile_matrix))
        tiles = set()
        for y in range(tg.height):
            for x in range(tg.width):
                tiles.add(tile_matrix[y][x])
        tg.tiles = tiles
        tg.set_grid(tile_matrix)
        return tg

# Divide screen into intractable/addressable grid fields
class MapGridManager:
    def __init__(self, screen: pygame.display, map_tile_width, map_tile_height, map_padding=0.1):
        # todo: check if tile height and width is odd?
        # map and border is differentiated
        self.screen = screen

        self.map_padding = map_padding

        self.map_tile_width = map_tile_width
        self.map_tile_height = map_tile_height

        self.update_grid()

    def update_grid(self):
        self._update_dimensions()
        self._generate_tiles()
        self._partition_tiles()

    def _update_dimensions(self):
        self.coordinate_system = CoordinateSystem(self.screen.get_width(), self.screen.get_height())  # TODO: unneeded?

        self.map_screen_width = self.screen.get_width() * (1 - self.map_padding)
        self.map_screen_height = self.screen.get_height() * (1 - self.map_padding)

        possible_tile_size_width = self.map_screen_width // self.map_tile_width
        possible_tile_size_height = self.map_screen_height // self.map_tile_height
        self.tile_size_width = min(possible_tile_size_width, possible_tile_size_height)

        self.map_size_width = self.tile_size_width * self.map_tile_width
        self.map_size_height = self.tile_size_width * self.map_tile_height

        self.border_size_total_width = self.screen.get_width() - self.map_size_width
        self.border_size_total_height = self.screen.get_height() - self.map_size_height

        self.border_tile_width = math.ceil((self.border_size_total_width / 2) / self.tile_size_width)
        self.border_tile_height = math.ceil((self.border_size_total_height / 2) / self.tile_size_width)

        # Borders tiles are added on each side
        self.tile_width = 2 * self.border_tile_width + self.map_tile_width
        self.tile_height = 2 * self.border_tile_height + self.map_tile_height

        self.size_width = self.tile_width * self.tile_size_width
        self.size_height = self.tile_height * self.tile_size_width

        self._tile_corner_offset = Vec2(
            - (self.size_width - self.screen.get_width()) // 2,
            - (self.size_height - self.screen.get_height()) // 2
        )

    def _generate_tiles(self):
        self.grid_tile_matrix = self._setup_grid_tile_matrix()
        self.grid_tiles = self._setup_grid_tile_set()

    def _partition_tiles(self):
        # TODO: handle edge case: self.border_tile_width = 0 and self.border_tile_height = 0
        # ========================
        self.tiles_wall = set(self.grid_tile_matrix[self.border_tile_height - 1][self.border_tile_width - 1:-(self.border_tile_width - 1)] \
               + self.grid_tile_matrix[-(self.border_tile_height)][self.border_tile_width - 1:-(self.border_tile_width - 1)] \
               + [i[self.border_tile_width - 1] for i in self.grid_tile_matrix[self.border_tile_height:-self.border_tile_height]] \
               + [i[-self.border_tile_width] for i in self.grid_tile_matrix[self.border_tile_height:-self.border_tile_height]])
        # ========================
        self.tiles_map = set()
        start_x = self.border_tile_width
        end_x = self.tile_width - self.border_tile_width - 1
        start_y = self.border_tile_height
        end_y = self.tile_height - self.border_tile_height - 1
        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                self.tiles_map.add(self.grid_tile_matrix[y][x])

        self.tiles_map_matrix = get_sub_matrix(
            self.grid_tile_matrix,
            self.border_tile_width, self.border_tile_height,
            self.map_tile_width, self.map_tile_height
        )
        # ========================
        self.tiles_out_of_map = {t for t in self.grid_tiles if (t not in self.tiles_map and t not in self.tiles_wall)}

        # === Set Textures here ===
        # Or use Sprite Manager?

        for t in self.tiles_wall:
            t.color = Vec3(128, 128, 128)

        for t in self.tiles_out_of_map:
            t.color = Vec3(160, 82, 45)

        for t in self.tiles_map:
            t.color = Vec3(255, 235, 205)

        # ========================
        self.tile_set_wall = TileSet.from_set(self.tiles_wall)
        self.tile_set_out_of_map = TileSet.from_set(self.tiles_out_of_map)
        self.tile_grid_map = TileGrid.from_matrix(self.tiles_map_matrix)


    def _setup_grid_tile_matrix(self) -> List[List[Vec2]]:
        grid_matrix = [list() for _ in range(self.tile_height)]
        for y in range(self.tile_height):
            for x in range(self.tile_width):
                tile_pos = Vec2(
                    x * self.tile_size_width,
                    y * self.tile_size_width
                )
                corrected_tile_pos = self._tile_corner_offset + tile_pos
                grid_matrix[y].append(
                    Tile(self.screen, *corrected_tile_pos.to_tuple(), self.tile_size_width)
                )

        return grid_matrix

    def _setup_grid_tile_set(self) -> Set[Tile]:
        tiles = set()
        for l in self.grid_tile_matrix:
            for el in l:
                tiles.add(el)
        return tiles

    def _apply_tile_offset(self, tile_pos: Vec2):
        return tile_pos + self._tile_corner_offset

    # TODO: following methods should be part of a more general 2x2 array/list class (see map tile selection and submatrix code)

    def tile_from_vec(self, virtual_pos: Vec2):
        return self.grid_tile_matrix[virtual_pos.y][virtual_pos.x]

    # Naming fragwürdig
    def get_tile_virtual_pos_by_pos(self, pos: Vec2):
        pos_corrected = pos - self._tile_corner_offset
        x_tile = math.ceil(pos_corrected.x // self.tile_size_width)
        y_tile = math.ceil(pos_corrected.y // self.tile_size_width)
        return Vec2(x_tile, y_tile)

    # TODO: not tested; what about edge cases? e.g. player leaves tile map
    # pos semantically translates to screen pixel pos
    def get_tile_by_pos(self, pos: Vec2):
        return self.tile_from_vec(self.get_tile_virtual_pos_by_pos(pos))

    def get_tiles_by_rect(self, rect):
        upper_left_pos = Vec2(rect[0], rect[1])
        lower_right_pos = Vec2(upper_left_pos.x + rect[2], upper_left_pos.y + rect[3])

        start_x, start_y = self.get_tile_virtual_pos_by_pos(upper_left_pos).to_tuple()
        end_x, end_y = self.get_tile_virtual_pos_by_pos(lower_right_pos).to_tuple()

        tiles = set()

        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                tiles.add(self.grid_tile_matrix[y][x])

        return tiles

# Render anything using proactive update calls for each drawable object
# How to delete something render queue? set seems to do weird stuff
class Renderer:
    def __init__(self):
        self.render_update_queue = queue.Queue()

    def add_to_update_queue(self, obj: Drawable):
        self.render_update_queue.put(obj)

    def render_next_update(self):
        assert not self.render_update_queue.empty()
        self.render_update_queue.get().draw()

    def render_all_updates(self):
        while not self.render_update_queue.empty():
            self.render_next_update()

# Aggregate all game logic and classes
class Game:
    def __init__(self, width, height):
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.grid_manager = MapGridManager(self.screen, *(41, 21), 0.1)
        self.renderer = Renderer()
        self.physics_handler = PhysicsHandler()

        self.running = False

        # === Game Logic

        self.max_speed = 0.6
        self.turn_speed = 0.008
        self.stop_speed = 0.006

        self.player_1 = Player(self.screen, pos=Vec2(100, 100))

        self.dynamic_objects = set()
        self.dynamic_objects.add(self.player_1)

    def run(self): #dispatch into game thread
        self.running = True

        self.clock = pygame.time.Clock()

        # Clear the screen
        self.screen.fill((255, 255, 255))

        self.init_render_update()

        while self.running:
            t0 = time.time()

            dt = self.clock.tick() # TODO: should be given to the physics handler and multiplied into update functions

            self.handle_events(pygame.event.get(), dt)
            self.update_realtime_objs()

            self.physics_handler.update_all(dt)
            self.render()
            # Set the title of the window
            self.set_caption_fps(time.time() - t0)

    def render(self):
        self.renderer.render_all_updates()
        #render_test(self.grid_manager)
        # Update the screen
        pygame.display.flip()

    def update_realtime_objs(self):
        # Get position before, to redraw tiles (bounding rectangle necessary)
        bounding_rect = self.player_1.collision_model.get_bounding_rect() # Extra class for rect?
        player_tiles = self.grid_manager.get_tiles_by_rect(bounding_rect)
        for t in player_tiles:
            t.color = Vec3(0, 255, 0)
            self.renderer.add_to_update_queue(t)

        self.physics_handler.add_to_update_queue(self.player_1.collision_model)
        self.renderer.add_to_update_queue(self.player_1.graphics_model)

    def handle_events(self, events, dt):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.VIDEORESIZE:
                self.screen_resize_handler()

            #if event.type == pygame.KEYDOWN:
            #    self.handle_key_input(event)

        self.handle_key_input(pygame.key.get_pressed(), dt)

    def handle_key_input(self, keys, dt):
        # TODO: automatically update moving objects (dynamic objects) (rendering; physics)

        #print(self.player_1.collision_model.dir)
        #print(self.player_1.collision_model.dir.abs(), self.max_speed)

        # Player 1 handling only currently
        any_key_pressed = False
        if self.player_1.collision_model.dir.abs() < self.max_speed:
            if keys[pygame.K_w]:
                self.player_1.collision_model.dir.y -= self.turn_speed * dt # how to handle x mirroring?
                any_key_pressed = True

            if keys[pygame.K_a]:
                self.player_1.collision_model.dir.x -= self.turn_speed * dt
                any_key_pressed = True

            if keys[pygame.K_s]:
                self.player_1.collision_model.dir.y += self.turn_speed * dt
                any_key_pressed = True

            if keys[pygame.K_d]:
                self.player_1.collision_model.dir.x += self.turn_speed * dt
                any_key_pressed = True

        if not any_key_pressed:
            new_abs = max(0, self.player_1.collision_model.dir.abs() - self.stop_speed * dt)
            self.player_1.collision_model.dir = self.player_1.collision_model.dir.norm() * new_abs



    def set_caption_fps(self, d_time):
        if d_time > 0:
            pygame.display.set_caption(
                f"Grid Testing - {self.screen.get_width()} x {self.screen.get_height()} - FPS: {1 / d_time:.1f}/s")

    # ======

    def screen_resize_handler(self):
        self.grid_manager.update_grid()
        self.init_render_update()

    def init_render_update(self):
        for t in self.grid_manager.grid_tiles:
            self.renderer.add_to_update_queue(t)

# Has to resolve any physics related task during each tick. Includes
# - applying movement
# - solving collisions
# - do that efficiently (check game chunk wise; separate between dynamic and static objects

class PhysicsHandler:
    def __init__(self):
        self.update_queue = queue.Queue()

    def add_to_update_queue(self, obj):
        self.update_queue.put(obj)

    def update_next(self, dt):
        assert not self.update_queue.empty()
        self.update_queue.get().update(dt)

    def update_all(self, dt):
        while not self.update_queue.empty():
            self.update_next(dt)


class GameObj(ABC):
    def __init__(self, *args, **kwargs):
        pass


class CollisionObj(GameObj):
    def __init__(self, pos: Vec2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos = pos

    def on_collide(self, other_collision_obj):
        pass


class StaticCollisionObj(CollisionObj):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DynamicCollisionObj(CollisionObj):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dir = Vec2(1, 0)  # always has to be normalized
        #self.speed = 0

    @abstractmethod
    def update(self, dt):
        raise NotImplementedError


class PlayerCollisionObj(DynamicCollisionObj):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = 20 # diameter of circle
        self.width = self.radius * 2

    def update(self, dt):
        #move_vec = self.dir * self.speed
        move_vec = self.dir
        self.pos = self.pos + move_vec * dt

    # returns (x, y, width, height)
    def get_bounding_rect(self):
        return (self.pos.x - self.radius, self.pos.y - self.radius, self.width, self.width)


class PlayerModel(Drawable):
    def __init__(self, display, player):
        super().__init__(display)
        self.player = player

    def draw(self):
        pygame.draw.circle(self.display, (255, 0, 0), self.player.pos.to_tuple(), self.player.radius)


# # Always updated
# class RealTimeObj(ABC):
#     def __init__(self):
#         pass
#
#     @abstractmethod
#     def update(self, *args, **kwargs):
#         raise NotImplementedError

class Player:
    def __init__(self, display, pos):
        super().__init__()
        self.collision_model = PlayerCollisionObj(pos)
        self.graphics_model = PlayerModel(display, self.collision_model)


def render_test(grid: MapGridManager):
    test_tile = grid.tile_grid_map.get_tile(Vec2(20, -10))
    test_tile.color = Vec3(255, 0, 0)
    test_tile.draw()

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    # Constants for the window size
    WIDTH, HEIGHT = 1920, 1080
    WINDOW_SCALE = 1
    WIDTH_SCALED, HEIGHT_SCALED = WIDTH * WINDOW_SCALE, HEIGHT * WINDOW_SCALE

    game = Game(WIDTH_SCALED, HEIGHT_SCALED)
    # Main game loop
    game.run()

    pygame.quit()
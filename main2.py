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
# TODO: does smalltalk allow custom hashing for sets?

def rad2deg(angle_in_radians):
    return angle_in_radians * 180 / math.pi

def deg2rad(angle_in_deg):
    return angle_in_deg * math.pi / 180

def pair_hash(x, y):
    return hash((1 / 2) * (x + y) * (x + y + 1) + y)


# matrix is here defined as a nxn list array
# TODO: conceptually used several times
def get_sub_matrix(matrix, start_x, start_y, width, height):
    assert len(matrix[0]) >= start_x + width
    assert len(matrix) >= start_y + height

    sub_matrix = [list() for _ in range(height)]
    for y in range(start_y, start_y + height):
        for x in range(start_x, start_x + width):
            sub_matrix[y-start_y].append(matrix[y][x])

    return sub_matrix


# Useful wrapper for nested 2d lists
# TODO: distinguish from matrices in mathematical context
# TODO: NOT PROPERLY USED YET, but rather conceptual; see grid_tile_matrix thingy
# TODO: maybe not necessary
class List2d:
    def __init__(self, nested_container):
        assert all(isinstance(i, list) for i in nested_container)
        if len(nested_container) > 0:
            first_len = len(nested_container[0])
            assert all(len(i) == first_len for i in nested_container)

        self.width = len(nested_container[0])
        self.height = len(nested_container)
        self.matrix = nested_container

        self.coordinate_system = CoordinateSystem(self.width, self.height)

    def get_sub_matrix(self, rect):
        sub_matrix = [list() for _ in range(rect.height)]
        for y in range(rect.y, rect.y + rect.height):
            for x in range(rect.x, rect.x + rect.width):
                sub_matrix[y - rect.y].append(self.matrix[y][x])

        return sub_matrix

    def __iter__(self):
        return iter(self.matrix)

    def __str__(self):
        str = ""
        str += "[\n"
        for i in self.matrix:
            str += f"{i},\n"
        str += "]\n"
        return str


# Coordinate system is directly mapped to matrix like structure (e.g. list of lists) for access



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

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"Vec2({self.x}, {self.y})"

    def __hash__(self):
        return hash(pair_hash(self.x, self.y))

    def abs(self):
        return abs(math.sqrt(self.x**2 + self.y**2))

    def to_tuple(self):
        return self.x, self.y

    def is_null(self):
        return self.x == 0 and self.y == 0

    def scalar_product(self, other):
        return self.x * other.x + self.y * other.y

    def inner_angle(self, other):
        return math.acos(self.scalar_product(other) / (self.abs() * other.abs()))

    def get_normal(self):
        return Vec2.as_unit(Vec2(self.y, -self.x))

    def get_counter(self):
        return Vec2(-self.x , -self.y)

    def to_unit(self):
        absolute = self.abs()
        if absolute == 0:
            return
        self.x = self.x * (1 / absolute)
        self.y = self.y * (1 / absolute)

    @staticmethod
    def as_unit(vec):
        absolute = vec.abs()
        if absolute == 0:
            return Vec2(0, 0)  # TODO: return None?
        return vec * (1 / absolute)


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

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def to_tuple(self):
        return self.x, self.y, self.z


class Straight:
    # Necessary for floating point errors
    DECIMAL_PRECISION = 5 # TODO: MOVE TO GLOBAL

    def __init__(self, support_vector: Vec2, dir_vector: Vec2):
        assert not dir_vector.is_null()
        self.support_vec = support_vector
        self.dir_vec = dir_vector

    def __eq__(self, other):
        return self.collinear_to(other) and self.on_straight(other.support_vec)

    def get_point(self, x):
        return self.support_vec + self.dir_vec * x

    def get_x(self, point: Vec2):
        delta_x = point.x - self.support_vec.x
        delta_y = point.y - self.support_vec.y

        if self.dir_vec.x == 0:
            if round(delta_x, self.DECIMAL_PRECISION) != 0:
                return None
            r1 = None # can be anything
        else:
            r1 = delta_x / self.dir_vec.x

        if self.dir_vec.y == 0:
            if round(delta_y, self.DECIMAL_PRECISION) != 0:
                return None
            r2 = None
        else:
            r2 = delta_y / self.dir_vec.y

        if r1 is None and r2 is None:
            return 0
        elif r1 is None:
            return r2
        elif r2 is None:
            return r1

        return r1 if round(r1 - r2, self.DECIMAL_PRECISION) == 0 else None

    # Maybe static?
    def on_straight(self, point: Vec2):
        return True if self.get_x(point) is not None else False

    # Also return stretch factor?
    def collinear_to(self, other_straight):
        if self.dir_vec.x == 0:
            return other_straight.dir_vec.x == 0
        elif self.dir_vec.y == 0:
            return other_straight.dir_vec.y == 0

        return round(other_straight.dir_vec.x / self.dir_vec.x - other_straight.dir_vec.y / self.dir_vec.y, self.DECIMAL_PRECISION) == 0

    # Returns 1 point (1 intersection), straight (identical), None (no intersection)
    def intersection_with_straight(self, other_straight):
        # - check for collinearity
        if self.collinear_to(other_straight):
            # - check identical
            if self == other_straight:
                return self
            return None

        ### TODO: add case distinction

        delta_x = other_straight.support_vec.x - self.support_vec.x
        delta_y = other_straight.support_vec.y - self.support_vec.y

        denominator = self.dir_vec.y * other_straight.dir_vec.x - self.dir_vec.x * other_straight.dir_vec.y

        # TODO: denominator only 0 if straights parallel?
        s = (delta_y * self.dir_vec.x - delta_x * self.dir_vec.y)/denominator

        if self.dir_vec.y == 0:
            if other_straight.dir_vec.y == 0:
                return other_straight.get_point(s) if delta_y == 0 else None
            return other_straight.get_point(s) if 0 == round(s + delta_y / other_straight.dir_vec.y) else None

        r = (delta_y + s * other_straight.dir_vec.y)/self.dir_vec.y
        # Check first equation
        return self.get_point(r) if 0 == round(r * self.dir_vec.x - s * other_straight.dir_vec.x - delta_x, self.DECIMAL_PRECISION) else None

    def get_reflection_angle(self, vec: Vec2):
        norm_vec = self.dir_vec.get_normal()
        a = vec.inner_angle(norm_vec)
        return a if a <= math.pi / 2 else math.pi - a

    def get_reflection_vec(self, vec: Vec2):
        norm_vec = self.dir_vec.get_normal()
        norm_vec = norm_vec if vec.scalar_product(norm_vec) >= 0 else norm_vec.get_counter()

        return vec - norm_vec * 2 * (vec.scalar_product(norm_vec) / norm_vec.abs())

    def get_parallel_component(self, vec: Vec2):
        dir_vec = self.dir_vec
        scalar_prod = dir_vec.scalar_product(vec)
        if scalar_prod <= 0:
            dir_vec = self.dir_vec.get_counter()
            scalar_prod = -scalar_prod

        return self.dir_vec * (self.dir_vec.scalar_product(vec) / self.dir_vec.abs()**2)


# Maybe as Straight sublass changing all methods to check if on stretch,
# since stretch is basically straight but with special check if point on stretch interval
class Stretch:
    DECIMAL_PRECISION = 5

    def __init__(self, start: Vec2, end: Vec2):
        self.start = start
        self.end = end
        self.straight = Straight(start, end - start)

    def get_x(self, point: Vec2):
        r = self.straight.get_x(point)
        if r is None:
            return None
        return r if 0 <= r <= 1 else None

    def on_stretch(self, point: Vec2):
        return True if self.get_x(point) is not None else False

    def intersection_with_stretch(self, other_stretch):
        res = self.straight.intersection_with_straight(other_stretch.straight)

        if res is None:
            return None
        return res if self.on_stretch(res) and other_stretch.on_stretch(res) else None

    def intersection_with_straight(self, other_straight: Straight):
        res = self.straight.intersection_with_straight(other_straight)
        if res is None:
            return None
        return res if self.on_stretch(res) else None

    def get_len(self):
        return self.straight.dir_vec.abs()

    # Enlarge stretch at both ends with absolute amount
    @staticmethod
    def as_enlarged(stretch, enlargement_amount):
        new_start_point = stretch.start + Vec2.as_unit(stretch.straight.dir_vec) * (-enlargement_amount)
        new_end_point = new_start_point + Vec2.as_unit(stretch.straight.dir_vec) * (stretch.straight.dir_vec.abs() + 2 * enlargement_amount)
        return Stretch(new_start_point, new_end_point)


# TODO: implement these classes where ever used
class Rect:
    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.pos = Vec2(x, y)
        self.center = Vec2(self.x + self.width // 2, self.y + self.height // 2)

    def get_upper_left(self):
        return self.pos

    def get_upper_right(self):
        return self.pos + Vec2(self.width, 0)

    def get_lower_right(self):
        return self.pos + Vec2(self.width, self.height)

    def get_lower_left(self):
        return self.pos + Vec2(0, self.height)


class Circle:
    def __init__(self, x, y, radius):
        self.x, self.y, self.radius = x, y, radius
        self.pos = Vec2(x, y)

    # TODO: use point abstraction or just vectors?
    # negative if straight intersects circle
    def distance_to_straight(self, straight):
        tmp_straight = Straight(self.pos, straight.dir_vec.get_normal())
        intersection = tmp_straight.intersection_with_straight(straight)
        # TODO: None if on straight?
        return (intersection - self.pos).abs() - self.radius

    def distance_to_stretch(self, stretch):
        tmp_straight = Straight(self.pos, stretch.straight.dir_vec.get_normal())
        intersection = stretch.intersection_with_straight(tmp_straight)
        if intersection is None:
            d1 = (stretch.start - self.pos).abs()
            d2 = (stretch.end - self.pos).abs()
            if d1 < d2:
                return d1 - self.radius
            return d2 - self.radius
        return (intersection - self.pos).abs() - self.radius


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
    def __init__(self, display, x, y, size, color=Vec3(0, 0, 0)):
        super().__init__(display) # If hashing is used, this must be put to the bottom of the init
        self.rect = Rect(x, y, size, size) # TODO: as rect or x, y etc as normal members?
        self.size = size
        self.color: Vec3 = color

        # self.sprite = None

        self.draw_border = True

    def draw(self):
        pygame.draw.rect(self.display, self.color.to_tuple(), (self.rect.x, self.rect.y, self.size, self.size))
        if self.draw_border:
            self.draw_borders()

    def draw_borders(self, width=1):
        pygame.draw.rect(self.display, (0, 0, 0), (self.rect.x, self.rect.y, self.size, self.size), width)

    def __eq__(self, other):
        return self.rect.x == other.rect.x and self.rect.y == other.rect.y and self.size == other.size and self.rect.pos == other.rect.pos and self.color == other.color

    def __hash__(self):
        return hash(pair_hash(self.rect.x, self.rect.y) + self.color.z)


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

    # Tiles are returned as 4 tuple, starting with upper left going clockwise
    def get_corner_tiles(self):
        return self.tile_grid[0][0], self.tile_grid[0][self.width-1], self.tile_grid[self.height-1][self.width-1], self.tile_grid[self.height-1][0]

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
        # todo: odd tile width's are not properly tested
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
        self.grid_index_coordinate_system = None

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
        self.tiles_map_and_wall_matrix = get_sub_matrix(
            self.grid_tile_matrix,
            self.border_tile_width - 1, self.border_tile_height - 1,
            self.map_tile_width + 2, self.map_tile_height + 2
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
        self.tile_grid_map_and_wall = TileGrid.from_matrix(self.tiles_map_and_wall_matrix)
        # TODO: also for all tiles?


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

    # TODO: not tested; what about edge cases? e.g. player leaves tile map
    # pos semantically translates to screen pixel pos
    def get_tile_by_pos(self, pos: Vec2):
        return self.tile_from_vec(self.get_tile_virtual_pos_by_pos(pos))

    # Naming fragwürdig
    def get_tile_virtual_pos_by_pos(self, pos: Vec2):
        pos_corrected = pos - self._tile_corner_offset
        x_tile = math.ceil(pos_corrected.x // self.tile_size_width)
        y_tile = math.ceil(pos_corrected.y // self.tile_size_width)
        return Vec2(x_tile, y_tile)

    # TODO: very similar to submatrix code and could also be implemented in TileGrid class
    def get_tiles_by_rect(self, rect):
        upper_left_pos = Vec2(rect[0], rect[1])
        lower_right_pos = Vec2(upper_left_pos.x + rect[2], upper_left_pos.y + rect[3])

        start_x, start_y = self.get_tile_virtual_pos_by_pos(upper_left_pos).to_tuple()
        end_x, end_y = self.get_tile_virtual_pos_by_pos(lower_right_pos).to_tuple()

        tiles = set()

        for y in range(start_y, end_y + 1):

            if not self.tile_pos_y_is_valid(y):
                continue

            for x in range(start_x, end_x + 1):

                if not self.tile_pos_x_is_valid(x):
                    continue

                tiles.add(self.grid_tile_matrix[y][x])

        return tiles

    # TODO: see considerations of get_tiles_by_rect func
    def tile_pos_x_is_valid(self, x):
        return 0 <= x <= self.tile_width - 1

    def tile_pos_y_is_valid(self, y):
        return 0 <= y <= self.tile_height - 1


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
        self.fps_cap = 0 # 0 equals None

        self.grid_manager = MapGridManager(self.screen, *(41, 21), 0.1)
        self.renderer = Renderer()
        self.physics_handler = PhysicsHandler()

        self.running = False

        # === Game Logic
        # TODO: extend barriers slightly
        self.max_speed = 0.6
        self.turn_speed = 0.008
        self.stop_speed = 0.006

        self.player_1 = Player(self.screen, pos=Vec2(100, 100), direction=Vec2(1, 0), speed=1)
        self.physics_handler.physics_objects.add(self.player_1.physics_model)

        self.dynamic_objects = set()
        self.dynamic_objects.add(self.player_1)
        # TODO: how to properly garbarge collect physics handler object list

    def pre_run(self):
        self.init_render_update()
        self.build_map_barrier()

    def run(self): #dispatch into game thread
        self.running = True

        self.clock = pygame.time.Clock()

        # Clear the screen
        self.screen.fill((255, 255, 255))

        self.pre_run()

        while self.running:
            t0 = time.time()

            dt = self.clock.tick(self.fps_cap) # TODO: should be given to the physics handler and multiplied into update functions

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

    # TODO: only temporary
    def update_realtime_objs(self):
        # Get position before, to redraw tiles (bounding rectangle necessary)
        bounding_rect = self.player_1.physics_model.get_bounding_rect() # Extra class for rect?
        player_tiles = self.grid_manager.get_tiles_by_rect(bounding_rect) # TODO: if no player tiles are returned player must have left map
        # assert len(player_tiles) > 0, "player left map"
        for t in player_tiles:
            t.color = Vec3(0, 255, 0)
            self.renderer.add_to_update_queue(t)

        self.physics_handler.add_to_update_queue(self.player_1.physics_model)
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

    # TODO: incorporate proper input handler + Upgrade movement logic
    def handle_key_input(self, keys, dt):
        # TODO: automatically update moving objects (dynamic objects) (rendering; physics)

        #print(self.player_1.collision_model.dir)
        #print(self.player_1.collision_model.dir.abs(), self.max_speed)

        # Player 1 handling only currently
        any_key_pressed = False
        if self.player_1.physics_model.dir.abs() < self.max_speed:
            if keys[pygame.K_w]:
                self.player_1.physics_model.dir.y -= self.turn_speed * dt # how to handle x mirroring?
                any_key_pressed = True

            if keys[pygame.K_a]:
                self.player_1.physics_model.dir.x -= self.turn_speed * dt
                any_key_pressed = True

            if keys[pygame.K_s]:
                self.player_1.physics_model.dir.y += self.turn_speed * dt
                any_key_pressed = True

            if keys[pygame.K_d]:
                self.player_1.physics_model.dir.x += self.turn_speed * dt
                any_key_pressed = True

        if not any_key_pressed:
            new_abs = max(0, self.player_1.physics_model.dir.abs() - self.stop_speed * dt)
            self.player_1.physics_model.dir = Vec2.as_unit(self.player_1.physics_model.dir) * new_abs

    def set_caption_fps(self, d_time):
        if d_time > 0:
            pygame.display.set_caption(
                f"Grid Testing - {self.screen.get_width()} x {self.screen.get_height()} - FPS: {1 / d_time:.1f}/s")

    # ======

    # TODO: old border has to be removed => or "replaced"
    def screen_resize_handler(self):
        self.grid_manager.update_grid()
        self.init_render_update()
        self.remove_map_barrier()
        self.build_map_barrier()

    # Draw everything at least once
    def init_render_update(self):
        for t in self.grid_manager.grid_tiles:
            self.renderer.add_to_update_queue(t)

    def build_map_barrier(self):
        # build map barrier collisions
        upper_left_c, upper_right_c, lower_right_c, lower_left_c = self.grid_manager.tile_grid_map_and_wall.get_corner_tiles()

        # TODO: dont use exact corners
        top_wall = BorderWall(upper_left_c.rect.get_lower_right(), upper_right_c.rect.get_lower_left())
        right_wall = BorderWall(upper_right_c.rect.get_lower_left(), lower_right_c.rect.get_upper_left())
        bottom_wall = BorderWall(lower_right_c.rect.get_upper_left(), lower_left_c.rect.get_upper_right())
        left_wall = BorderWall(lower_left_c.rect.get_upper_right(), upper_left_c.rect.get_lower_right())
        self.barrier_walls = [top_wall, right_wall, bottom_wall, left_wall]

        # Inefficient for deletion / TODO: use set?
        self.physics_handler.physics_objects = self.physics_handler.physics_objects.union(
            {i.physics_model for i in self.barrier_walls}
        )  # TODO: how are objects removed?

    # TODO: solve this better
    def remove_map_barrier(self):
        if self.barrier_walls is None: # barrier walls must be defined in init
            return

        for i in self.barrier_walls:
            self.physics_handler.physics_objects.remove(i.physics_model)

        print(len(self.barrier_walls))


# Has to resolve any physics related task during each tick. Includes
# - applying movement
# - solving collisions
# - do that efficiently (check game chunk wise; separate between dynamic and static objects
class PhysicsHandler:
    def __init__(self):
        self.physics_objects = set() # should be set
        self.update_queue = queue.Queue()

    def add_to_update_queue(self, obj):
        self.update_queue.put(obj)

    def update_next(self, dt):
        assert not self.update_queue.empty()
        update_obj = self.update_queue.get()
        # Run physics
        # Do not update walls for movement and collision
        if update_obj.MOVEMENT_STATIC:
            return

        # Apply changes
        update_obj.update_position(dt)

        # Check if collisions are happening
        # Trivial solution: test against all objects (in the future: use grid based system from tiles)
        for obj in self.physics_objects:
            if obj == update_obj:
                continue

            # TODO: get physics model here? It must be implemented anyways
            # ASSUMPTION: prior position is ALWAYS legal
            CollisionHandler.check_and_resolve(dt, update_obj, obj)


    def update_all(self, dt):
        while not self.update_queue.empty():
            self.update_next(dt)


# Basic conceptual representation of in-game objects
class GameObj(ABC):
    def __init__(self, *args, **kwargs):
        pass


# Objects which follow physical interactions that are visible
class PhysicsObj(GameObj):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_model = None
        self.graphics_model = None


# Implement handler for all geometry models
class CollisionHandler:
    def __init__(self):
        pass

    # Checks if proper collision handler can be found
    @staticmethod
    def check_and_resolve(dt, first, second):
        if CollisionHandler.find_solver_and_run(dt, first, second):
            return True
        elif CollisionHandler.find_solver_and_run(dt, second, first):
            return True
        raise NotImplementedError

    @staticmethod
    def find_solver_and_run(dt, first, second):
        if isinstance(first, CirclePhysicsModel) and isinstance(second, StaticBarrierModel):
            CollisionHandler.dynamic_circle_static_barrier_solver(dt, first, second)
            return True
        return False

    # === PHYSICS SOLVER CODE ===

    @staticmethod
    def dynamic_circle_static_barrier_solver(dt, circle_model, barrier_model):
        # Check collision by either 1.) check if new pos "went through" wall (disable with stuff like teleportation) 2.) check if new pos causes glitching into the barrier

        ### TODO: CONTINUE HERE -> provide feedback / reset circle position
        # Enlarge barrier
        b_stretch = barrier_model.get_stretch()
        tmp_barrier_stretch = Stretch.as_enlarged(
            b_stretch, circle_model.radius
        )

        collision_detected = False

        c_pos_original = circle_model.prior_position()
        c_pos_new = circle_model.pos
        # First check movement path collision otherwise distance

        if c_pos_original != c_pos_new:
            tmp_mvmt_stretch = Stretch(
                c_pos_original,
                c_pos_new
            )
            # Check collision with straight (point refers to intersected point on stretch)
            collision_point = tmp_mvmt_stretch.intersection_with_stretch(tmp_barrier_stretch)
            if collision_point is not None:
                collision_detected = True

        # Check collision by distance
        if not collision_detected:
            c_circle = circle_model.get_circle()
            distance = c_circle.distance_to_stretch(b_stretch)
            if distance <= 0:
                collision_detected = True

        # Resolve collision
        if collision_detected:
            # TODO: How to incorporate "reflection" (do here or in object with handler => reflection is not always wanted)? Add collision handler for physics object?

            # Get intersection

            dir_vec = c_pos_new - c_pos_original
            s_normal = b_stretch.straight.dir_vec.get_normal()
            scalar_prod = dir_vec.scalar_product(s_normal)
            if scalar_prod < 0:
                s_normal = s_normal.get_counter()
                scalar_prod = -scalar_prod

            # check movement direction and normal vector are not orthogonal
            if dir_vec.abs() > 0 and round(scalar_prod, Straight.DECIMAL_PRECISION) != 0:
                tmp_mvmt_straight = Straight(
                    c_pos_original,
                    dir_vec
                )

            else: # edge case if no movement but collision TODO: test (other obj has to suddenly appear in current obj)
                dir_vec = s_normal
                # Choose correct normal vector // if scalar product was 0 this is unkown
                dir_vec = dir_vec if dir_vec.scalar_product(c_pos_original) > 0 else dir_vec.get_counter()
                scalar_prod = dir_vec.scalar_product(dir_vec)
                tmp_mvmt_straight = Straight(
                    c_pos_original,
                    dir_vec
                )

            # Assumption: collision point always exists
            collision_point_stretch = tmp_mvmt_straight.intersection_with_straight(b_stretch.straight)
            if collision_point_stretch is None:
                print("OH NO, there must be an intersection")

            # r + 0.5 as error margin
            circle_model.pos = collision_point_stretch - dir_vec * ((circle_model.radius + 0.5) / scalar_prod)

            # Break object against wall
            circle_model.dir = b_stretch.straight.get_parallel_component(circle_model.dir)

            #!!! Collision MUST be resolved after handling !!!


# Abstract physical representation of dynamic objects that allows collisions
class PhysicsModel(ABC):
    MOVEMENT_STATIC = False # No movement code must be supplied

    def __init__(self, *args, **kwargs):
        pass

    def next_position(self, dt):
        if not self.MOVEMENT_STATIC:
            raise NotImplementedError

    def prior_position(self):
        if not self.MOVEMENT_STATIC:
            raise NotImplementedError

    def update_position(self, dt):
        if not self.MOVEMENT_STATIC:
            raise NotImplementedError

    # TODO: actually needed?
    def revert_position(self):
        if not self.MOVEMENT_STATIC:
            raise NotImplementedError


class PhysicsPointModel(PhysicsModel):
    def __init__(self, position, direction, magnitude, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos = position
        self._pos_prior = None
        self.dir = direction  # Must be normed
        self.mag = magnitude # TODO: not yet used

        self.check_collisions = True
        self.skip_pos_update = False
        # Practical for static objects => also not supposed to be checked in collision calculation

    def next_position(self, dt):
        return self.pos + self.dir * self.mag * dt

    def update_position(self, dt):
        self._pos_prior = self.pos
        self.pos = self.next_position(dt)

    def prior_position(self):
        return self._pos_prior

    def revert_position(self):
        if self._pos_prior is None:
            return
        self.pos = self._pos_prior

    # TODO: change if magnitude is used
    def reset_dir_mag(self):
        self.dir = Vec2(0, 0)


# Basic circle model
class CirclePhysicsModel(PhysicsPointModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = 12  # TODO: should scale with window size
        self.width = self.radius * 2

    def get_bounding_rect(self):
        return self.pos.x - self.radius, self.pos.y - self.radius, self.width, self.width

    def get_circle(self):
        return Circle(self.pos.x, self.pos.y, self.radius)


# Basic building block for 1d walls
class StaticBarrierModel(PhysicsModel):
    MOVEMENT_STATIC = True

    def __init__(self, start, end, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end

    def get_stretch(self):
        return Stretch(self.start, self.end)

    def __hash__(self):
        return hash(hash(self.start) + hash(self.end))


class PlayerPhysicsModel(CirclePhysicsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PlayerModel(Drawable):
    def __init__(self, display, player_physics_model):
        super().__init__(display)
        self.physics_model = player_physics_model

    def draw(self):
        pygame.draw.circle(self.display, (255, 0, 0), self.physics_model.pos.to_tuple(), self.physics_model.radius)


class Player(PhysicsObj):
    def __init__(self, display, pos, direction, speed):
        super().__init__()
        self.physics_model = PlayerPhysicsModel(pos, direction, speed)
        self.graphics_model = PlayerModel(display, self.physics_model)


class BorderWall(PhysicsObj):
    def __init__(self, start, end):
        super().__init__()
        self.physics_model = StaticBarrierModel(start, end)


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    # Constants for the window size
    WIDTH, HEIGHT = 1920, 1080
    WINDOW_SCALE = 0.6
    WIDTH_SCALED, HEIGHT_SCALED = WIDTH * WINDOW_SCALE, HEIGHT * WINDOW_SCALE

    game = Game(WIDTH_SCALED, HEIGHT_SCALED)
    # Main game loop
    game.run()

    pygame.quit()
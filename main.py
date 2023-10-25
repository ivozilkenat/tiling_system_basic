import pygame
import sys
import math
import time
from typing import List


# Tiles are typically 16x16 oder 32x32
# Todo: draw tiles in a single call


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


# Todo: assert anything about divisibility?
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

    def contains_virtual(self, virtual_pos: Vec2):
        return self.definition_area[0] <= virtual_pos.x <= self.definition_area[1] and self.value_area[0] <= virtual_pos.y <= self.value_area[1]

    def contains_pos(self, pos: Vec2):
        return 0 <= pos.x <= self.width and 0 <= pos.y <= self.height

# TODO: desired: Exact width of playing grid should be provided
# TODO: is auto resizing allowed? => changes grid size

class Tile:
    def __init__(self, x, y, size, color = Vec3(0, 0, 0)):
        self.x: int = x
        self.y: int = y
        self.size: int = size

        self.pos = Vec2(self.x, self.y)
        self.center = Vec2(self.x + self.size // 2, self.y + self.size // 2)

        #self.sprite = None
        self.color: Vec3 = color

    def draw(self, screen: pygame.display, border = True):
        pygame.draw.rect(screen, self.color.to_tuple(), (self.x, self.y, self.size, self.size))
        if border:
            self.draw_border(screen)

    def draw_border(self, screen: pygame.display, width = 1):
        pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.size, self.size), width)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.size == other.size and self.pos == other.pos and self.color == other.color

    def __hash__(self):
        return hash((1/2) * ( self.x + self.y ) * ( self.x + self.y + 1 ) + self.y + self.color.x + self.color.y + self.color.z)


class TileSet:
    def __init__(self, *args, **kwargs):
        self.tiles = set()

    # TODO: single source of bad performance -> only redraw if update necessary
    def draw(self, border = True):
        for t in self.tiles:
            t.draw(border)


    @staticmethod
    def from_set(tile_set: set):
        ts = TileSet()
        ts.tiles = tile_set
        return ts


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

    # @staticmethod
    # def from_list(tile_list, x_dim: int, y_dim: int):
    #     assert len(tile_list) == x_dim * y_dim
    #     matrix = [list() for _ in range(y_dim)]
    #     for y in range(y_dim):
    #         for x in range(x_dim):
    #             matrix[y].append(tile_list[y * y_dim + x])
    #     tg = TileGrid(x_dim, y_dim)
    #     tg.tiles = set(tile_list)
    #     tg.set_grid(matrix)
    #     return tg


class GridSystem:
    def __init__(self, screen: pygame.display, min_tile_width_count):
        assert min_tile_width_count % 2 != 0, "Tile count must be odd"
        self.screen = screen

        self.min_tile_width_count = min_tile_width_count

        self.screen_width = None
        self.screen_height = None

        self.coordinate_system = None
        self.tile_width_count = None
        self.tile_width = None
        self.width = None
        self.height = None
        self._tile_corner_offset = None
        self.grid_corner_matrix = None
        self.grid_corners = None

        self.refresh_screen()

    def refresh_screen(self):
        if self.screen.get_width() == self.screen_width and self.screen.get_height() == self.screen_height:
            return

        self.screen_width = self.screen.get_width()
        self.screen_height = self.screen.get_height()

        self.coordinate_system = CoordinateSystem(self.screen_width, self.screen_height)
        self.tile_width_count = self.min_tile_width_count
        self.tile_width = self.coordinate_system.width // self.tile_width_count

        self.width = math.ceil(self.coordinate_system.width / self.tile_width)
        self.height = math.ceil(self.coordinate_system.height / self.tile_width)

        if self.width % 2 == 0:
            self.width += 1
        if self.height % 2 == 0:
            self.height += 1

        self._tile_corner_offset = Vec2(
            - (self.tile_width * self.width - self.screen_width) // 2,
            - (self.tile_width * self.height - self.screen_height) // 2
        )

        self.grid_corner_matrix = self.calc_grid_matrix()
        self.grid_corners = self.calc_grid_corners()

    def tile_to_screen_pos(self, tile_virtual: Vec2):
       # return self.coordinate_system.virtual_to_screen()
        return

    def get_tile_center(self, tile_virtual: Vec2):
        return

    def apply_tile_offset(self, tile_pos: Vec2):
        return tile_pos + self._tile_corner_offset

    def get_ring(self, index) -> List[Vec2]:
        assert index >= 0
        if index > 0:
            return self.grid_corner_matrix[index][index:-index] \
                + self.grid_corner_matrix[-(index + 1)][index:-index] \
                + [i[index] for i in self.grid_corner_matrix[index + 1:-(index + 1)]] \
                + [i[-(index + 1)] for i in self.grid_corner_matrix[index + 1:-(index + 1)]]
        else:
            return self.grid_corner_matrix[0] \
                + self.grid_corner_matrix[-1] \
                + [i[0] for i in self.grid_corner_matrix[1:-1]] \
                + [i[-1] for i in self.grid_corner_matrix[1:-1]]

    def calc_grid_matrix(self) -> List[List[Vec2]]:
        grid_matrix = [list() for _ in range(self.height)]
        for x in range(self.width):
            for y in range(self.height):
                tile_pos = Vec2(
                    x * self.tile_width,
                    y * self.tile_width
                )
                corrected_tile_pos = self.apply_tile_offset(tile_pos)
                grid_matrix[y].append(corrected_tile_pos)

        return grid_matrix

    def calc_grid_corners(self):
        corners = set()
        for l in self.grid_corner_matrix:
            for el in l:
                corners.add(el)
        return corners


class GridSystem2:
    def __init__(self, screen: pygame.display, map_tile_width, map_tile_height, map_padding=0.1):
        # todo: check if tile height and widht is odd?
        # map and border is differentiated
        self.screen = screen

        self.map_padding = map_padding

        self.map_tile_width = map_tile_width
        self.map_tile_height = map_tile_height

        self.screen_width = None
        self.screen_height = None

        self.refresh_screen()

    def refresh_screen(self):
        # Only redraw if screen changed
        if self.screen.get_width() == self.screen_width and self.screen.get_height() == self.screen_height:
            return

        self.screen_width = self.screen.get_width()
        self.screen_height = self.screen.get_height()

        self.coordinate_system = CoordinateSystem(self.screen_width, self.screen_height) # TODO: unneeded?

        self.map_screen_width = self.screen_width * (1 - self.map_padding)
        self.map_screen_height = self.screen_height * (1 - self.map_padding)

        possible_tile_size_width = self.map_screen_width // self.map_tile_width
        possible_tile_size_height = self.map_screen_height // self.map_tile_height
        self.tile_size_width = min(possible_tile_size_width, possible_tile_size_height)

        self.map_size_width = self.tile_size_width * self.map_tile_width
        self.map_size_height = self.tile_size_width * self.map_tile_height

        self.border_size_total_width = self.screen_width - self.map_size_width
        self.border_size_total_height = self.screen_height - self.map_size_height

        self.border_tile_width = math.ceil((self.border_size_total_width / 2) / self.tile_size_width)
        self.border_tile_height = math.ceil((self.border_size_total_height / 2) / self.tile_size_width)

        # Borders tiles are added on each side
        self.tile_width = 2 * self.border_tile_width + self.map_tile_width
        self.tile_height = 2 * self.border_tile_height + self.map_tile_height

        self.size_width = self.tile_width * self.tile_size_width
        self.size_height = self.tile_height * self.tile_size_width

        self._tile_corner_offset = Vec2(
            - (self.size_width - self.screen_width) // 2,
            - (self.size_height - self.screen_height) // 2
        )

        self.grid_corner_matrix = self._calc_grid_matrix()
        self.grid_corners = self._calc_grid_corners()

        self._calc_tile_sets()

    def tile_to_screen_pos(self, tile_virtual: Vec2):
       # return self.coordinate_system.virtual_to_screen()
        return

    def get_tile_center(self, tile_virtual: Vec2):
        return

    def apply_tile_offset(self, tile_pos: Vec2):
        return tile_pos + self._tile_corner_offset

    def _calc_tile_sets(self):
        # TODO: handle edge case: self.border_tile_width = 0 and self.border_tile_height = 0
        self.tiles_wall = set(self.grid_corner_matrix[self.border_tile_height - 1][self.border_tile_width - 1:-(self.border_tile_width - 1)] \
               + self.grid_corner_matrix[-(self.border_tile_height)][self.border_tile_width - 1:-(self.border_tile_width - 1)] \
               + [i[self.border_tile_width - 1] for i in self.grid_corner_matrix[self.border_tile_height:-self.border_tile_height]] \
               + [i[-self.border_tile_width] for i in self.grid_corner_matrix[self.border_tile_height:-self.border_tile_height]])

        self.tiles_map = set()
        start_x = self.border_tile_width
        end_x = self.tile_width - self.border_tile_width - 1
        start_y = self.border_tile_height
        end_y = self.tile_height - self.border_tile_height - 1
        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                self.tiles_map.add(self.grid_corner_matrix[y][x])

        self.tiles_out_of_map = {t for t in self.grid_corners if (t not in self.tiles_map and t not in self.tiles_wall)}

    def _calc_grid_matrix(self) -> List[List[Vec2]]:
        grid_matrix = [list() for _ in range(self.tile_height)]
        for x in range(self.tile_width):
            for y in range(self.tile_height):
                tile_pos = Vec2(
                    x * self.tile_size_width,
                    y * self.tile_size_width
                )
                corrected_tile_pos = self.apply_tile_offset(tile_pos)
                grid_matrix[y].append(corrected_tile_pos)

        return grid_matrix

    def _calc_grid_corners(self):
        corners = set()
        for l in self.grid_corner_matrix:
            for el in l:
                corners.add(el)
        return corners


class GridSystem3:
    def __init__(self, screen: pygame.display, map_tile_width, map_tile_height, map_padding=0.1):
        # todo: check if tile height and widht is odd?
        # map and border is differentiated
        self.screen = screen

        self.map_padding = map_padding

        self.map_tile_width = map_tile_width
        self.map_tile_height = map_tile_height

        self.screen_width = None
        self.screen_height = None

        self.refresh_screen()

    def refresh_screen(self):
        # Only redraw if screen changed
        if self.screen.get_width() == self.screen_width and self.screen.get_height() == self.screen_height:
            return

        self.screen_width = self.screen.get_width()
        self.screen_height = self.screen.get_height()

        self.coordinate_system = CoordinateSystem(self.screen_width, self.screen_height) # TODO: unneeded?

        self.map_screen_width = self.screen_width * (1 - self.map_padding)
        self.map_screen_height = self.screen_height * (1 - self.map_padding)

        possible_tile_size_width = self.map_screen_width // self.map_tile_width
        possible_tile_size_height = self.map_screen_height // self.map_tile_height
        self.tile_size_width = min(possible_tile_size_width, possible_tile_size_height)

        self.map_size_width = self.tile_size_width * self.map_tile_width
        self.map_size_height = self.tile_size_width * self.map_tile_height

        self.border_size_total_width = self.screen_width - self.map_size_width
        self.border_size_total_height = self.screen_height - self.map_size_height

        self.border_tile_width = math.ceil((self.border_size_total_width / 2) / self.tile_size_width)
        self.border_tile_height = math.ceil((self.border_size_total_height / 2) / self.tile_size_width)

        # Borders tiles are added on each side
        self.tile_width = 2 * self.border_tile_width + self.map_tile_width
        self.tile_height = 2 * self.border_tile_height + self.map_tile_height

        self.size_width = self.tile_width * self.tile_size_width
        self.size_height = self.tile_height * self.tile_size_width

        self._tile_corner_offset = Vec2(
            - (self.size_width - self.screen_width) // 2,
            - (self.size_height - self.screen_height) // 2
        )

        self.grid_tile_matrix = self._calc_grid_tile_matrix()
        self.grid_tiles = self._calc_grid_tiles()

        self._calc_tile_sets()

    def apply_tile_offset(self, tile_pos: Vec2):
        return tile_pos + self._tile_corner_offset

    def _calc_tile_sets(self):
        # TODO: handle edge case: self.border_tile_width = 0 and self.border_tile_height = 0
        self.tiles_wall = set(self.grid_tile_matrix[self.border_tile_height - 1][self.border_tile_width - 1:-(self.border_tile_width - 1)] \
               + self.grid_tile_matrix[-(self.border_tile_height)][self.border_tile_width - 1:-(self.border_tile_width - 1)] \
               + [i[self.border_tile_width - 1] for i in self.grid_tile_matrix[self.border_tile_height:-self.border_tile_height]] \
               + [i[-self.border_tile_width] for i in self.grid_tile_matrix[self.border_tile_height:-self.border_tile_height]])

        self.tiles_map = list()
        start_x = self.border_tile_width
        end_x = self.tile_width - self.border_tile_width - 1
        start_y = self.border_tile_height
        end_y = self.tile_height - self.border_tile_height - 1
        for y in range(start_y, end_y + 1):
            for x in range(start_x, end_x + 1):
                self.tiles_map.append(self.grid_tile_matrix[y][x])

        # TODO: not working correctly
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        self.tiles_map_matrix = [list() for _ in range(delta_y)]
        for y in range(delta_y):
            for x in range(delta_x):
                self.tiles_map_matrix[y].append(self.tiles_map[y * delta_x + x])

        self.tiles_out_of_map = {t for t in self.grid_tiles if (t not in self.tiles_map and t not in self.tiles_wall)}

        for t in self.tiles_wall:
            t.color = Vec3(128, 128, 128)

        for t in self.tiles_out_of_map:
            t.color = Vec3(160, 82, 45)

        for t in self.tiles_map:
            t.color = Vec3(255, 235, 205)


        self.tile_set_wall = TileSet.from_set(self.tiles_wall)
        self.tile_set_out_of_map = TileSet.from_set(self.tiles_out_of_map)
        self.tile_grid_map = TileGrid.from_matrix(self.grid_tile_matrix)

        # self.tile_grid_map = TileGrid.from_list(
        #     self.tiles_map,
        #     self.map_tile_width,
        #     self.map_tile_height
        # )

    def _calc_grid_tile_matrix(self) -> List[List[Vec2]]:
        grid_matrix = [list() for _ in range(self.tile_height)]
        for y in range(self.tile_height):
            for x in range(self.tile_width):
                tile_pos = Vec2(
                    x * self.tile_size_width,
                    y * self.tile_size_width
                )
                corrected_tile_pos = self.apply_tile_offset(tile_pos)
                grid_matrix[y].append(
                    Tile(*corrected_tile_pos.to_tuple(), self.tile_size_width)
                )

        return grid_matrix

    def _calc_grid_tiles(self):
        tiles = set()
        for l in self.grid_tile_matrix:
            for el in l:
                tiles.add(el)
        return tiles


def render_test_grid3(screen, grid: GridSystem3):
    grid.tile_set_wall.draw(screen)
    grid.tile_set_out_of_map.draw(screen)
    grid.tile_grid_map.draw(screen)

    for x in range(2):
        test_tile = grid.tile_grid_map.get_tile(Vec2(x, x))
        test_tile.color = Vec3(255, 0, 0)
        test_tile.draw(screen)


def render_test_grid(screen, grid: GridSystem2, width = 0):
    #wall_path = grid.get_ring(0)
    wall_path = grid.tiles_out_of_map
    #wall_stone = grid.get_ring(1)
    wall_stone = grid.tiles_wall
    floor = grid.tiles_map

    brown = Vec3(160, 82, 45)
    gray = Vec3(128, 128, 128)
    floor_color = Vec3(255, 235, 205)

    for c, i in enumerate(wall_path):
        pygame.draw.rect(screen, brown.to_tuple(), (*i.to_tuple(), grid.tile_size_width, grid.tile_size_width), width)

    for c, i in enumerate(wall_stone):
        pygame.draw.rect(screen, gray.to_tuple(), (*i.to_tuple(), grid.tile_size_width, grid.tile_size_width), width)

    for c, i in enumerate(floor):
        pygame.draw.rect(screen, floor_color.to_tuple(), (*i.to_tuple(), grid.tile_size_width, grid.tile_size_width), width)

    for i in grid.grid_corners:
        pygame.draw.rect(screen, (0, 0, 0), (*i.to_tuple(), grid.tile_size_width, grid.tile_size_width), 1)


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    # Constants for the window size
    WIDTH, HEIGHT = 1920, 1080
    WINDOW_SCALE = 1
    WIDTH_SCALED, HEIGHT_SCALED = WIDTH * WINDOW_SCALE, HEIGHT * WINDOW_SCALE

    # Create a window
    screen = pygame.display.set_mode((WIDTH_SCALED, HEIGHT_SCALED), pygame.RESIZABLE)

    grid = GridSystem3(screen, *(41, 21), 0.1)

    WHITE = (255, 255, 255)

    # Main game loop
    running = True
    while running:
        t0 = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill(WHITE)

        # Draw a red rectangle
        grid.refresh_screen()
        render_test_grid3(screen, grid)

        # Update the screen
        pygame.display.flip()

        # Set the title of the window
        pygame.display.set_caption(f"Grid Testing - {screen.get_width()} x {screen.get_height()} - FPS: {1/(time.time() - t0):.1f}/s")


    # Quit Pygame
    pygame.quit()
    sys.exit()
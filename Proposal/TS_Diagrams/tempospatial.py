from typing import Dict, List, Tuple
from PIL import Image
from functools import partial
import numpy as np
import itertools as iter

CACHE_SIZE  = 32
CACHE_WIDTH = 4

# Thread execution params
WIDTH  = 16
HEIGHT = 16

def pattern_stencil(size):
    m = size // 2
    def pattern_internal(x, y):
        return [(a + x - m, b + y - m) for b in range(size) for a in range(size)]
    return pattern_internal

def pattern_matrix():
    def pattern_internal(x, y):
        xs = [(x, y) for x in range(WIDTH)]
        ys = [(x, y + HEIGHT) for y in range(HEIGHT)]
        return list(set(iter.chain(xs, ys)))
    return pattern_internal

def pattern_matrix_stencil(size):
    def pattern_internal(x, y):
        stencil = pattern_stencil(size)(x, y)
        matrix  = pattern_matrix()(x, y)
        return list(set(iter.chain(stencil, matrix)))
    return pattern_internal
    

def remap_column(column_width):
    def remap_internal(ids):
        groups: Dict[int, List[Tuple[int, int]]] = dict()
        for (x, y) in ids:
            g = x // column_width
            if g not in groups:
                groups[g] = []
            groups[g].append((x, y))

        remapped: List[int, int] = []
        for g in groups:
            remapped.extend(groups[g])

        return remapped
    return remap_internal

def remap_tile(tile_size):
    def remap_internal(ids):
        groups: Dict[int, List[Tuple[int, int]]] = dict()
        for (x, y) in ids:
            g = (x // tile_size, y // tile_size)
            if g not in groups:
                groups[g] = []
            groups[g].append((x, y))

        remapped: List[int, int] = []
        for g in groups:
            remapped.extend(groups[g])

        return remapped
    return remap_internal

pattern = pattern_matrix_stencil(5)
#pattern = pattern_stencil(7)
#pattern = pattern_matrix()
thread_ids = [(x, y) for y in range(HEIGHT) for x in range(WIDTH)]
thread_count = len(thread_ids)

#thread_ids = remap_tile(2)(thread_ids)
thread_ids = remap_column(2)(thread_ids)

# Gather addresses
def to_linear_space(x, y):
    return y * WIDTH + x

rs: List[List[int]] = []
ws: List[int] = []
min_addr = 0
max_addr = WIDTH * HEIGHT

for (i, (x, y)) in enumerate(thread_ids):
    r = [to_linear_space(x, y) for (x, y) in pattern(x, y)]
    w = to_linear_space(x, y)
    rs.append(r)
    ws.append(w)

    min_addr = min(min_addr, min(r))
    max_addr = max(max_addr, max(r))

# Remap addresses
for (t_i, (r, w)) in enumerate(zip(rs, ws)):
    for (a_i, addr) in enumerate(r):
        r[a_i] = addr - min_addr
    rs[t_i] = r
    ws[t_i] = w - min_addr
max_addr -= min_addr
min_addr = 0

cache: Dict[int, int] = dict()
total_loads  = 0
cache_misses = 0
cache_evicts = 0

print(f'Thread Layout: {WIDTH}x{HEIGHT} ({thread_count})')
print(f'Memory Range : 0 - {max_addr}')
image = Image.new('RGB', (thread_count, max_addr))
for x in range(thread_count):
    for addr in range(max_addr):
        cache_addr = addr // CACHE_WIDTH
        y = max_addr - addr - 1

        # Uncached memory
        image.putpixel((x, y), (255, 255, 255))

        
        if cache_addr in cache.keys():
            # Cached memory
            image.putpixel((x, y), (128, 128, 128))

        if addr in rs[x]:
            total_loads += 1
            if cache_addr in cache.keys():
                image.putpixel((x, y), (0, 0, 0))
            else:
                cache_misses += 1
                if len(cache.keys()) >= CACHE_SIZE:
                    cache_evicts += 1
                    evict = min(cache.keys(), key=lambda k: cache[k])
                    del cache[evict]
                image.putpixel((x, y), (255, 0, 0))

            cache[cache_addr] = x
        #if ws[x] == addr:
        #    image.putpixel((x, y), (0, 0, 0))
print(f'Loads: {total_loads}\tMisses: {cache_misses}\tEvicts: {cache_evicts}')
print(f'{cache_misses/total_loads:0.3}% Miss\t{cache_evicts/total_loads:0.3}% Evict')

image.save('diagram.png')

import csv
import math

STENCIL      = 7
COLUMNS      = [8, 16, 27, 32, 37, 64, 128]
WIDTH        = 8

RANGE = range(32, 512+1, 4)

def stencil_column(c, i):
    return (STENCIL + i) * math.ceil((STENCIL + c - 1) / WIDTH + 1) * math.ceil(i / c)

def stencil_column_mem(c, i):
    return (STENCIL + c) * (STENCIL + 1)

def stencil_naive(i):
    return math.ceil(i * i / WIDTH) * STENCIL + math.ceil(STENCIL / WIDTH) * STENCIL

def stencil_naive_mem(c):
    return (math.ceil(STENCIL / WIDTH) + 1) * STENCIL

def stencil_opt(i):
    return math.ceil(i / WIDTH)

with open('stencil_loads.csv', 'w', newline='') as file:
    w = csv.writer(file)
    w.writerow(['I', 'naive', 'naive_opt'] + [f'{c}' for c in COLUMNS])
    for i in RANGE:
        w.writerow([i, stencil_naive(i), stencil_opt(i)] + [stencil_column(c, i) for c in COLUMNS])

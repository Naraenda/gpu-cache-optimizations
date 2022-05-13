import subprocess
from typing import List

COLUMN_VALUES = [
    0,
    #4,
    #8,
    #16,
    #30,
    #31,
    32,
    #33,
    #34,
    #48,
    64,
    96,
    128,
    256,
    512,
    1024,
    2048,
]

BLOCK_SIZES = [
    32,
    64,
    128,
    256,
    512,
    1024,
]

INPUT_SIZES = [
    128,
    256,
    512,
    1024,
    2048,
    #4096,
    #8192,
]

print('# Compile')
#subprocess.run('nvcc stencil.cu -arch=sm_72')
subprocess.run('nvcc matrix.cu -arch=sm_72')

def colval_vs_blocksize():
    summary = ['bench']
    for b in BLOCK_SIZES:
        summary.append(f',\tb{b:4}')
    summary.append('\n')

    for c in COLUMN_VALUES:
        n = c if c > 0 else 'naive'
        summary.append(f'{n:5}')
        for b in BLOCK_SIZES:
            f = f'results/stencil_{n}_bs{b}.log'
            print(f"# Benchmarking ({n})")
            subprocess.run(f'nvprof --print-gpu-summary --normalized-time-unit ms --log-file {f} a {c} {b}')
            
            with open(f, 'r') as file:
                data = file.readlines()

            res = f'{n:5} {data[5].strip()}'
            avg = list(x for x in res.split(' ') if x != '')[7].strip('ms')
            summary.append(f',\t{avg:4}')
        
        summary.append('\n')
    with open('results.txt', 'w') as file:
        file.writelines(summary)

def colval_vs_input():
    b = 512

    summary = ['size']
    for c in COLUMN_VALUES:
        summary.append(f',\tc{c}')
    summary.append('\n')

    for i in INPUT_SIZES:
        summary.append(f'{i:5}')
        for c in COLUMN_VALUES:
            avg = 1e32
            for b in BLOCK_SIZES:
                n = c if c > 0 else 'naive'
                f = f'results/stencil_{n}_bs{b}_i{i}.log'
                subprocess.run(f'nvprof --print-gpu-summary --normalized-time-unit ms --log-file {f} a {c} {b} {i}')

                with open(f, 'r') as file:
                    data = file.readlines()

                res = f'{n:5} {data[5].strip()}'
                cur_avg = float(list(x for x in res.split(' ') if x != '')[7].strip('ms'))
                avg = min(avg, cur_avg)
            summary.append(f',\t{avg:4}')
        summary.append('\n')

    with open('results.txt', 'w') as file:
        file.writelines(summary)

# colval_vs_blocksize()
colval_vs_input()
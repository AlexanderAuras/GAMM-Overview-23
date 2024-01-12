import itertools
import sys


params = [
    [("tik_alpha", x) for x in [1e-2, 1e-1, 0.0, 1e0]],
    [("lr", x) for x in [1e-5, 1e-4, 1e-3, 1e-2]],
    [("weight_decay", x) for x in [0.0, 1e-5, 1e-4, 1e-3]],
    [("intern_lr", x) for x in [1e-5, 1e-4, 1e-3, 1e-2]],
]
combis = list(map(lambda x: dict(x), itertools.product(*params)))

if len(sys.argv) < 2:
    print(len(combis))
else:
    [print(f"--{k}={v}", end=" ") for k, v in combis[int(sys.argv[1])].items()]

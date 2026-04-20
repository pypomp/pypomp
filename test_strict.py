from pypomp.core.pomp import Pomp
from pypomp.core.parameters import PompParameters
def foo(p: Pomp, theta: PompParameters):
    p.train(10, 10, {"a": 0.1}, theta=theta)


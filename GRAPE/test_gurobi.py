import gurobipy as gb
import numpy as np

L = 1000
m = gb.Model()
c = [0.01*(l+1) for l in range(L)]
p = [1/L] * L
eta = 0.01
zeta = m.addVar(lb=-np.infty)
cos = m.addVars(L, lb=0)
m.addConstrs(cos[l] >= c[l] - zeta for l in range(L))
m.setObjective(zeta + 1 / eta * gb.quicksum(p[l] * cos[l] for l in range(L)))
m.optimize()
print(zeta.x)
print([cos[l].x for l in range(L)])

g = 0
for l in range(L):
    if zeta.x <= c[l]:
        g += p[l]
print(g)



import nengo
from nengo.processes import Piecewise
import numpy as np

# def NOT():
#     with nengo.Network() as NOT:
#         NOT.I = nengo.Ensemble(200, dimensions=1, radius=1)
#         NOT.O = nengo.Ensemble(200, dimensions=1, radius=1)

#         def not_func(x):
#             return -1 if x > 0 else 1

#         nengo.Connection(NOT.I, NOT.O, function=not_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
#     return NOT

def AND(dim=2):
    with nengo.Network() as AND:
        AND.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        AND.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def and_func(x):
            x_arr = np.array(x)
            return 1 if (x_arr > 0).all() else -1

        nengo.Connection(AND.I, AND.O, function=and_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return AND

def OR(dim=2):
    with nengo.Network() as OR:
        OR.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        OR.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def or_func(x):
            x_arr = np.array(x)
            return 1 if (x_arr > 0).any() else -1

        nengo.Connection(OR.I, OR.O, function=or_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return OR

def NAND(dim=2):
    with nengo.Network() as NAND:
        NAND.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        NAND.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def nand_func(x):
            x_arr = np.array(x)
            return -1 if (x_arr > 0).all() else 1

        nengo.Connection(NAND.I, NAND.O, function=nand_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return NAND

def NOR(dim=2):
    with nengo.Network() as NOR:
        NOR.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        NOR.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def nor_func(x):
            x_arr = np.array(x)
            return 1 if (x_arr < 0).all() else -1

        nengo.Connection(NOR.I, NOR.O, function=nor_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return NOR

def D_FlipFlop():
    with nengo.Network() as DFF:
        DFF.D = nengo.Ensemble(200, dimensions=1, radius=1)
        DFF.Clk = nengo.Ensemble(200, dimensions=1, radius=1)
        DFF.NAND0 = NAND()
        DFF.NAND1 = NAND()
        DFF.NAND2 = NAND(3)
        DFF.NAND3 = NAND()
        DFF.NAND4 = NAND()
        DFF.NAND5 = NAND()

        nengo.Connection(DFF.NAND3.O, DFF.NAND0.I[0])
        nengo.Connection(DFF.NAND1.O, DFF.NAND0.I[1])
        nengo.Connection(DFF.NAND0.O, DFF.NAND1.I[0])
        nengo.Connection(DFF.Clk, DFF.NAND1.I[1])
        nengo.Connection(DFF.NAND1.O, DFF.NAND2.I[0])
        nengo.Connection(DFF.Clk, DFF.NAND2.I[1])
        nengo.Connection(DFF.NAND3.O, DFF.NAND2.I[2])
        nengo.Connection(DFF.NAND2.O, DFF.NAND3.I[0])
        nengo.Connection(DFF.D, DFF.NAND3.I[1])
        nengo.Connection(DFF.NAND1.O, DFF.NAND4.I[0])
        nengo.Connection(DFF.NAND5.O, DFF.NAND4.I[1])
        nengo.Connection(DFF.NAND4.O, DFF.NAND5.I[0])
        nengo.Connection(DFF.NAND2.O, DFF.NAND5.I[1])

    return DFF

def JK_FlipFlop():
    with nengo.Network() as JKFF:
        JKFF.J = nengo.Ensemble(200, dimensions=1, radius=1)
        JKFF.K = nengo.Ensemble(200, dimensions=1, radius=1)
        JKFF.AND0 = AND()
        JKFF.NOR0 = NOR()
        JKFF.OR0 = OR()
        JKFF.DFF0 = D_FlipFlop()

        nengo.Connection(JKFF.DFF0.NAND5.O, JKFF.AND0.I[0])
        nengo.Connection(JKFF.J, JKFF.AND0.I[1])
        nengo.Connection(JKFF.K, JKFF.NOR0.I[0])
        nengo.Connection(JKFF.DFF0.NAND5.O, JKFF.NOR0.I[1])

        nengo.Connection(JKFF.AND0.O, JKFF.OR0.I[0])
        nengo.Connection(JKFF.NOR0.O, JKFF.OR0.I[1])
        nengo.Connection(JKFF.OR0.O, JKFF.DFF0.D)

        JKFF.Q = nengo.Ensemble(200, dimensions=1, radius=1)
        JKFF.Q_ = nengo.Ensemble(200, dimensions=1, radius=1)
        nengo.Connection(JKFF.DFF0.NAND4.O, JKFF.Q)
        nengo.Connection(JKFF.DFF0.NAND5.O, JKFF.Q_)
    return JKFF

model = nengo.Network(label='T Flip-Flop')
with model:
    JKff1 = JK_FlipFlop()
    stim_J = nengo.Node(1)
    stim_K = nengo.Node(-1)
    stim_Clk = nengo.Node(Piecewise({0: 1, 0.5: -1, 1: 1, 1.5: -1, 2: 1, 2.5: -1, 3: 1, 3.5: -1, 4: 1, 4.5: -1, 5: 1,
                                    5.5: -1, 6: 1, 6.5: -1, 7: 1, 7.5: -1, 8: 1, 8.5: -1, 9: 1, 9.5: -1, 10: 1,
                                    10.5: -1, 11: 1, 11.5: -1, 12: 1, 12.5: -1, 13: 1, 13.5: -1, 14: 1, 14.5: -1, 15: 1,
                                    15.5: -1, 16: 1, 16.5: -1, 17: 1, 17.5: -1, 18: 1, 18.5: -1, 19: 1, 19.5: -1, 20: 1}))
    nengo.Connection(stim_J, JKff1.J)
    nengo.Connection(stim_K, JKff1.K)
    nengo.Connection(stim_Clk, JKff1.DFF0.Clk)


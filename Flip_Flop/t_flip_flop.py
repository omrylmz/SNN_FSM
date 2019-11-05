import nengo
from nengo.processes import Piecewise
import numpy as np

def XOR(dim=2):
    with nengo.Network() as XOR:
        XOR.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        XOR.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def xor_func(x):
            x_arr = np.array(x)
            return 1 if (x_arr > 0).sum() % 2 else -1

        nengo.Connection(XOR.I, XOR.O, function=xor_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return XOR

def NAND(dim=2):
    with nengo.Network() as NAND:
        NAND.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        NAND.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def nand_func(x):
            x_arr = np.array(x)
            return -1 if (x_arr > 0).all() else 1

        nengo.Connection(NAND.I, NAND.O, function=nand_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return NAND

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

def T_FlipFlop():
    with nengo.Network() as TFF:
        TFF.T = nengo.Ensemble(200, dimensions=1, radius=1)
        TFF.XOR0 = XOR()
        TFF.DFF0 = D_FlipFlop()

        nengo.Connection(TFF.DFF0.NAND4.O, TFF.XOR0.I[0])
        nengo.Connection(TFF.T, TFF.XOR0.I[1])
        nengo.Connection(TFF.XOR0.O, TFF.DFF0.D)

        TFF.Q = nengo.Ensemble(200, dimensions=1, radius=1)
        TFF.Q_ = nengo.Ensemble(200, dimensions=1, radius=1)
        nengo.Connection(TFF.DFF0.NAND4.O, TFF.Q)
        nengo.Connection(TFF.DFF0.NAND5.O, TFF.Q_)
    return TFF

model = nengo.Network(label='T Flip-Flop')
with model:
    Tff1 = T_FlipFlop()
    stim_T = nengo.Node(1)
    stim_Clk = nengo.Node(Piecewise({0: 1, 0.5: -1, 1: 1, 1.5: -1, 2: 1, 2.5: -1, 3: 1, 3.5: -1, 4: 1, 4.5: -1, 5: 1,
                                    5.5: -1, 6: 1, 6.5: -1, 7: 1, 7.5: -1, 8: 1, 8.5: -1, 9: 1, 9.5: -1, 10: 1,
                                    10.5: -1, 11: 1, 11.5: -1, 12: 1, 12.5: -1, 13: 1, 13.5: -1, 14: 1, 14.5: -1, 15: 1,
                                    15.5: -1, 16: 1, 16.5: -1, 17: 1, 17.5: -1, 18: 1, 18.5: -1, 19: 1, 19.5: -1, 20: 1}))
    nengo.Connection(stim_T, Tff1.T)
    nengo.Connection(stim_Clk, Tff1.DFF0.Clk)


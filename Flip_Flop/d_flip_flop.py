import nengo
from nengo.processes import Piecewise
import numpy as np

def NAND(dim=2):
    with nengo.Network() as NAND:
        NAND.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        NAND.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def nand_func(x):
            x_arr = np.array(x)
            return -1 if (x_arr > 0).all() else 1

        nengo.Connection(NAND.I, NAND.O, function=nand_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return NAND

def NOT():
    with nengo.Network() as NOT:
        NOT.I = nengo.Ensemble(200, dimensions=1, radius=1)
        NOT.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def not_func(x):
            return -1 if x > 0 else 1

        nengo.Connection(NOT.I, NOT.O, function=not_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return NOT

def D_Latch():
    with nengo.Network() as DL:
        DL.D = nengo.Ensemble(200, dimensions=1, radius=1)
        DL.Enable = nengo.Ensemble(200, dimensions=1, radius=1)
        NAND0 = NAND()
        NAND1 = NAND()
        NAND2 = NAND()
        NAND3 = NAND()
        NOT0 = NOT()

        nengo.Connection(DL.D, NAND0.I[0])
        nengo.Connection(DL.Enable, NAND0.I[1])
        nengo.Connection(DL.D, NOT0.I)
        nengo.Connection(NOT0.O, NAND1.I[0])
        nengo.Connection(DL.Enable, NAND1.I[1])
        nengo.Connection(NAND0.O, NAND2.I[0])
        nengo.Connection(NAND3.O, NAND2.I[1])
        nengo.Connection(NAND2.O, NAND3.I[0])
        nengo.Connection(NAND1.O, NAND3.I[1])

        DL.Q = nengo.Ensemble(200, dimensions=1, radius=1)
        DL.Q_ = nengo.Ensemble(200, dimensions=1, radius=1)
        nengo.Connection(NAND2.O, DL.Q)
        nengo.Connection(NAND3.O, DL.Q_)
    return DL

def D_FlipFlop():
    with nengo.Network() as DFF:
        DFF.D = nengo.Ensemble(200, dimensions=1, radius=1)
        DFF.Clk = nengo.Ensemble(200, dimensions=1, radius=1)
        DFF.NOT0 = NOT()
        DFF.DL0 = D_Latch()
        DFF.DL1 = D_Latch()

        nengo.Connection(DFF.D, DFF.DL0.D)
        nengo.Connection(DFF.Clk, DFF.NOT0.I)
        nengo.Connection(DFF.Clk, DFF.DL0.Enable)
        nengo.Connection(DFF.NOT0.O, DFF.DL1.Enable)
        nengo.Connection(DFF.DL0.Q, DFF.DL1.D)

        DFF.Q = nengo.Ensemble(200, dimensions=1, radius=1)
        DFF.Q_ = nengo.Ensemble(200, dimensions=1, radius=1)
        nengo.Connection(DFF.DL1.Q, DFF.Q)
        nengo.Connection(DFF.DL1.Q_, DFF.Q_)
    return DFF

model = nengo.Network(label='D Flip-Flop from two D latchs')
with model:
    Dff = D_FlipFlop()
    stim_D = nengo.Node(1)
    stim_Clk = nengo.Node(Piecewise({0: 1, 0.5: -1, 1: 1, 1.5: -1, 2: 1, 2.5: -1, 3: 1, 3.5: -1, 4: 1, 4.5: -1, 5: 1,
                                    5.5: -1, 6: 1, 6.5: -1, 7: 1, 7.5: -1, 8: 1, 8.5: -1, 9: 1, 9.5: -1, 10: 1,
                                    10.5: -1, 11: 1, 11.5: -1, 12: 1, 12.5: -1, 13: 1, 13.5: -1, 14: 1, 14.5: -1, 15: 1,
                                    15.5: -1, 16: 1, 16.5: -1, 17: 1, 17.5: -1, 18: 1, 18.5: -1, 19: 1, 19.5: -1, 20: 1}))
    nengo.Connection(stim_D, Dff.D)
    nengo.Connection(stim_Clk, Dff.Clk)

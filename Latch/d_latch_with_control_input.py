import nengo
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
        DL.NAND0 = NAND()
        DL.NAND1 = NAND()
        DL.NAND2 = NAND()
        DL.NAND3 = NAND()
        DL.NOT0 = NOT()

        nengo.Connection(DL.D, DL.NAND0.I[0])
        nengo.Connection(DL.Enable, DL.NAND0.I[1])
        nengo.Connection(DL.D, DL.NOT0.I)
        nengo.Connection(DL.NOT0.O, DL.NAND1.I[0])
        nengo.Connection(DL.Enable, DL.NAND1.I[1])
        nengo.Connection(DL.NAND0.O, DL.NAND2.I[0])
        nengo.Connection(DL.NAND3.O, DL.NAND2.I[1])
        nengo.Connection(DL.NAND2.O, DL.NAND3.I[0])
        nengo.Connection(DL.NAND1.O, DL.NAND3.I[1])

        DL.Q = nengo.Ensemble(200, dimensions=1, radius=1)
        DL.Q_ = nengo.Ensemble(200, dimensions=1, radius=1)
        nengo.Connection(DL.NAND2.O, DL.Q)
        nengo.Connection(DL.NAND3.O, DL.Q_)
    return DL

model = nengo.Network(label='D Latch with control input')
with model:
    Dl = D_Latch()
    stim_S = nengo.Node(-1)
    stim_Enable = nengo.Node(1)

    nengo.Connection(stim_S, Dl.D)
    nengo.Connection(stim_Enable, Dl.Enable)

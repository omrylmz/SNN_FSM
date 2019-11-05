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

def SR_Latch():
    with nengo.Network() as SRL:
        SRL.S = nengo.Ensemble(200, dimensions=1, radius=1)
        SRL.R = nengo.Ensemble(200, dimensions=1, radius=1)
        SRL.NAND0 = NAND()
        SRL.NAND1 = NAND()

        nengo.Connection(SRL.S, SRL.NAND0.I[0])
        nengo.Connection(SRL.NAND1.O, SRL.NAND0.I[1])
        nengo.Connection(SRL.NAND0.O, SRL.NAND1.I[0])
        nengo.Connection(SRL.R, SRL.NAND1.I[1])

        SRL.Q = nengo.Ensemble(200, dimensions=1, radius=1)
        SRL.Q_ = nengo.Ensemble(200, dimensions=1, radius=1)
        nengo.Connection(SRL.NAND0.O, SRL.Q)
        nengo.Connection(SRL.NAND1.O, SRL.Q_)
    return SRL

model = nengo.Network(label='SR Latch with NAND gates')
with model:
    SRl = SR_Latch()
    stim_S = nengo.Node(1)
    stim_R = nengo.Node(1)

    nengo.Connection(stim_R, SRl.R)
    nengo.Connection(stim_S, SRl.S)
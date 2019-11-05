import nengo
import numpy as np

def NOR(dim=2):
    with nengo.Network() as NOR:
        NOR.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        NOR.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def nor_func(x):
            x_arr = np.array(x)
            return 1 if (x_arr < 0).all() else -1

        nengo.Connection(NOR.I, NOR.O, function=nor_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return NOR

def SR_Latch():
    with nengo.Network() as SRL:
        SRL.R = nengo.Ensemble(200, dimensions=1, radius=1)
        SRL.S = nengo.Ensemble(200, dimensions=1, radius=1)
        NOR0 = NOR()
        NOR1 = NOR()

        nengo.Connection(SRL.S, NOR0.I[0])
        nengo.Connection(NOR1.O, NOR0.I[1])
        nengo.Connection(SRL.R, NOR1.I[0])
        nengo.Connection(NOR0.O, NOR1.I[1])
        
        SRL.Q = nengo.Ensemble(200, dimensions=1, radius=1)
        SRL.Q_ = nengo.Ensemble(200, dimensions=1, radius=1)
        nengo.Connection(NOR0.O, SRL.Q_)
        nengo.Connection(NOR1.O, SRL.Q)
    return SRL

model = nengo.Network(label='SR Latch with NOR gates')
with model:
    SRl = SR_Latch()
    stim_S = nengo.Node(1)
    stim_R = nengo.Node(-1)

    nengo.Connection(stim_R, SRl.R)
    nengo.Connection(stim_S, SRl.S)

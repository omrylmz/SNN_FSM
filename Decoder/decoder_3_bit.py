import nengo
from nengo.processes import Piecewise
import numpy as np

def NOT():
    with nengo.Network() as NOT:
        NOT.I = nengo.Ensemble(200, dimensions=1, radius=1)
        NOT.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def not_func(x):
            return -1 if x > 0 else 1

        nengo.Connection(NOT.I, NOT.O, function=not_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return NOT

def AND(dim=2):
    with nengo.Network() as AND:
        AND.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        AND.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def and_func(x):
            x_arr = np.array(x)
            return 1 if (x_arr > 0).all() else -1

        nengo.Connection(AND.I, AND.O, function=and_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return AND

def AND_3bit():
    with nengo.Network() as AND:
        AND.I = nengo.Ensemble(1800, dimensions=3, radius=1)
        AND.O = nengo.Ensemble(200, dimensions=1, radius=1)
        AND.O01 = nengo.Ensemble(200, dimensions=1, radius=1)
        AND.I12 = nengo.Ensemble(200, dimensions=2, radius=1)

        def and_func(x):
            return 1 if (x[0] > 0) and (x[1] > 0) else -1

        nengo.Connection(AND.O01, AND.I12[0])
        nengo.Connection(AND.I[2], AND.I12[1])
        nengo.Connection(AND.I[0:2], AND.O01, function=and_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
        nengo.Connection(AND.I12, AND.O, function=and_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
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

def Decoder_3to8():
    with nengo.Network() as Decoder:
        Decoder.I = nengo.Ensemble(1800, dimensions=3, radius=1)
        Decoder.notI = nengo.Ensemble(1800, dimensions=3, radius=1)
        Decoder.O = nengo.Ensemble(1600, dimensions=8, radius=1)

        def not_func(x):
            return -1 if x > 0 else 1

        for i in range(3):
            nengo.Connection(Decoder.I[i], Decoder.notI[i], function=not_func, 
                            learning_rule_type=nengo.PES(learning_rate=2e-4))

        AND0 = AND_3bit()
        AND1 = AND_3bit()
        AND2 = AND_3bit()
        AND3 = AND_3bit()
        AND4 = AND_3bit()
        AND5 = AND_3bit()
        AND6 = AND_3bit()
        AND7 = AND_3bit()

        nengo.Connection(Decoder.notI[0], AND0.I[0])
        nengo.Connection(Decoder.notI[1], AND0.I[1])
        nengo.Connection(Decoder.notI[2], AND0.I[2])

        nengo.Connection(Decoder.I[0], AND1.I[0])
        nengo.Connection(Decoder.notI[1], AND1.I[1])
        nengo.Connection(Decoder.notI[2], AND1.I[2])

        nengo.Connection(Decoder.notI[0], AND2.I[0])
        nengo.Connection(Decoder.I[1], AND2.I[1])
        nengo.Connection(Decoder.notI[2], AND2.I[2])

        nengo.Connection(Decoder.I[0], AND3.I[0])
        nengo.Connection(Decoder.I[1], AND3.I[1])
        nengo.Connection(Decoder.notI[2], AND3.I[2])

        nengo.Connection(Decoder.notI[0], AND4.I[0])
        nengo.Connection(Decoder.notI[1], AND4.I[1])
        nengo.Connection(Decoder.I[2], AND4.I[2])

        nengo.Connection(Decoder.I[0], AND5.I[0])
        nengo.Connection(Decoder.notI[1], AND5.I[1])
        nengo.Connection(Decoder.I[2], AND5.I[2])

        nengo.Connection(Decoder.notI[0], AND6.I[0])
        nengo.Connection(Decoder.I[1], AND6.I[1])
        nengo.Connection(Decoder.I[2], AND6.I[2])

        nengo.Connection(Decoder.I[0], AND7.I[0])
        nengo.Connection(Decoder.I[1], AND7.I[1])
        nengo.Connection(Decoder.I[2], AND7.I[2])

        nengo.Connection(AND0.O, Decoder.O[0])
        nengo.Connection(AND1.O, Decoder.O[1])
        nengo.Connection(AND2.O, Decoder.O[2])
        nengo.Connection(AND3.O, Decoder.O[3])
        nengo.Connection(AND4.O, Decoder.O[4])
        nengo.Connection(AND5.O, Decoder.O[5])
        nengo.Connection(AND6.O, Decoder.O[6])
        nengo.Connection(AND7.O, Decoder.O[7])
    return Decoder

model = nengo.Network(label='3-bit Decoder')
with model:
    Decoder = Decoder_3to8()
    stim_I0 = nengo.Node(1)
    stim_I1 = nengo.Node(1)
    stim_I2 = nengo.Node(1)

    nengo.Connection(stim_I0, Decoder.I[0])
    nengo.Connection(stim_I1, Decoder.I[1])
    nengo.Connection(stim_I2, Decoder.I[2])
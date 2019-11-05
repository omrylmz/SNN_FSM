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

def NAND(dim=2):
    with nengo.Network() as NAND:
        NAND.I = nengo.Ensemble(dim*dim*200, dimensions=dim, radius=1)
        NAND.O = nengo.Ensemble(200, dimensions=1, radius=1)

        def nand_func(x):
            x_arr = np.array(x)
            return -1 if (x_arr > 0).all() else 1

        nengo.Connection(NAND.I, NAND.O, function=nand_func, learning_rule_type=nengo.PES(learning_rate=2e-4))
    return NAND

def Decoder_3to8():
    with nengo.Network() as Decoder:
        Decoder.I = nengo.Ensemble(1800, dimensions=3, radius=1)
        Decoder.notI = nengo.Ensemble(1800, dimensions=3, radius=1)
        Decoder.O = nengo.Ensemble(1600, dimensions=8, radius=1)

        def not_func(x):
            return -1 if x > 0 else 1

        for i in range(3):
            nengo.Connection(Decoder.I[i], Decoder.notI[i], function=not_func, learning_rule_type=nengo.PES(learning_rate=2e-4))

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

model = nengo.Network(label='State Machine')
with model:
    stim_Clk = nengo.Node(Piecewise({0: 1, 0.5: -1, 1: 1, 1.5: -1, 2: 1, 2.5: -1, 3: 1, 3.5: -1, 4: 1, 4.5: -1, 5: 1,
                                    5.5: -1, 6: 1, 6.5: -1, 7: 1, 7.5: -1, 8: 1, 8.5: -1, 9: 1, 9.5: -1, 10: 1,
                                    10.5: -1, 11: 1, 11.5: -1, 12: 1, 12.5: -1, 13: 1, 13.5: -1, 14: 1, 14.5: -1, 15: 1,
                                    15.5: -1, 16: 1, 16.5: -1, 17: 1, 17.5: -1, 18: 1, 18.5: -1, 19: 1, 19.5: -1, 20: 1}))

    ready = nengo.Node(-1)
    reset_fallen = nengo.Node(-1)
    ball_detected = nengo.Node(-1)
    ball_close = nengo.Node(-1)
    ball_lost = nengo.Node(-1)
    wait = nengo.Node(-1)

    not_ready = NOT()
    not_reset_fallen = NOT()
    not_ball_detected = NOT()
    not_ball_close = NOT()
    not_ball_lost = NOT()
    not_wait = NOT()

    nengo.Connection(ready, not_ready.I)
    nengo.Connection(reset_fallen, not_reset_fallen.I)
    nengo.Connection(ball_detected, not_ball_detected.I)
    nengo.Connection(ball_close, not_ball_close.I)
    nengo.Connection(ball_lost, not_ball_lost.I)
    nengo.Connection(wait, not_wait.I)

    Decoder0 = Decoder_3to8()
    D0 = D_FlipFlop()
    D1 = D_FlipFlop()
    D2 = D_FlipFlop()

    nengo.Connection(stim_Clk, D0.Clk)
    nengo.Connection(stim_Clk, D1.Clk)
    nengo.Connection(stim_Clk, D2.Clk)
    nengo.Connection(D0.Q, Decoder0.I[0])
    nengo.Connection(D1.Q, Decoder0.I[1])
    nengo.Connection(D2.Q, Decoder0.I[2])

    # 000 = Initial State
    # 001 = Ball-Searching State
    # 002 = Approaching State
    # 003 = Decision State
    # 004 = Ball-Kicking State

    AND0 = AND()
    nengo.Connection(Decoder0.O[0], AND0.I[0])
    nengo.Connection(ready, AND0.I[1])

    AND1 = AND_3bit()
    nengo.Connection(Decoder0.O[1], AND1.I[0])
    nengo.Connection(not_reset_fallen.O, AND1.I[1])
    nengo.Connection(not_ball_detected.O, AND1.I[2])

    AND2 = AND_3bit()
    nengo.Connection(Decoder0.O[2], AND2.I[0])
    nengo.Connection(not_reset_fallen.O, AND2.I[1])
    nengo.Connection(ball_close, AND2.I[2])

    AND3 = AND()
    nengo.Connection(Decoder0.O[3], AND3.I[0])
    nengo.Connection(not_reset_fallen.O, AND3.I[1])

    AND4 = AND()
    nengo.Connection(not_ball_lost.O, AND4.I[0])
    nengo.Connection(wait, AND4.I[1])

    AND5 = AND()
    nengo.Connection(AND3.O, AND5.I[0])
    nengo.Connection(AND4.O, AND5.I[1])

    AND6 = AND()
    nengo.Connection(Decoder0.O[4], AND6.I[0])
    nengo.Connection(not_reset_fallen.O, AND6.I[1])

    OR0 = OR()
    nengo.Connection(AND0.O, OR0.I[0])
    nengo.Connection(AND1.O, OR0.I[1])

    OR1 = OR()
    nengo.Connection(AND2.O, OR1.I[0])
    nengo.Connection(AND3.O, OR1.I[1])

    OR2 = OR()
    nengo.Connection(OR0.O, OR2.I[0])
    nengo.Connection(OR1.O, OR2.I[1])

    OR3 = OR()
    nengo.Connection(OR2.O, OR3.I[0])
    nengo.Connection(AND6.O, OR3.I[1])

    nengo.Connection(OR3.O, D0.D)

    AND7 = AND_3bit()
    nengo.Connection(Decoder0.O[1], AND7.I[0])
    nengo.Connection(not_reset_fallen.O, AND7.I[1])
    nengo.Connection(ball_detected, AND7.I[2])

    AND8 = AND()
    nengo.Connection(Decoder0.O[2], AND8.I[0])
    nengo.Connection(not_reset_fallen.O, AND8.I[1])

    AND9 = AND_3bit()
    nengo.Connection(Decoder0.O[3], AND9.I[0])
    nengo.Connection(not_reset_fallen.O, AND9.I[1])
    nengo.Connection(ball_lost, AND9.I[2])

    AND10 = AND()
    nengo.Connection(Decoder0.O[3], AND10.I[0])
    nengo.Connection(not_reset_fallen.O, AND10.I[1])

    AND11 = AND()
    nengo.Connection(not_ball_lost.O, AND11.I[0])
    nengo.Connection(wait, AND11.I[1])

    AND12 = AND()
    nengo.Connection(AND10.O, AND12.I[0])
    nengo.Connection(AND11.O, AND12.I[1])

    OR4 = OR()
    nengo.Connection(AND7.O, OR4.I[0])
    nengo.Connection(AND8.O, OR4.I[1])

    OR5 = OR()
    nengo.Connection(AND9.O, OR5.I[0])
    nengo.Connection(AND12.O, OR5.I[1])

    OR6 = OR()
    nengo.Connection(OR4.O, OR6.I[0])
    nengo.Connection(OR5.O, OR6.I[1])

    nengo.Connection(OR6.O, D1.D)

    AND13 = AND()
    nengo.Connection(Decoder0.O[3], AND13.I[0])
    nengo.Connection(not_reset_fallen.O, AND13.I[1])

    AND14 = AND()
    nengo.Connection(not_ball_lost.O, AND14.I[0])
    nengo.Connection(not_wait.O, AND14.I[1])

    AND15 = AND()
    nengo.Connection(AND13.O, AND15.I[0])
    nengo.Connection(AND14.O, AND15.I[1])

    nengo.Connection(AND15.O, D2.D)
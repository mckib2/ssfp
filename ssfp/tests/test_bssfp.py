import numpy as np
from ssfp import bssfp
from ssfp.bssfp import _result_shape


def test_result_shape():
    assert _result_shape([(2, 3), (2, 3)])[0] == (2, 3)
    assert _result_shape([(2, 3), (3, 2)])[0] == (2, 3, 3, 2)
    assert _result_shape([(2, 3), (2, 3), (3,)])[0] == (2, 3, 3)
    assert _result_shape([(2, 3), (2, 3), (3,), (3,)])[0] == (2, 3, 3)
    assert _result_shape([(1,), (2, 3), (2, 3)])[0] == (1, 2, 3)
    assert _result_shape([(2, 3), (1,), (2, 3)])[0] == (2, 3, 1)


def test_scalar():
    s = bssfp(T1=1, T2=.5, TR=3e-3, alpha=np.deg2rad(30))
    assert isinstance(s, np.complex128)
    assert not isinstance(s, np.ndarray)


def test_mult_pcs():
    s = bssfp(T1=1, T2=.5, TR=3e-3, alpha=np.deg2rad(30), phase_cyc=[0.0, np.pi])
    assert s.shape == (2,)


def test_mult_alphas():
    s = bssfp(T1=1, T2=.5, TR=3e-3, alpha=(np.deg2rad(30), np.deg2rad(60)))
    assert s.shape == (2,)


def test_mult_pcs_and_alphas():
    s = bssfp(T1=1, T2=.5, TR=3e-3, alpha=(np.deg2rad(30), np.deg2rad(60)), phase_cyc=[0, np.pi/3])
    assert s.shape == (2,)

    s = bssfp(T1=1, T2=.5, TR=3e-3, alpha=(np.deg2rad(30), np.deg2rad(60)), phase_cyc=[0, np.pi/3, 2*np.pi/3])
    assert s.shape == (2, 3)

    s = bssfp(T1=1, T2=.5, TR=3e-3, alpha=(np.deg2rad(30), np.deg2rad(45), np.deg2rad(60)), phase_cyc=[0, np.pi])
    assert s.shape == (3, 2)


def test_mult_t1s():
    s = bssfp(T1=[1, 2], T2=.5, TR=3e-3, alpha=np.deg2rad(30))
    assert s.shape == (2,)


def test_mult_t2s():
    s = bssfp(T1=1, T2=[.5, .3], TR=3e-3, alpha=np.deg2rad(30))
    assert s.shape == (2,)


def test_mult_t1s_and_t2s():
    s = bssfp(T1=[1, 2, 3], T2=[.5, .3, .4], TR=3e-3, alpha=np.deg2rad(30))
    assert s.shape == (3,)

    s = bssfp(T1=[1, 2], T2=[.5, .3, .4], TR=3e-3, alpha=np.deg2rad(30))
    assert s.shape == (2, 3)

    s = bssfp(T1=[1, 2, 3], T2=[.5, .3], TR=3e-3, alpha=np.deg2rad(30))
    assert s.shape == (3, 2)

    s = bssfp(T1=[[1, 2, 3], [4, 5, 6]], T2=[.5, .3], TR=3e-3, alpha=np.deg2rad(30))
    assert s.shape == (2, 3, 2)


def test_mult_t1s_and_t2s_and_alphas_and_pcs():
    s = bssfp(T1=[1, 2], T2=[.5, .3, .4], TR=3e-3, alpha=np.linspace(.01, np.pi, 4), phase_cyc=np.linspace(-2*np.pi, 2*np.pi, 5, endpoint=False))
    assert s.shape == (2, 3, 4, 5)


def test_mult_t1s_and_t2s_and_alphas_and_field_map_and_pcs():
    s = bssfp(
        T1=[1, 2],
        T2=[.5, .3, .4],
        TR=3e-3,
        alpha=np.linspace(.01, np.pi, 4),
        field_map=np.linspace(0, 2*np.pi, 5),
        phase_cyc=np.linspace(-2*np.pi, 2*np.pi, 6, endpoint=False),
    )
    assert s.shape == (2, 3, 4, 5, 6)


def test_mult_t1s_and_t2s_and_trs_and_alphas_and_field_map_and_pcs():
    s = bssfp(
        T1=[1, 2],
        T2=[.5, .3, .4],
        TR=[3e-3, 4e-3, 5e-3, 6e-3],
        alpha=np.linspace(.01, np.pi, 5),
        field_map=np.linspace(0, 2*np.pi, 6),
        phase_cyc=np.linspace(-2*np.pi, 2*np.pi, 7, endpoint=False),
    )
    assert s.shape == (2, 3, 4, 5, 6, 7)


def test_mult_t1s_and_t2s_and_trs_and_alphas_and_field_map_and_pcs_and_m0s():
    s = bssfp(
        T1=[1, 2],
        T2=[.5, .3, .4],
        TR=[3e-3, 4e-3, 5e-3, 6e-3],
        alpha=np.linspace(.01, np.pi, 5),
        field_map=np.linspace(0, 2*np.pi, 6),
        phase_cyc=np.linspace(-2*np.pi, 2*np.pi, 7, endpoint=False),
        M0=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    assert s.shape == (2, 3, 4, 5, 6, 7, 8)


def test_field_map_phase_cyc_false_equivalence():
    TR = 3e-3
    s_fm = bssfp(T1=2, T2=.05, TR=TR, alpha=np.deg2rad(20), field_map=np.linspace(-1/(np.pi*TR), 1/(np.pi*TR), 256), phase_cyc=0)
    s_pc = bssfp(T1=2, T2=.05, TR=TR, alpha=np.deg2rad(20), field_map=0, phase_cyc=np.linspace(-np.pi, np.pi, 256))

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(np.abs(s_fm), label="field_map")
    # ax.plot(np.abs(s_pc), label="phase_cyc")
    # ax.legend()
    # ax_phase = ax.twinx()
    # ax_phase.plot(np.angle(s_fm), "--")
    # ax_phase.plot(np.angle(s_pc), "--")
    # plt.show()

    assert not np.allclose(s_fm, s_pc)

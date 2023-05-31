import meep as mp
import multiprocessing
import contextlib
import io

import math
import cmath
import pathlib
import numpy as np
import meep.materials as material
import pickle

tpv_file_path = (pathlib.Path(__file__).parent.resolve().parent.resolve() /
                 "assets" / "tpv_model").resolve()

resol = 40
sim_eps = 1e-5

def load_target():
    target_path = str((tpv_file_path / "tpv_target.txt").resolve())
    target = np.loadtxt(target_path)
    wavelength = np.flip(target[:, 0], axis=0)
    target_value = np.flip(target[:, 1], axis=0)
    return target_value, wavelength

def load(filename):
    with open(filename, "rb") as f:
        r = pickle.load(f)
    return r


def DeBoorCST(Pt, degree, n):
    n_int = n  # 11 for hollow design  n is the number of control points between two pints??

    Pctr = np.vstack(
        (Pt[Pt.shape[0] - degree - 1:, :], Pt, Pt[0:degree - 1, :]))
    p = Pctr.shape[0]
    q = Pctr.shape[1]
    nt = (n_int - 1) * (p - 1) + 1
    tx, _ = np.linspace(0, 1, nt, True, True)
    Xl = np.zeros((degree, degree, q))
    tx1, _ = np.linspace(0, 1, p - degree + 2, True, True)
    t = np.hstack((np.zeros(degree - 1), tx1, np.ones(degree - 1)))
    BS = np.zeros((nt, q))
    m_min = 1
    m_max = nt
    for m in range(1, nt):
        t0 = tx[m - 1]
        idx = [idx for (idx, val) in enumerate(t) if val <= t0]
        k = idx[-1] + 1
        if (k > p):
            BS = BS[1:m - 2, :]
            print('return')
            return
        if (k <= degree + 1):
            m_min = max(m, m_min)
        if (k >= p - degree + 2):
            m_max = min(m, m_max)
        Xl[:, 0, :] = Pctr[k - degree: k, :]
        for j in range(2, degree + 1):
            for i in range(j, degree + 1):
                num = t0 - t[k - degree + i - 1]
                s = t[k + i - j] - t[k - degree + i - 1]
                wt = num / s
                Xl[i - 1, j - 1, :] = (1 - wt) * Xl[i - 2,
                                                    j - 2, :] + wt * Xl[i - 1, j - 2, :]
        BS[m - 1, :] = Xl[degree - 1, degree - 1, :]
    BS[-1, :] = Pctr[-1, :]
    BS2 = BS[m_min: m_max, :]
    BS2[-1, :] = BS2[0, :]
    return BS2


def x2params(x):
    assert x.shape[0] == 1
    M = 4
    order = 3
    r1 = np.zeros((x.shape[0], 4))
    r2 = np.zeros((x.shape[0], 4))
    r3 = np.zeros((x.shape[0], 4))
    r4 = np.zeros((x.shape[0], 4))
    p = x[:, 0:1]
    ts = x[:, 1:2]
    tm = x[:, 2:3]
    for i in range(4):
        r1[:, i] = x[:, 3 + 4 * i]
    for i in range(4):
        r2[:, i] = x[:, 4 + 4 * i]
    for i in range(4):
        r3[:, i] = x[:, 5 + 4 * i]
    for i in range(4):
        r4[:, i] = x[:, 6 + 4 * i]

    sizenum = x.shape[0]
    nt = 5

    a = (nt - 1) * (M - 1) + 1
    theta = np.arange(0, np.pi / 2 + np.pi /
                      (2 * (M - 1)), np.pi / (2 * (M - 1)))
    Pointx = np.zeros([a * 4, sizenum])
    Pointy = np.zeros([a * 4, sizenum])
    x0 = np.zeros([sizenum, M * 4])
    y0 = np.zeros([sizenum, M * 4])

    for ii in range(0, 4):  # there will be four unit cells in a supercell.
        r = np.column_stack((r1[:, ii], r2[:, ii], r3[:, ii], r4[:, ii]))
        # initialize the coordinate of x
        x = np.zeros([r.shape[0], (r.shape[1] - 1) * 4])
        # initialize the coordinate of x
        y = np.zeros([r.shape[0], (r.shape[1] - 1) * 4])

        for i in range(0, M):                         # convert polar to Cartesian
            x[:, i] = r[:, i] * np.cos(theta[i])
            y[:, i] = r[:, i] * np.sin(theta[i])

        x0[:, ii * M:(ii + 1) * M] = x[:, 0:M]
        y0[:, ii * M:(ii + 1) * M] = y[:, 0:M]

        # mirroring x to 2nd quadrant
        x[:, M:(M - 1) * 2 + 1] = -np.fliplr(x[:, 0:M - 1])
        y[:, M:(M - 1) * 2 + 1] = np.fliplr(y[:, 0:M - 1])
        # mirroring x to lower half plane
        x[:, M * 2 - 1:(M - 1) * 4] = np.fliplr(x[:, 1:(M - 1) * 2])
        y[:, M * 2 - 1:(M - 1) * 4] = -np.fliplr(y[:, 1:(M - 1) * 2])

        ps = np.zeros([2, (M - 1) * 4, sizenum])
        P_con = np.zeros([ps.shape[0], (M - 1) * 4, sizenum])
        Pointarrx = []
        Pointarry = []

        for i in range(0, sizenum):
            ps[:, :, i] = np.vstack((x[i, :], y[i, :]))
            P_con[:, :, i] = ps[:, :, i]
            Point = DeBoorCST(np.transpose(P_con[:, :, i]), order, nt)
            Pointarrx.append(Point[:, 0])
            Pointarry.append(Point[:, 1])
        Pointx1 = np.array(Pointarrx).T
        Pointy1 = np.array(Pointarry).T
        Pointx[a * ii:a * (ii + 1), :] = Pointx1[6: 19, :]
        Pointy[a * ii:a * (ii + 1), :] = Pointy1[6: 19, :]

    pnum = 13  # 13 points for each unit cell, total 4 unit cells in a supercell
    numpara = pnum * 2 * 4 + 3

    for i in range(0, M):
        Pointx[pnum * (i + 1) - 1, :] = 0
        Pointy[pnum * i, :] = 0

    # 27 means total 5x5=25 length of post and one height h and one periodicity p
    para = np.zeros([numpara, sizenum])

    para[0, :] = p.T
    para[1, :] = ts.T
    para[2, :] = tm.T
    para[3:pnum * 4 + 3, :] = Pointx
    para[pnum * 4 + 3:, :] = Pointy
    idx = np.argsort(para[0, :])
    sortpara = para[:, idx]
    gparan = np.vstack((sortpara.T))
    return gparan


def V3(a1=0.0, a2=0.0, a3=0.0):
    """Pack vector for passage to core meep routines."""
    v = v3(a1, a2, a3)
    return mp.Vector3(v[0], v[1], v[2])


def v3(a1=0.0, a2=0.0, a3=0.0):
    """Unpack vector returned by core meep routine."""
    if type(a1) in [list, np.array, np.ndarray]:
        return np.array([float(x) for x in (list(a1) + [0, 0])[0:3]])
    if isinstance(a1, mp.Vector3):
        return a1.__array__()
    return np.array([a1, a2, a3])


def pw_amp(k, x0):
    def _pw_amp(x):
        return cmath.exp(1j * 2 * math.pi * k.dot(x + x0))
    return _pw_amp


def _simulate_background(params):
    print("cal background")
    # default unit length is 1 um
    resolution = 50     # pixels/um

    p_ = params[0]
    p = p_.tolist() * 2              # periodicity
    ts_ = params[1]
    ts = ts_.tolist()            # height of tungsten
    tm_ = params[2]
    tm = tm_.tolist()            # height of Al2O3

    t_sub = 0.08

    tpml = 1         # PML thickness
    tair = 1        # air thickness

    sz = 2 * (tpml + tair) + ts + tm + t_sub
    cell_size = mp.Vector3(p, p, sz)

    pml_layers = [mp.PML(thickness=tpml, direction=mp.Z, side=mp.High),
                  mp.Absorber(thickness=tpml, direction=mp.Z, side=mp.Low)]

    lmin = 1         # source min wavelength
    lmax = 3         # source max wavelength
    fmin = 1 / lmax       # source min frequency  unit of THz
    fmax = 1 / lmin       # source max frequency
    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    # CCW rotation angle (degrees) about Y-axis of PW current source; 0 degrees along -z axis
    theta = math.radians(0)

    # k with correct length (plane of incidence: XZ)

    k = mp.Vector3(math.sin(theta), 0, math.cos(theta)).scale(fcen)

    src_pos = 0.8 * tair + 0.5 * tm
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ey, center=mp.Vector3(0, 0, src_pos),
                         size=mp.Vector3(p, p, 0),
                         amp_func=pw_amp(k, mp.Vector3(0, 0, src_pos)))]

    sim = mp.Simulation(cell_size=cell_size,
                        geometry=[],
                        sources=sources,
                        boundary_layers=pml_layers,
                        k_point=k,
                        resolution=resolution)

    nfreq = 500
    refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
        center=mp.Vector3(0, 0, tm / 2.0 + 0.5 * tair), size=mp.Vector3(p, p, 0)))
    trans = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(
        0, 0, -tm / 2.0 - ts - t_sub - 0.5 * tair), size=mp.Vector3(p, p, 0)))

    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        25, mp.Ey, mp.Vector3(0, 0, tm / 2.0 + 0.5 * tair), sim_eps))

    straight_refl_data = sim.get_flux_data(refl)
    straight_trans_flux = mp.get_fluxes(trans)

    return straight_refl_data, straight_trans_flux


def _simulate(x):
    if len(x.shape) == 1:
        x = np.reshape(x, (1, -1))
    else:
        assert len(x.shape) == 2 and x.shape[0] == 1
    params = x2params(x)[0]

    # default unit length is 1 um
    um_scale = 300
    resolution = 50     # pixels/um

    params = params / 1000

    straight_refl_data, straight_tran_flux = _simulate_background(params)

    a = 0.72         # lattice periodicity
    p_ = params[0]
    p = p_.tolist() * 2              # periodicity
    ts_ = params[1]
    ts = ts_.tolist()            # height of tungsten
    tm_ = params[2]
    tm = tm_.tolist()            # height of Al2O3

    t_sub = 0.08

    whole_sdata = np.transpose(np.reshape(params[3:], (2, 52)))
    [ve1, ve2, ve3, ve4] = np.split(whole_sdata, 4, axis=0)
    ve1_r1 = np.array(list(reversed(list(ve1[:, 0]))))
    ve1_r2 = np.array(list(reversed(list(ve1[:, 1]))))
    ve2_r1 = np.array(list(reversed(list(ve2[:, 0]))))
    ve2_r2 = np.array(list(reversed(list(ve2[:, 1]))))
    ve3_r1 = np.array(list(reversed(list(ve3[:, 0]))))
    ve3_r2 = np.array(list(reversed(list(ve3[:, 1]))))
    ve4_r1 = np.array(list(reversed(list(ve4[:, 0]))))
    ve4_r2 = np.array(list(reversed(list(ve4[:, 1]))))
    vert1 = np.r_[ve1, np.c_[-ve1_r1[1:], ve1_r2[1:]], -
                  ve1[1:], np.c_[ve1_r1[1:], -ve1_r2[1:]]]
    vert2 = np.r_[ve2, np.c_[-ve2_r1[1:], ve2_r2[1:]], -
                  ve2[1:], np.c_[ve2_r1[1:], -ve2_r2[1:]]]
    vert3 = np.r_[ve3, np.c_[-ve3_r1[1:], ve3_r2[1:]], -
                  ve3[1:], np.c_[ve3_r1[1:], -ve3_r2[1:]]]
    vert4 = np.r_[ve4, np.c_[-ve4_r1[1:], ve4_r2[1:]], -
                  ve4[1:], np.c_[ve4_r1[1:], -ve4_r2[1:]]]
    b = np.zeros((48, 1))
    verts1 = np.c_[vert1[0:48], b]
    verts2 = np.c_[vert2[0:48], b]
    verts3 = np.c_[vert3[0:48], b]
    verts4 = np.c_[vert4[0:48], b]

    "create the the vector list for each point"
    vertlist1 = [None] * 48
    vertlist2 = [None] * 48
    vertlist3 = [None] * 48
    vertlist4 = [None] * 48
    for i in range(48):
        vertlist1[i] = V3(verts1[i])
        vertlist2[i] = V3(verts2[i])
        vertlist3[i] = V3(verts3[i])
        vertlist4[i] = V3(verts4[i])

    tpml = 1         # PML thickness
    tair = 1        # air thickness

    sz = 2 * (tpml + tair) + ts + tm + t_sub
    cell_size = mp.Vector3(p, p, sz)

    pml_layers = [mp.PML(thickness=tpml, direction=mp.Z, side=mp.High),
                  mp.Absorber(thickness=tpml, direction=mp.Z, side=mp.Low)]

    lmin = 1         # source min wavelength
    lmax = 3         # source max wavelength
    fmin = 1 / lmax       # source min frequency  unit of THz
    fmax = 1 / lmin       # source max frequency
    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    n_air = 1
    Air = mp.Medium(index=n_air)

    metal_range = mp.FreqRange(min=1.0 / 12.398, max=1.0 / .20664)
    eV_um_scale = 1.0 / 1.23984193
    W_plasma_frq = 13.22 * eV_um_scale
    W_f0 = 0.206
    W_frq0 = 1e-10
    W_gam0 = 0.064 * eV_um_scale
    W_sig0 = W_f0 * W_plasma_frq ** 2 / W_frq0 ** 2
    W_f1 = 0.054
    W_frq1 = 1.004 * eV_um_scale  # 1.235 um
    W_gam1 = 0.530 * eV_um_scale
    W_sig1 = W_f1 * W_plasma_frq ** 2 / W_frq1 ** 2
    W_f2 = 0.166
    W_frq2 = 1.917 * eV_um_scale  # 0.647 um
    W_gam2 = 1.281 * eV_um_scale
    W_sig2 = W_f2 * W_plasma_frq ** 2 / W_frq2 ** 2
    W_f3 = 0.706
    W_frq3 = 3.580 * eV_um_scale  # 0.346 um
    W_gam3 = 3.332 * eV_um_scale
    W_sig3 = W_f3 * W_plasma_frq ** 2 / W_frq3 ** 2

    W_susc = [mp.DrudeSusceptibility(frequency=W_frq0, gamma=W_gam0, sigma=W_sig0),
              mp.LorentzianSusceptibility(
                  frequency=W_frq1, gamma=W_gam1, sigma=W_sig1),
              mp.LorentzianSusceptibility(
                  frequency=W_frq2, gamma=W_gam2, sigma=W_sig2),
              mp.LorentzianSusceptibility(frequency=W_frq3, gamma=W_gam3, sigma=W_sig3), ]

    W = mp.Medium(epsilon=1.0, E_susceptibilities=W_susc,
                  valid_freq_range=metal_range)

    geometry = [mp.Block(material=Air, size=mp.Vector3(p, p, tair + tpml),
                         center=mp.Vector3(0, 0, 0.5 * tm + 0.5 * (tpml + tair))),

                # four different prism
                mp.Prism(material=W, vertices=vertlist1, height=tm, axis=mp.Vector3(0, 0, 1),
                         center=mp.Vector3(p / 4, p / 4, 0)),
                mp.Prism(material=W, vertices=vertlist2, height=tm, axis=mp.Vector3(0, 0, 1),
                         center=mp.Vector3(-p / 4, p / 4, 0)),
                mp.Prism(material=W, vertices=vertlist3, height=tm, axis=mp.Vector3(0, 0, 1),
                         center=mp.Vector3(p / 4, -p / 4, 0)),
                mp.Prism(material=W, vertices=vertlist4, height=tm, axis=mp.Vector3(0, 0, 1),
                         center=mp.Vector3(-p / 4, -p / 4, 0)),

                mp.Block(material=material.Al2O3, size=mp.Vector3(p, p, ts),
                         center=mp.Vector3(0, 0, -0.5 * tm - 0.5 * ts)),
                mp.Block(material=W, size=mp.Vector3(p, p, t_sub),
                         center=mp.Vector3(0, 0, -0.5 * tm - ts - 0.5 * t_sub)),
                mp.Block(material=Air, size=mp.Vector3(p, p, tair + tpml),
                         center=mp.Vector3(0, 0, -0.5 * tm - ts - t_sub - 0.5 * (tpml + tair)))]

    # CCW rotation angle (degrees) about Y-axis of PW current source; 0 degrees along -z axis
    theta = math.radians(0)

    # k with correct length (plane of incidence: XZ)

    k = mp.Vector3(math.sin(theta), 0, math.cos(theta)).scale(fcen)

    src_pos = 0.8 * tair + 0.5 * tm
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ey, center=mp.Vector3(0, 0, src_pos),
                         size=mp.Vector3(p, p, 0),
                         amp_func=pw_amp(k, mp.Vector3(0, 0, src_pos)))]

    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        sources=sources,
                        boundary_layers=pml_layers,
                        k_point=k,
                        resolution=resolution)

    nfreq = 500
    refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
        center=mp.Vector3(0, 0, tm / 2.0 + 0.5 * tair), size=mp.Vector3(p, p, 0)))
    tran = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(
        0, 0, -tm / 2.0 - ts - t_sub - 0.5 * tair), size=mp.Vector3(p, p, 0)))

    sim.load_minus_flux_data(refl, straight_refl_data)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            25, mp.Ey, mp.Vector3(0, 0, tm / 2.0 + 0.5 * tair), sim_eps))

    calc_refl_flux = mp.get_fluxes(refl)
    calc_tran_flux = mp.get_fluxes(tran)
    flux_freqs = mp.get_flux_freqs(refl)

    nfreq = 500
    fq = []
    wl = []
    Rs = []
    Ts = []
    for i in range(nfreq):
        fq = np.append(fq, flux_freqs[i])
        wl = np.append(wl, 1 / flux_freqs[i])
        Rs = np.append(
                Rs, -calc_refl_flux[i] / straight_tran_flux[i]
            )
        Ts = np.append(Ts, calc_tran_flux[i] / straight_tran_flux[i])
    Rs = np.array(Rs)
    return Rs


def random_x(num, seed=None):
    M = 4
    np.random.seed(seed)

    p0 = np.linspace(350, 500, 6)

    p = np.random.choice(p0, [num, 1])
    #p = np.transpose(p)
    pc = np.repeat(p, 16, axis=1) - 20

    theta = np.arange(0, np.pi / 2 + np.pi /
                      (2 * (M - 1)), np.pi / (2 * (M - 1)))
    resol = 20

    r = (pc / 2 - resol) * np.random.random([num, 16]) + resol

    ts = 20 + 100 * np.random.random([num, 1])  # height range [10, 100]
    tm = 10 + 70 * np.random.random([num, 1])
    x = np.zeros((num, 19))
    x[:, 0] = p.T
    x[:, 1] = ts.T
    x[:, 2] = tm.T
    x[:, 3:] = r
    return x


def simulate(x):
    with contextlib.redirect_stderr(io.StringIO()) as _stdio:
        with contextlib.redirect_stdout(io.StringIO()) as _stderr:
            y = _simulate(x)
    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    target, wavelength = load_target()
    print(wavelength)
    print(target)
    plt.plot(wavelength, target)
    plt.savefig("./tmp.jpg")
    # print(target)
    # print(wavelength)
    # from tqdm import tqdm
    # x = random_x(30)
    # # x[:, 0] = 350
    # import time
    # st = time.time()
    # # x = np.loadtxt("/home/yangjq/Working/tmp/model_TPV_Meep/generatedcontrol_13236.txt", skiprows=1)
    # xs = [x[i:i + 1] for i in range(x.shape[0])]
    # # _simulate(xs[0])
    # # exit()
    # # print(len(xs))
    # # exit()
    # ys = []
    # with multiprocessing.Pool(processes=30) as pool:
    #     for y in tqdm(pool.imap_unordered(simulate, xs), total=300):
    #         ys.append(y)
    #     # ys = pool.map(simulate, xs)
    # print(len(ys))
    # # print(ys)
    # et = time.time()
    # print("all time ", et - st)
    # print(xs)
    # exit()
    # print(x)
    # exit()
    # y = simulate(x)
    # print(y)

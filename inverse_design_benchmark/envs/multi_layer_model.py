import numpy as np
import scipy.constants as physconst
import scipy.interpolate as interp
import math
import cmath
import matplotlib.pyplot as plt

import pathlib

material_database_path = (pathlib.Path(__file__).parent.resolve().parent / 
                          "assets" / "multi_layer_model" / 
                          "Material_Database").resolve()
target_path = material_database_path.parent / "target.txt"

layer_material_range = ['ZnO', 'AlN', 'Al2O3', 'MgF2', 'SiO2', 'TiO2', 'SiC']
layer_thickness_range = (0.0, 1.0)
max_layer_num = 10

########################################################################################################################
# basic settings
angle = np.linspace(0, 84, 15)    # unit: degree
wavelength = np.linspace(0.3, 20, 2001)

########################################################################################################################

def load_target():
    target = np.loadtxt(target_path)
    return target, wavelength

########################################################################################################################
# transfer matrix method
def transfer_matrix_calculation(n, i, Air_background, layer_thickness, k):

    M_in = (1 / (2 * n[0][i])) * np.array([[n[0][i] + Air_background.index[i], n[0][i] - Air_background.index[i]]
                                              , [n[0][i] - Air_background.index[i],
                                                 n[0][i] + Air_background.index[i]]])  # air to structure
    M_mid = np.eye(2, 2)
    if len(layer_thickness) - 1 != 0:  # more than 1 layer
        for iM in range(0, len(layer_thickness) - 1):
            pr = np.array([[cmath.exp(n[iM][i] * k[iM][i] * layer_thickness[iM] * unit * 1j), 0], [0, cmath.exp(
                n[iM][i] * k[iM][i] * layer_thickness[iM] * unit * (-1) * 1j)]])  # propagate in layers
            cr = (1 / (2 * n[iM + 1][i])) * np.array([[n[iM + 1][i] + n[iM][i], n[iM + 1][i] - n[iM][i]],
                                                      [n[iM + 1][i] - n[iM][i],
                                                       n[iM + 1][i] + n[iM][i]]])  # crossover the interface
            M_mid = np.dot(cr, np.dot(pr, M_mid))
    p_ = np.array([[cmath.exp(n[-1][i] * k[-1][i] * layer_thickness[-1] * unit * 1j), 0],
                   [0, cmath.exp(n[-1][i] * k[-1][i] * layer_thickness[-1] * unit * (-1) * 1j)]])
    M_ = (1 / (2 * Air_background.index[i])) * np.array([[Air_background.index[i] + n[-1][i], Air_background.index[i] - n[-1][i]],
                                                [Air_background.index[i] - n[-1][i], Air_background.index[i] + n[-1][i]]])  # structure to air
    M = np.dot(M_, np.dot(p_, np.dot(M_mid, M_in)))
    return M

def modify_material(TE, ia, Air_background, angle, layer_material, database, k_o):
    if TE:
        Air_background.index = Air_background.index / Air_background.index[0]
        Air_background.index = Air_background.index * np.cos(angle[ia] / 180 * math.pi)
    else:
        Air_background.index = Air_background.index / Air_background.index[0]
        Air_background.index = Air_background.index / np.cos(angle[ia] / 180 * math.pi)

    n = np.zeros((len(layer_material), len(Air_background.index)), dtype=complex)
    k = np.zeros((len(layer_material), len(Air_background.index)), dtype=complex)
    ni = []
    judgement = []
    for i in range(len(layer_material)):
        j = 0
        while database[j].name != layer_material[i]:
            j = j + 1

        ni.append(database[j].index)
        judgement.append(np.sin(angle[ia] / 180 * math.pi) / database[j].index)
    ni = np.array(ni)
    judgement = np.array(judgement)

    for i in range(judgement.shape[0]):
        for ii in range(judgement.shape[1]):
            mod = cmath.sqrt(1 - judgement[i][ii] * judgement[i][ii])
            if TE:
                n[i][ii] = ni[i][ii] * mod
                k[i][ii] = k_o[ii]
            else:
                n[i][ii] = ni[i][ii] / mod
                k[i][ii] = k_o[ii] * mod * mod

    return ni, n, k, Air_background  #, Ag_background

# constants
unit = 1e-6
fre = physconst.c / (wavelength * unit) / 1e12
k_o = 2 * math.pi * fre * 1e12 / physconst.c
fl_fre = list(physconst.c / (wavelength * unit) / 1e12)
fl_fre = np.array(list(reversed(fl_fre)))
#############################################################################################
# material database


# 1.Air
class material_Air(object):
    def __init__(self):
        self.name = 'Air'
        index_Air = []
        for ma in range(len(wavelength)):
            index_Air.append(math.sqrt(1))
        self.index = index_Air
Air = material_Air()

class material_Air_background(object):
    def __init__(self):
        self.name = 'Air_b'
        index_Air_b = []
        for ma in range(len(wavelength)):
            index_Air_b.append(math.sqrt(1))
        index_Air_b = np.array(index_Air_b)
        self.index = index_Air_b
Air_background = material_Air_background()

# 2.SiO2
class material_SiO2(object):
    def __init__(self):
        self.name = 'SiO2'
        index_SiO2 = []
        ####################################################################
        # load experiment dataset(sorted) and interp
        # unit: fre:THz
        r_path = str(material_database_path / "SiO2" / "SiO2_Fit_eps_r.txt")
        i_path = str(material_database_path / "SiO2" / "SiO2_Fit_eps_i.txt")
        SiO2_Fit_eps_r = np.loadtxt(r_path)
        SiO2_Fit_eps_i = np.loadtxt(i_path)
        fr_SiO2 = interp.interp1d(SiO2_Fit_eps_r[:, 0], SiO2_Fit_eps_r[:, 1], kind='cubic')
        fi_SiO2 = interp.interp1d(SiO2_Fit_eps_i[:, 0], SiO2_Fit_eps_i[:, 1], kind='cubic')
        eps_r_interp_SiO2 = fr_SiO2(fl_fre)
        eps_i_interp_SiO2 = fi_SiO2(fl_fre)
        ####################################################################
        for ma in range(len(wavelength)):
            index_SiO2.append(cmath.sqrt(complex(eps_r_interp_SiO2[len(wavelength) - ma - 1], eps_i_interp_SiO2[len(wavelength) - ma - 1])))
        self.index = index_SiO2
SiO2 = material_SiO2()

# 3.TiO2
class material_TiO2(object):
    def __init__(self):
        self.name = 'TiO2'
        index_TiO2 = []
        ####################################################################
        # load experiment dataset(sorted) and interp
        # unit: fre:THz
        r_path = str(material_database_path / "TiO2" / "TiO2_Fit_eps_r.txt")
        i_path = str(material_database_path / "TiO2" / "TiO2_Fit_eps_i.txt")
        TiO2_Fit_eps_r = np.loadtxt(r_path)
        TiO2_Fit_eps_i = np.loadtxt(i_path)
        fr_TiO2 = interp.interp1d(TiO2_Fit_eps_r[:, 0], TiO2_Fit_eps_r[:, 1], kind='cubic')
        fi_TiO2 = interp.interp1d(TiO2_Fit_eps_i[:, 0], TiO2_Fit_eps_i[:, 1], kind='cubic')
        eps_r_interp_TiO2 = fr_TiO2(fl_fre)
        eps_i_interp_TiO2 = fi_TiO2(fl_fre)
        ####################################################################
        for ma in range(len(wavelength)):
            index_TiO2.append(cmath.sqrt(complex(eps_r_interp_TiO2[len(wavelength) - ma - 1], eps_i_interp_TiO2[len(wavelength) - ma - 1])))
        self.index = index_TiO2
TiO2 = material_TiO2()

# 4.silver (fixed as substrate)
class material_Ag(object):
    def __init__(self):
        self.name = 'Ag'
        index_Ag = []
        ####################################################################
        # load experiment dataset(sorted) and interp
        # unit: fre:THz
        r_path = str(material_database_path / "Ag" / "Ag_Fit_eps_r.txt")
        i_path = str(material_database_path / "Ag" / "Ag_Fit_eps_i.txt")
        Ag_Fit_eps_r = np.loadtxt(r_path)
        Ag_Fit_eps_i = np.loadtxt(i_path)
        fr_Ag = interp.interp1d(Ag_Fit_eps_r[:, 0], Ag_Fit_eps_r[:, 1], kind='cubic')
        fi_Ag = interp.interp1d(Ag_Fit_eps_i[:, 0], Ag_Fit_eps_i[:, 1], kind='cubic')
        eps_r_interp_Ag = fr_Ag(fl_fre)
        eps_i_interp_Ag = fi_Ag(fl_fre)
        ####################################################################
        for ma in range(len(wavelength)):
            index_Ag.append(cmath.sqrt(complex(eps_r_interp_Ag[len(wavelength) - ma - 1], eps_i_interp_Ag[len(wavelength) - ma - 1])))
        self.index = index_Ag
Ag = material_Ag()

# 5.SiC
class material_SiC(object):
    def __init__(self):
        self.name = 'SiC'
        index_SiC = []
        ####################################################################
        # load experiment dataset(sorted) and interp
        # unit: fre:THz
        r_path = str(material_database_path / "SiC" / "SiC_Fit_eps_r.txt")
        i_path = str(material_database_path / "SiC" / "SiC_Fit_eps_i.txt")
        SiC_Fit_eps_r = np.loadtxt(r_path)
        SiC_Fit_eps_i = np.loadtxt(i_path)
        fr_SiC = interp.interp1d(SiC_Fit_eps_r[:, 0], SiC_Fit_eps_r[:, 1], kind='cubic')
        fi_SiC = interp.interp1d(SiC_Fit_eps_i[:, 0], SiC_Fit_eps_i[:, 1], kind='cubic')
        eps_r_interp_SiC = fr_SiC(fl_fre)
        eps_i_interp_SiC = fi_SiC(fl_fre)
        ####################################################################
        for ma in range(len(wavelength)):
            index_SiC.append(cmath.sqrt(complex(eps_r_interp_SiC[len(wavelength) - ma - 1], eps_i_interp_SiC[len(wavelength) - ma - 1])))
        self.index = index_SiC
SiC = material_SiC()

# 6.MgF2
class material_MgF2(object):
    def __init__(self):
        self.name = 'MgF2'
        index_MgF2 = []
        ####################################################################
        # load experiment dataset(sorted) and interp
        # unit: fre:THz
        r_path = str(material_database_path / "MgF2" / "MgF2_Fit_eps_r.txt")
        i_path = str(material_database_path / "MgF2" / "MgF2_Fit_eps_i.txt")
        MgF2_Fit_eps_r = np.loadtxt(r_path)
        MgF2_Fit_eps_i = np.loadtxt(i_path)
        fr_MgF2 = interp.interp1d(MgF2_Fit_eps_r[:, 0], MgF2_Fit_eps_r[:, 1], kind='cubic')
        fi_MgF2 = interp.interp1d(MgF2_Fit_eps_i[:, 0], MgF2_Fit_eps_i[:, 1], kind='cubic')
        eps_r_interp_MgF2 = fr_MgF2(fl_fre)
        eps_i_interp_MgF2 = fi_MgF2(fl_fre)
        ####################################################################
        for ma in range(len(wavelength)):
            index_MgF2.append(cmath.sqrt(complex(eps_r_interp_MgF2[len(wavelength) - ma - 1], eps_i_interp_MgF2[len(wavelength) - ma - 1])))
        self.index = index_MgF2
MgF2 = material_MgF2()

# 7.Al2O3
class material_Al2O3(object):
    def __init__(self):
        self.name = 'Al2O3'
        index_Al2O3 = []
        ####################################################################
        # load experiment dataset(sorted) and interp
        # unit: fre:THz
        r_path = str(material_database_path / "Al2O3" / "Al2O3_Fit_eps_r.txt")
        i_path = str(material_database_path / "Al2O3" / "Al2O3_Fit_eps_i.txt")
        Al2O3_Fit_eps_r = np.loadtxt(r_path)
        Al2O3_Fit_eps_i = np.loadtxt(i_path)
        fr_Al2O3 = interp.interp1d(Al2O3_Fit_eps_r[:, 0], Al2O3_Fit_eps_r[:, 1], kind='cubic')
        fi_Al2O3 = interp.interp1d(Al2O3_Fit_eps_i[:, 0], Al2O3_Fit_eps_i[:, 1], kind='cubic')
        eps_r_interp_Al2O3 = fr_Al2O3(fl_fre)
        eps_i_interp_Al2O3 = fi_Al2O3(fl_fre)
        ####################################################################
        for ma in range(len(wavelength)):
            index_Al2O3.append(cmath.sqrt(complex(eps_r_interp_Al2O3[len(wavelength) - ma - 1], eps_i_interp_Al2O3[len(wavelength) - ma - 1])))
        self.index = index_Al2O3
Al2O3 = material_Al2O3()

# 8.AlN
class material_AlN(object):
    def __init__(self):
        self.name = 'AlN'
        index_AlN = []
        ####################################################################
        # load experiment dataset(sorted) and interp
        # unit: fre:THz
        r_path = str(material_database_path / "AlN" / "AlN_Fit_eps_r.txt")
        i_path = str(material_database_path / "AlN" / "AlN_Fit_eps_i.txt")
        AlN_Fit_eps_r = np.loadtxt(r_path)
        AlN_Fit_eps_i = np.loadtxt(i_path)
        fr_AlN = interp.interp1d(AlN_Fit_eps_r[:, 0], AlN_Fit_eps_r[:, 1], kind='cubic')
        fi_AlN = interp.interp1d(AlN_Fit_eps_i[:, 0], AlN_Fit_eps_i[:, 1], kind='cubic')
        eps_r_interp_AlN = fr_AlN(fl_fre)
        eps_i_interp_AlN = fi_AlN(fl_fre)
        ####################################################################
        for ma in range(len(wavelength)):
            index_AlN.append(cmath.sqrt(complex(eps_r_interp_AlN[len(wavelength) - ma - 1], eps_i_interp_AlN[len(wavelength) - ma - 1])))
        self.index = index_AlN
AlN = material_AlN()

# 9.ZnO
class material_ZnO(object):
    def __init__(self):
        self.name = 'ZnO'
        index_ZnO = []
        ####################################################################
        # load experiment dataset(sorted) and interp
        # unit: fre:THz
        r_path = str(material_database_path / "ZnO" / "ZnO_Fit_eps_r.txt")
        i_path = str(material_database_path / "ZnO" / "ZnO_Fit_eps_i.txt")
        ZnO_Fit_eps_r = np.loadtxt(r_path)
        ZnO_Fit_eps_i = np.loadtxt(i_path)
        fr_ZnO = interp.interp1d(ZnO_Fit_eps_r[:, 0], ZnO_Fit_eps_r[:, 1], kind='cubic')
        fi_ZnO = interp.interp1d(ZnO_Fit_eps_i[:, 0], ZnO_Fit_eps_i[:, 1], kind='cubic')
        eps_r_interp_ZnO = fr_ZnO(fl_fre)
        eps_i_interp_ZnO = fi_ZnO(fl_fre)
        ####################################################################
        for ma in range(len(wavelength)):
            index_ZnO.append(cmath.sqrt(complex(eps_r_interp_ZnO[len(wavelength) - ma - 1], eps_i_interp_ZnO[len(wavelength) - ma - 1])))
        self.index = index_ZnO
ZnO = material_ZnO()

# whole database
database = []
database.append(Air)
database.append(SiO2)
database.append(TiO2)
database.append(Ag)
database.append(SiC)
database.append(MgF2)
database.append(Al2O3)
database.append(AlN)
database.append(ZnO)

########################################################################################################################
# input control parameters
# layer_material = ['ZnO', 'AlN', 'Al2O3', 'MgF2', 'SiO2', 'TiO2', 'Ag']
# layer_thickness = [0.055, 0.2, 0.1, 0.24, 0.3, 0.2, 0.1]    # unit: um
def simulate(layer_material, layer_thickness):
    layer_material.append("Ag")
    layer_thickness.append(0.1)
    Air_background = material_Air_background()
    #######################################################################################################################
    FR = []
    # find material and recond in sequence
    for TE in range(0, 2):  # First TM, Then TE
        for ia in range(len(angle)):
            ###############################################################################################################
            # step 1: modify material index due to angle
            ni, n, k, Air_background = modify_material(TE, ia, Air_background, angle, layer_material, database, k_o)

            ###########################################################################################################
            # step 2: calculate spectrum
            S21 = []
            S22 = []
            S11 = []
            S12 = []
            for i in range(len(wavelength)):
                # transfer matrix 1: calculate M
                n_f = n
                k_f = k
                layer_thickness_f = layer_thickness

                M = transfer_matrix_calculation(n=n_f, i=i, Air_background=Air_background, layer_thickness=layer_thickness_f, k=k_f)

                # transfer matrix 2: M to S
                S21.append(np.conj((M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) / M[1, 1]))
                S22.append(np.conj(M[0, 1] / M[1, 1]))
                S11.append(np.conj(-M[1, 0] / M[1, 1]))
                S12.append(np.conj(1 / M[1, 1]))

            S11 = np.array(S11)
            S21 = np.array(S21)
            S12 = np.array(S12)
            S22 = np.array(S22)
            Forward_Reflectivity = abs(S11) * abs(S11) # Forward incidence of wave
            FR.append(Forward_Reflectivity)
            Forward_Transmissivity = abs(S21) * abs(S21) # Forward incidence of wave
            Backward_Reflectivity = abs(S12) * abs(S12) # Backward incidence of wave
            Backward_Transmissivity = abs(S22) * abs(S22) # Backward incidence of wave

    Average_Reflection = np.sum(np.array(FR), axis=0) / 2 / len(angle) # 2 means TE and TM
    return Average_Reflection

######################################################################################################################
# # plot
def viz(Average_Reflection, save_path):
    plt.figure(50)
    print(wavelength)
    plt.plot(wavelength, np.reshape(Average_Reflection, (len(wavelength),)))
    plt.title('Emissivity')
    plt.xlabel('wavelength(um)')
    plt.xlim(wavelength[0], wavelength[-1])
    plt.ylim(0, 1)
    # plt.show()
    plt.savefig(save_path)

if __name__ == "__main__":
    layer_material = ['ZnO', 'AlN', 'Al2O3', 'MgF2', 'SiO2', 'TiO2', 'Ag']
    layer_thickness = [0.055, 0.2, 0.1, 0.24, 0.3, 0.2, 0.1]    # unit: um
    # layer_material = ['ZnO', 'AlN', 'Al2O3', 'MgF2', 'SiO2', 'TiO2', 'Ag']
    # layer_thickness = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]    # unit: um
    target = load_target()
    print(target.shape)
    # res = simulate(layer_material=layer_material, layer_thickness=layer_thickness)
    # print(res.shape)
    viz(target, "./tmp_target.jpg")
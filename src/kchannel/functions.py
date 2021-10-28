import numpy as np
import MDAnalysis as mda
import pandas as pd
import time
import os
import logging
import sys
from functools import wraps

def countTime(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.time()
            print(f"Total execution time: {end - start:.5f} s\n\n")
    return _time_it

def createLogger(loc):
    # remove existing handlers, if any
    logger = logging.getLogger('kchannel')
    logger.handlers = []

    logger = logging.getLogger('kchannel')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(loc, mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger

def detectSF(coor, quiet=False, o_cutoff=4.75, og1_cutoff=6.0):
    """ read coordinate file and return zero-indexed atom indices defining SF

    Parameters
    ----------
    coor : str
        The path to the coordinate file containing the ion channel
    quiet : bool, optional
        A flag used to print detailed information about SF (default is False)
    o_cutoff: float
        Distance cutoff for determining if backbone oxygen atoms are close enough to meet the geometric
        requirement for being a part of SF
    og1_cutoff: float
        Same as o_cutoff, but it is for OG1 atom (like threonine's OG1)

    Returns
    -------
    sf_layer: dict
        consisting of two dictionaries containing atom idx (zero-based) and residue id of
        SF's oxygen (backbone and hydroxyl) and CA atoms respectively

    """
    u = mda.Universe(coor, in_memory=False)

    # Finding backbone oxygen atoms that meet the geometric requirements
    protein_o = u.select_atoms(f"protein and name O and (resname GLY VAL TYR THR PHE ILE)", updating=False)



    n_o = len(protein_o)
    n_neighbors_resid = {}
    sf_o_resid = []

    for i in range(n_o):
        for j in range(i+1, n_o):
            if abs(protein_o[i].resid - protein_o[j].resid) > 2 or protein_o[i].resid == protein_o[j].resid:
                if np.linalg.norm(protein_o[i].position - protein_o[j].position) < o_cutoff:
                    for key in (protein_o[i].resid, protein_o[j].resid):
                        if key not in n_neighbors_resid:
                            n_neighbors_resid[key] = 1
                        else:
                            n_neighbors_resid[key] += 1

    for key in n_neighbors_resid.keys():
        if n_neighbors_resid[key] > 2:
            if ((n_neighbors_resid.get(key+1) or 0) or (n_neighbors_resid.get(key-1) or 0)) and \
            ((n_neighbors_resid.get(key+2) or 0) or (n_neighbors_resid.get(key-2) or 0)):
                sf_o_resid.append(key)

    sf_o = u.select_atoms(f"protein and name O and resid {' '.join([str(s) for s in sf_o_resid])}", updating=False)

    # +1 to include the next residue after SF residues
    sf_ca_resid = sf_o_resid
    sf_ca = u.select_atoms(f"protein and name CA and resid {' '.join([str(s) for s in sf_ca_resid])}", updating=False)


    # Finding hydroxyl oxygen atoms (OG1) that meet the geometric requirements
    protein_og1 = u.select_atoms(f"protein and name OG1", updating=False)

    n_og1 = len(protein_og1)
    n_neighbors_atomid = {}
    sf_og1_atomid = []

    for i in range(n_og1):
        for j in range(i+1, n_og1):
            if abs(protein_og1[i].id - protein_og1[j].id) > 2:
                if np.linalg.norm(protein_og1[i].position - protein_og1[j].position) < og1_cutoff:
                    for key in (protein_og1[i].id, protein_og1[j].id):
                        if key not in n_neighbors_atomid:
                            n_neighbors_atomid[key] = 1
                        else:
                            n_neighbors_atomid[key] += 1

    for key in n_neighbors_atomid.keys():
        if n_neighbors_atomid[key] > 2:
            sf_og1_atomid.append(key)

    sf_og1 = u.select_atoms(f"protein and bynum  {' '.join([str(s) for s in sf_og1_atomid])}", updating=False)

    #bynum -> one-index
    #index -> zero-index
    sf_o = u.select_atoms(f"bynum  {' '.join([str(atom.id) for atom in sf_o+sf_og1])}", updating=False)
    sf_ca = u.select_atoms(f"bynum  {' '.join([str(atom.id) for atom in sf_ca])}", updating=False)



    sf_layer = {'O':{}, 'CA':{}}
    for sf_atom, name in zip([sf_o, sf_ca], ['O', 'CA']):
        # number of layer = # atoms // 4
        layer_id = len(sf_atom) // 4 - 1
        for atom in sf_atom:
            if layer_id not in sf_layer[name].keys():
                sf_layer[name][layer_id] = {'idx': [atom.ix], 'resid':[atom.resid],
                                           'resname':[atom.resname], 'name':[atom.name]}
            else:
                sf_layer[name][layer_id]['idx'].append(atom.ix)
                sf_layer[name][layer_id]['resid'].append(atom.resid)
                sf_layer[name][layer_id]['resname'].append(atom.resname)
                sf_layer[name][layer_id]['name'].append(atom.name)
            layer_id = layer_id-1 if layer_id != 0 else (len(sf_atom) // 4 - 1)

    # rearrange order so that the first atom is the diagonal pair of the second atom,
    # and the third is the diagonal pair of the fourth one
    sf_ca_0_pos = u.select_atoms('all').positions[sf_layer['CA'][0]['idx']]
    i_oppositeTo0 = np.linalg.norm(sf_ca_0_pos - sf_ca_0_pos[0], axis=1).argmax()

    if not quiet:
        print(f"idx\tlayer\tchain\tresid\tresname\tname")
    for atomtype in sf_layer.keys():
        for layer in sf_layer[atomtype].keys():
            for info in sf_layer[atomtype][layer].keys():
                sf_layer[atomtype][layer][info][1], sf_layer[atomtype][layer][info][i_oppositeTo0] = \
                sf_layer[atomtype][layer][info][i_oppositeTo0], sf_layer[atomtype][layer][info][1]
            if atomtype == 'O' and not quiet:
                for chain_id, idx in enumerate(sf_layer[atomtype][layer]['idx']):
                    resid = sf_layer[atomtype][layer]['resid'][chain_id]
                    resname = sf_layer[atomtype][layer]['resname'][chain_id]
                    name = sf_layer[atomtype][layer]['name'][chain_id]
                    print(f"{idx}\t{layer}\t{chain_id}\t{resid}\t{resname}\t{name}")

    return sf_layer

def getNonProteinIndex(coor):
    """ read coordinate file and return zero-indexed atom indices defining SF

    Parameters
    ----------
    coor : str
        The path to the coordinate file containing the system

    Returns
    -------
    indices: tuple
        arrays of zero-indexed indices of K, Cl and water respectively

    """
    u = mda.Universe(coor, in_memory=False)

    indices = (u.select_atoms(f"resname K", updating=False).ix,
           u.select_atoms(f"resname CL", updating=False).ix,
           u.select_atoms(f"resname SOL and name OW", updating=False).ix)

    return indices

def findBound(positions, sol_idx, bs_layer_idx, additional_bs_cutoff=4.0,
              d_min_inner_cutoff=4.0, d_min_outer_cutoff=4.0):
    """ read soluent (K) or solvent (water) positions and assign their
    (zero-based) indicies to binding sites

    Parameters
    ----------
    coor : str
        The path to the coordinate file containing the system
    sol_idx : array of int
        containing zero-based indices of particles such as oxygen of water, K and Cl
    bs_layer_idx : 2-dimensional array of int
        containing zero-based indices of atoms of layers (from SF) defining the binding sites
        e.g.
            bs_layer_idx[0] contains idx of atoms of layers forming the first boundary of the first binding site
            bs_layer_idx[1] contains idx of atoms of layers forming the second boundary of the first binding site
                            + the first boundary of the second binding site
    additional_bs_cutoff: float
        cutoff distance (in z-direction) in Angstrom for S0 and Scav
    d_min_inner_cutoff: float
        threshold distance in Angstrom for determining if the particle is touching and thus occupying
        inner (neither the first nor the last binding sites) BSs
    d_min_outer_cutoff: float
        threshold distance in Angstrom for determining if the particle is touching and thus occupying
        outer (the first or the last binding sites) BSs

    Returns
    -------
    occupancy: list of lists
        # lists = # binding sites
        each of the lists contains (zero-based) indices of particles occupying the corresponding BS

    """
    # TODO: account for channel moving across period boundary (when channel moves across PB in z-direction)

    bs_layer_pos = np.array([positions[indices] for indices in bs_layer_idx])
    bs_layer_pos_z = np.mean(bs_layer_pos, axis=1)[:, 2]
    sol_pos_z = positions[sol_idx][:, 2]

    # check if it is sorted in descending order which is needed
    # this ensures that first and second indices define the first BS (S0)
    assert all(np.sort(bs_layer_pos_z)[::-1] == bs_layer_pos_z)

    # adding S0 and Scav boundary
    bs_full_pos_z = np.array([bs_layer_pos_z[0] + additional_bs_cutoff] +
                             list(bs_layer_pos_z) +
                             [bs_layer_pos_z[-1] - additional_bs_cutoff])

    # keep indices and pos_z only for sol within the range of bs_full_pos_z
    bound_idx = np.argwhere((sol_pos_z < bs_full_pos_z[0]) &
                          (sol_pos_z > bs_full_pos_z[-1])).reshape(-1)
    sol_idx = sol_idx[bound_idx]
    sol_pos_z = sol_pos_z[bound_idx]


    # number of binding sites
    n_bs = len(bs_full_pos_z) - 1
    # sol_in_bs[i] == 0 when i-th particle, which is unbound, is "above" the first BS (S0)
    # sol_in_bs[i] == j when i-th particle, which is unbound, is bound by the (j-1)-th BS,
    #                   for j between 1 and < n_bs
    # sol_in_bs[i] == n_bs when i-th particle, which is unbound, is "below" the last BS (Scav)
    sol_in_bs = np.searchsorted(-bs_full_pos_z, -sol_pos_z)

    occupancy = [[] for i in range(n_bs)]

    for idx, i in zip(sol_idx, sol_in_bs):
        # switching from boundaries to BSs
        if i < 1 or i > n_bs:
            continue
        bs_i = i-1

        if bs_i == 0:
            # check with lower layer, need at least two SF atoms touching the particle
            d = np.linalg.norm(positions[idx] - bs_layer_pos[bs_i], axis=1)
            # touching one of the SF oxygen is ok
            if np.sum(d < d_min_outer_cutoff) > 0:
                occupancy[bs_i].append(idx)

        elif bs_i == n_bs-1:
            # check with the upper layer instead
            d  = np.linalg.norm(positions[idx] - bs_layer_pos[bs_i-1], axis=1)
            # touching one of the SF oxygen is ok
            if np.sum(d < d_min_outer_cutoff) > 0:
                occupancy[bs_i].append(idx)

        else:
            # upper layer
            d_u = np.linalg.norm(positions[idx] - bs_layer_pos[bs_i-1], axis=1)
            # lower layer
            d_l = np.linalg.norm(positions[idx] - bs_layer_pos[bs_i], axis=1)
            # need to touch two of them
            if np.sum(d_u < d_min_inner_cutoff) + np.sum(d_l < d_min_inner_cutoff) > 1:
                occupancy[bs_i].append(idx)

    return occupancy

def checkFlips(pos_all, sf_o_idx, cutoff=5):
    """ check SF oxygen flips in a given frame

    Parameters
    ----------
    pos_all : narray
        positions of all atoms in the system
    sf_o_idx : narray
        indices for oxygen atoms in each layer
    cutoff : float
        cutoff distance for determining if a flip occurs

    Returns
    -------
    flips: array
        number of flips in each layer. flips[0] is the # flips for the first layer
    """
    flips = np.zeros(len(sf_o_idx), dtype=int)
    n_o = len(sf_o_idx[0])

    for i, layer in enumerate(sf_o_idx):
        pos = pos_all[layer]
        flip = [(np.linalg.norm(pos - pos[j], axis=1).sum() / (n_o-1)) > cutoff
                for j in range(len(layer))]
        flips[i] = np.sum(flip)

    return flips

def computeSFAtomDistance(pos_all, sf_atom_idx):
    """ calculate diagonal distance of SF atoms, e.g. O and CA

    Parameters
    ----------
    pos_all : narray
        positions of all atoms in the system
    sf_atom_idx : narray
        indices for atoms of interest in each layer. The first index and second index are
        in opposite pair. Same for the third and the fourth.

    Returns
    -------
    d: array
        all computed diagonal distances
    """
    d = []
    for layer_id, indices in enumerate(sf_atom_idx):
        pos = pos_all[indices]
        d.append(np.linalg.norm(pos[0] - pos[1]))
        d.append(np.linalg.norm(pos[2] - pos[3]))
    d = np.array(d)
    return d

def computeOccupancy_6BS(k_occ_whole, w_occ_whole, bs_ignore=[0, 5]):
    """ compute SF occupancy labels

    Parameters
    ----------
    k_occ_whole: list of lists of int
        k_occ_whole[i][j] contains the indicies of K+ bound in j-th BS for frame i

    w_occ_whole: list of lists of int
        w_occ_whole[i][j] contains the indicies of water bound in j-th BS for frame i

    bs_ignore: list of int
        contains indices of BSs from which no warning about double occupancy is reported

    Returns
    -------
    occ_whole: narray
       occ_whole[i] is the occupancy label for frame i

    occ_whole: list of tuples
       frame number and BS that double occupancy occurs

    """

    n_bs = len(k_occ_whole[0])
    assert n_bs == 6

    occ_whole = []
    double_occ = []

    for t, (k_occ, w_occ) in enumerate(zip(k_occ_whole, w_occ_whole)):
        occ = ['0' for j in range(n_bs)]
        for i in range(len(occ)):
            if len(k_occ[i]) > 0 and len(w_occ[i]) > 0:
                occ[i] = 'C'
                if i not in bs_ignore:
                    #print(f'At frame {t}, double occupancy in S{i}')
                    double_occ.append((t, i))
            elif len(k_occ[i]) > 0:
                occ[i] = 'K'
            elif len(w_occ[i]) > 0:
                occ[i] = 'W'
        occ_whole.append(''.join(occ))

    occ_whole = np.array(occ_whole)
    return occ_whole, double_occ

def computeJumps_6BS(k_occ_all, w_occ_all):
    """ compute ion and water net jump for the whole trajectory

    Parameters
    ----------
    k_occ_all: list of lists
        # lists = # binding sites
        each of the lists contains indices of ion occupying the corresponding binding site

    w_occ_all: list of lists
        same as above, except that it is for water

    Returns
    -------
    jumps: narray
        jumps[i, 0] saves # net jumps of ion occurred between i-th step and (i+1)-th step
        jumps[i, 1] saves # net jumps of water occurred between i-th step and (i+1)-th step

    """

    jumps = np.zeros((len(k_occ_all)-1, 2), dtype=int)
    for i in range(len(jumps)):
        jumps[i, 0] = computeJump_6BS(k_occ_all[i], k_occ_all[i+1], t0=i)
        jumps[i, 1] = computeJump_6BS(w_occ_all[i], w_occ_all[i+1], t0=i)
    return jumps

def computeJump_6BS(occ_t0, occ_t1, t0=0.0):
    """ compute net number of jumps given the current and the next occupation states

    Parameters
    ----------
    occ_t0 : list of lists of int
        each list contains atom idx that occupies the corresponding binding site at t=t0
        e.g. occ_t0 = [[], [], [33577], [33596], [], []]
    occ_t1 : list of lists of int
        each list contains atom idx that occupies the corresponding binding site at t=t1
        e.g. occ_t1 = [[33577], [], [], [33596], [], []]
    t0 : float (optional)
        current timestep, only used for error report/debug

    Returns
    -------
    jump: int
        net number of jumps given the current and the next occupation states
    """
    # only work for 6 binding sites
    assert len(occ_t0) == 6

    jump = 0
    checked = []

    # check occ_t1 for new positions of particles present in occ_t0
    for bs_i_t0, bs_t0 in enumerate(occ_t0):
        for sol_i_t0 in bs_t0:
            new_pos = [i for (i, bs_t1) in enumerate(occ_t1) if sol_i_t0 in bs_t1]
            if len(new_pos) > 1:
                print(f"At time {time}, same solvent/solute is identified more than once")
                raise Exception
            elif len(new_pos) == 0:
                if bs_i_t0 == 2 or bs_i_t0 == 3:
                    print(f"At step {t0}, new position of idx {sol_i_t0} in S{bs_i_t0} cannot be found, jump too much?")
                    raise Exception
                elif bs_i_t0 == 0 or bs_i_t0 == 1:
                    # assume it has escaped through S0, so set it beyond S0
                    new_pos_idx = -1
                else:
                    # assume it has escaped through Scav, so set it beyond Scav
                    new_pos_idx = 6
            else:
                new_pos_idx = new_pos[0]
            # jump > 0 means forward, toward S0
            jump += bs_i_t0 - new_pos_idx

            if sol_i_t0 not in checked:
                checked.append(sol_i_t0)
            else:
                print(f"At step {time}, {sol_i_t0} was found twice")
                raise Exception

    for bs_i_t1, bs_t1 in enumerate(occ_t1):
        for sol_i_t1 in bs_t1:
            if sol_i_t1 in checked:
                continue
            else:
                if bs_i_t1 == 2 or bs_i_t1 == 3:
                    print(f"At step {t0+1}, history of idx {sol_i_t1} which is found in S2/S3 \
    cannot be traced, jump too much?")
                    raise Exception
                elif bs_i_t1 == 0 or bs_i_t1 == 1:
                    # assume it has entered through S0, so set it beyond S0
                    old_pos_idx = -1
                else:
                    # assume it has entered through Scav, so set it beyond Scav
                    old_pos_idx = 6
                jump += old_pos_idx - bs_i_t1
    return jump

def computeJumps_6BS_ignoreS0Scav(k_occ_all, w_occ_all):
    """ compute ion and water net jump for the whole trajectory
        *** it ignores jump across S0 and Scav (S5) ***
        e.g. t_0 = [[],[],[],[],[],[]]
             t_1 = [[],[],[],[],[2],[]]
             jump = 1

    Parameters
    ----------
    k_occ_all: list of lists
        # lists = # binding sites
        each of the lists contains indices of ion occupying the corresponding binding site

    w_occ_all: list of lists
        same as above, except that it is for water

    Returns
    -------
    jumps: narray
        jumps[i, 0] saves # net jumps of ion occurred between i-th step and (i+1)-th step
        jumps[i, 1] saves # net jumps of water occurred between i-th step and (i+1)-th step

    """

    jumps = np.zeros((len(k_occ_all)-1, 2), dtype=int)
    for i in range(len(jumps)):
        jumps[i, 0] = computeJump_6BS_ignoreS0Scav(k_occ_all[i], k_occ_all[i+1], t0=i)
        jumps[i, 1] = computeJump_6BS_ignoreS0Scav(w_occ_all[i], w_occ_all[i+1], t0=i)
    return jumps

def computeJump_6BS_ignoreS0Scav(occ_t0, occ_t1, t0=0.0):
    """ compute net number of jumps given the current and the next occupation states
        *** it ignores jump across S0 and Scav (S5) ***

    Parameters
    ----------
    occ_t0 : list of lists of int
        each list contains atom idx that occupies the corresponding binding site at t=t0
        e.g. occ_t0 = [[], [], [33577], [33596], [], []]
    occ_t1 : list of lists of int
        each list contains atom idx that occupies the corresponding binding site at t=t1
        e.g. occ_t1 = [[33577], [], [], [33596], [], []]
    t0 : float (optional)
        current timestep, only used for error report/debug

    Returns
    -------
    jump: int
        net number of jumps given the current and the next occupation states
    """
    # only work for 6 binding sites
    assert len(occ_t0) == 6

    jump = 0
    checked = []

    # check occ_t1 for new positions of particles present in occ_t0
    for bs_i_t0, bs_t0 in enumerate(occ_t0):
        for sol_i_t0 in bs_t0:
            new_pos = [i for (i, bs_t1) in enumerate(occ_t1) if sol_i_t0 in bs_t1]
            if len(new_pos) > 1:
                print(f"At time {time}, same solvent/solute is identified more than once")
                raise Exception
            elif len(new_pos) == 0:
                if bs_i_t0 == 2 or bs_i_t0 == 3:
                    print(f"At step {t0}, new position of idx {sol_i_t0} in S{bs_i_t0} cannot be found, jump too much?")
                    raise Exception
                elif bs_i_t0 == 0 or bs_i_t0 == 1:
                    # assume it has escaped through S0, so set it beyond S0
                    new_pos_idx = 0
                else:
                    # assume it has escaped through Scav, so set it beyond Scav
                    new_pos_idx = 5
            else:
                new_pos_idx = new_pos[0]
            # jump > 0 means forward, toward S0
            jump += bs_i_t0 - new_pos_idx

            if sol_i_t0 not in checked:
                checked.append(sol_i_t0)
            else:
                print(f"At step {time}, {sol_i_t0} was found twice")
                raise Exception

    for bs_i_t1, bs_t1 in enumerate(occ_t1):
        for sol_i_t1 in bs_t1:
            if sol_i_t1 in checked:
                continue
            else:
                if bs_i_t1 == 2 or bs_i_t1 == 3:
                    print(f"At step {t0+1}, history of idx {sol_i_t1} which is found in S2/S3 \
    cannot be traced, jump too much?")
                    raise Exception
                elif bs_i_t1 == 0 or bs_i_t1 == 1:
                    # assume it has entered through S0, so set it beyond S0
                    old_pos_idx = 0
                else:
                    # assume it has entered through Scav, so set it beyond Scav
                    old_pos_idx = 5
                jump += old_pos_idx - bs_i_t1
    return jump

@countTime
def run(coor, traj, CADistance=False, ignoreS0ScavJump=True):
    path = os.path.dirname(traj)

    log_loc = os.path.join(path, 'results.log')

    # remove handlers if any
    logger = createLogger(log_loc)

    u = mda.Universe(coor, traj, in_memory=False)

    sf_idx = detectSF(coor, quiet=True)
    sf_o_idx = np.array([sf_idx['O'][i]['idx'] for i in range(len(sf_idx['O']))])
    sf_ca_idx = np.array([sf_idx['CA'][i]['idx'] for i in range(len(sf_idx['CA']))])

    k_idx, cl_idx, water_idx = getNonProteinIndex(coor)

    k_occupancy = []
    w_occupancy = []

    occupancy = np.zeros(len(u.trajectory), dtype='<6U')
    jumps = np.zeros((len(u.trajectory), 2), dtype=int)
    flips = np.zeros((len(u.trajectory), len(sf_o_idx)), dtype=int)

    if CADistance:
        # start with first BS (S0) x2, end with the last BS x2
        o_d_diag = np.zeros((len(u.trajectory), 2*len(sf_o_idx)), dtype=float)
        ca_d_diag = np.zeros((len(u.trajectory), 2*len(sf_ca_idx)), dtype=float)

    print("Reading trajectory")
    for ts in u.trajectory:
        if CADistance:
            o_d_diag[ts.frame] = computeSFAtomDistance(ts.positions, sf_o_idx)
            ca_d_diag[ts.frame] = computeSFAtomDistance(ts.positions, sf_ca_idx)

        flips[ts.frame] = checkFlips(ts.positions, sf_o_idx)
        k_occ = findBound(ts.positions, k_idx, sf_o_idx)
        w_occ = findBound(ts.positions, water_idx, sf_o_idx)
        k_occupancy.append(k_occ)
        w_occupancy.append(w_occ)
        if ts.frame % 1000 == 0:
            print('\r'+f'Finished processing frame {ts.frame} / {len(u.trajectory)}', end=' ')
    print("")
    occupancy[:len(k_occupancy)], double_occ = computeOccupancy_6BS(k_occupancy, w_occupancy)
    logger.info(f"Double occupancy at found in {len(double_occ)} frames. Check log file for details.")
    for t, i in double_occ:
        logger.debug(f"At frame {t}, double occupancy in S{i}")

    if ignoreS0ScavJump:
        jumps[:len(k_occupancy)-1] = computeJumps_6BS_ignoreS0Scav(k_occupancy, w_occupancy)
    else:
        jumps[:len(k_occupancy)-1] = computeJumps_6BS(k_occupancy, w_occupancy)

    if ignoreS0ScavJump:
        n_k_netjumps = np.sum(jumps[:, 0]) // 5 #(len(occupancy[0]) + 1)
        n_w_netjumps = np.sum(jumps[:, 1]) // 5 #(len(occupancy[0]) + 1)
    else:
        n_k_netjumps = np.sum(jumps[:, 0]) // (len(occupancy[0]) + 1)
        n_w_netjumps = np.sum(jumps[:, 1]) // (len(occupancy[0]) + 1)
    current = n_k_netjumps * 1.602e-19 / (u.trajectory.totaltime*1e-12) * 1e12
    logger.info("=================================")
    logger.info(f"Total time: {u.trajectory.totaltime/1e3:.6f} ns")
    logger.info(f"dt: {u.trajectory.dt/1e3:.6f} ns")
    logger.info(f"Number of K+: {len(k_idx)}")
    logger.info(f"Number of water: {len(water_idx)}")
    logger.info(f"Number of net water permeation = {n_w_netjumps}")
    logger.info(f"Number of net ion permeation = {n_k_netjumps}")
    logger.info(f"Current = {current:.5f} pA")


    if CADistance:
        data = np.hstack((occupancy.reshape(-1, 1), jumps, flips, ca_d_diag, o_d_diag)).astype("<8U")
        columns = ['occupancy', 'j_k', 'j_w'] + [f"flip_{i}" for i in range(len(sf_o_idx))] + \
        [f"{sf_idx['CA'][i]['name'][0]}_{sf_idx['CA'][i]['resid'][0]}_{j}" for i in range(len(sf_idx['CA'])) for j in range(2)] + \
        [f"{sf_idx['O'][i]['name'][0]}_{sf_idx['O'][i]['resid'][0]}_{j}" for i in range(len(sf_idx['O'])) for j in range(2)]
    else:
        data = np.hstack((occupancy.reshape(-1, 1), jumps, flips)).astype("<8U")
        columns = ['occupancy', 'j_k', 'j_w'] + [f'flip_{i}' for i in range(len(sf_o_idx))]

    data_loc = os.path.join(path, 'results.csv')
    data = pd.DataFrame(data, columns=columns)
    _ = data.to_csv(data_loc)
    logger.info(f"Results saved to {data_loc}")
    logger.info(f"Log saved to {log_loc}")
    logger.info("=================================")

    # remove all handlers to avoid multiple logging
    logger.handlers = []

    return data


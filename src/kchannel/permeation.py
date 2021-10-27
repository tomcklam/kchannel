import re
import os
import numpy as np
import scipy.stats
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class Channel:
    def __init__(self):
        self.occupancy_4_all = []
        self.occupancy_6_all = []
        self.jumps_all = []
        self.dts = None
        self.dt = None
        self.currents = None
        self.current = None
        self.total_times = None

def permeationEventsPartition(occupancy, jump, seedState, n_bs_jump):
    """ partitioning trajectory into permeation events

    Parameters
    ----------
    occupancy: array of size N
        tranjectory expressed in the form of SF occupancy

    jumps: arrays of size (N-1, 2)
        net jumps for ion and water for all trajectories

    seedState: string
        the SF occupation state that the cycles start and end in

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    Returns
    -------
    stationaryPhase_indices: list of tuples
        index for the start and the end of stationary phases
    permeationCycle_indices: array of int
        index for the start and the end of permeation cycles, which are
        stationary phase (accumulated jump remains as  int((n_bs_jump+1)*i+offset)
        + conduction phase (accumulated jump increases from
        int((n_bs_jump+1)*i+offset to int((n_bs_jump+1)*(i+1)+offset)
    """
    k_jump = jump[:, 0]
    w_jump = jump[:, 1]

    n_k_netjumps = np.sum(jump[:, 0]) // (n_bs_jump+1)

    # prepend the cumsum with a "0" to align it with occ,
    # now k_netjump_cum[i] is defined as the accumulated
    # k net jump before t=i
    k_netjump_cum = np.zeros(len(k_jump)+1, dtype=int)
    k_netjump_cum[1:] = np.cumsum(k_jump)

    # ignore the last state as no jump info is available
    # and find "seed" with no k and w jump
    indices = np.argwhere(occupancy[:-1] == seedState).reshape(-1)
    # 2. finding "seed" with no k and w jump
    indices = indices[~np.any(jump[indices], axis=1)]
    seed_idx = indices[0]
    offset = k_netjump_cum[seed_idx]

    stationaryPhase_indices = []
    for i in range(n_k_netjumps):
        indices_i = indices[k_netjump_cum[indices] == int((n_bs_jump+1)*i+offset)]
        if len(indices_i) > 0:
            start_i, end_i = indices_i[0], indices_i[-1]
            stationaryPhase_indices.append((start_i, end_i))
        else:
            stationaryPhase_indices.append(())

    # permeation cycle = stationary phase + conduction phase
    #                  = start of i-th SP to start of (i+1)-th SP
    permeationCycle_indices = []
    for i in range(len(stationaryPhase_indices)-1):
        try:
            start_i = stationaryPhase_indices[i][0]
            end_i = stationaryPhase_indices[i+1][0]
            permeationCycle_indices.append([start_i, end_i])
        except:
            print(f"{i}-th cycle between is discarded as {seedState} is not found")
            continue

    print(f"Total permeation events: {n_k_netjumps}")
    print(f"Identified cycles: {len(permeationCycle_indices)}")
    permeationCycle_indices = np.array(permeationCycle_indices)

    return stationaryPhase_indices, permeationCycle_indices

def findCycles(occupancy_all, jumps_all, seedState, n_bs_jump=4):
    """ Given occupancy and jumps of the trajectories, the seed state, and n_bs_jump
    that define the # BSs in which jumps in and out are considered,
    give the cycles that start and end in the seed state

    Parameters
    ----------
    occupancy_all: list of arrays of size N
        occupancy for all trajectories

    jumps_all: list of arrays of size (N-1, 2)
        net jumps for ion and water for all trajectories

    seedState: string
        the SF occupation state that the cycles start and end in

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    Returns
    -------
    permeationCycle_indices: list of lists of arrays
        contain the cycles identified in each trajectory
    """
    permeationCycles = []
    p_indices_all = []

    for occupancy, jumps in zip(occupancy_all, jumps_all):
        _ , p_indices = permeationEventsPartition(occupancy, jumps, seedState, n_bs_jump=n_bs_jump)
        p_indices_all.append(p_indices)

        permeationCycles_ = [cycleCompression(occupancy[i:j+1],
                                              jumps[i:j+1],
                                              n_bs_jump=n_bs_jump)
                             for (i, j) in p_indices]
        permeationCycles.append(permeationCycles_)

    return permeationCycles, p_indices_all

def cycleCompression(occupancy_cycle, k_jumps_sub, n_bs_jump):
    """ Given an uncompressed (involving osciliation between states without net jumps)
        trajectory segment that starts and ends in the same state and records one
        complete permeation event + the associated jump vectors, compute the "cleaned"
        cycle keeping only the first hit states

    Parameters
    ----------
    occupancy_cycle: array
        occupancy for a trajectory segment

    k_jumps_sub: array
        net jumps for ion in the associated occupancy_cycle

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    Returns
    -------
    occupancy_compressed: array
        "cleaned" cycle keeping only the first hit states
    """
    _, idx = np.unique(occupancy_cycle, return_index=True)
    unique = occupancy_cycle[np.sort(idx)]

    for state in unique:
        keep = np.ones(len(occupancy_cycle), dtype=bool)
        indices = np.argwhere(occupancy_cycle == state).reshape(-1)

        for i in range(len(indices)-1):
            start_i = indices[i]
            end_i = indices[i+1]
            k_netjumps_i = k_jumps_sub[start_i:end_i]

            # discard if the state repeats itself without net ion jump
            if (end_i - start_i) > 0 and np.sum(k_netjumps_i) == 0:
                keep[start_i:end_i] = False

        occupancy_cycle = occupancy_cycle[keep]
        k_jumps_sub = k_jumps_sub[keep]

    occupancy_compressed = occupancy_cycle
    k_jumps_compressed = k_jumps_sub

    if np.sum(k_jumps_compressed) != (n_bs_jump+1):
        print(f"netjump != {(n_bs_jump+1)}")
    return occupancy_compressed

def hittingTimes(occupancy, jumps, intStates, finalStates, n_bs_jump=4, backward=False):
    """ compute hitting time for transition pairs within one permeation event,
        i.e. abs(k_netjumps) < n_bs_jump+1

    Parameters
    ----------
    occupancy: array of size N
        tranjectory expressed in the form of SF occupancy

    jumps: arrays of size (N-1, 2)
        net jumps for ion and water for all trajectories

    intStates: list of strings
        the SF occupation state that the transitions start in

    finalStates: list of strings
        the SF occupation state that the transitions end in

    n_bs_jump: int
        # binding sites considered in permeation
        It is used to compute number of k jumps (n_bs_jump+1) one complete
        permeation event takes

    backward: boolean, False by default
        whether the hitting times corresponding to transitions in which the ion movement is
        against the gradient. If True, then only transitions with +ve k_netjump are taken into account

    Returns
    -------
    hittingTimes: list of int

    k_netjumps_count: list of int
        # net k jumps involved in the transitions

    w_netjumps_count: list of int
        # net water jumps involved in the transitions
    """

    #intStates should contain 1 state only
    hittingTimes = []

    # count number of k and w jumps happening during the period for which hitting times are computed
    k_netjumps_counts = []
    w_netjumps_counts = []

    k_netjumps = jumps[:, 0]
    w_netjumps = jumps[:, 1]

    waiting = False
    hittingTime = 0

    # count for each initial state to avoid overlooking some of the initial states in scenario like
    #    initialState_1 -> xxx -> initialState_2 -> xxx -> finalState_2
    # where initialState_2 won't be counted if otherwise
    for intState in intStates:
        for i, s in enumerate(occupancy):
            if s in intState and waiting is False:
                waiting = True
                start_idx = i
            elif s in finalStates and waiting is True:
                end_idx = i
                if jumps is None:
                    k_netjump = 0
                else:
                    k_netjump = int(np.sum(k_netjumps[start_idx:end_idx]))
                    w_netjump = int(np.sum(w_netjumps[start_idx:end_idx]))

                # restrict the scope to transitions within one permeation event,
                # i.e. 0 <= abs(k_netjump) < (n_bs_jump+1)
                if (k_netjump >= 0 and k_netjump < (n_bs_jump+1) and backward is False) or \
                    (k_netjump <= 0 and k_netjump > -(n_bs_jump+1) and backward is True):

                    hittingTime = end_idx - start_idx
                    hittingTimes.append(hittingTime)
                    k_netjumps_counts.append(k_netjump)
                    w_netjumps_counts.append(w_netjump)

                waiting = False
                start_idx = None
                end_idx = None

    return hittingTimes, k_netjumps_counts, w_netjumps_counts

def permeationTimes(occupancy_all, jumps_all, pairs, n_bs_jump=4, dt=20/1000):
    data = []
    for initialStates, finalStates in pairs:
        hTs_all = []
        k_j_counts_all = []
        w_j_counts_all = []
        inititalStates_label = ','.join(initialStates)
        finalStates_label = ','.join(finalStates)

        for occupancy, jumps in zip(occupancy_all, jumps_all):
            hTs, k_j_counts, w_j_counts = hittingTimes(occupancy, jumps, initialStates, finalStates,
                                                                  n_bs_jump=4, backward=False)
            hTs_all += hTs
            k_j_counts_all += k_j_counts
            w_j_counts_all += w_j_counts
        hTs_all = np.asarray(hTs_all) * dt
        n_hTs = len(hTs_all)

        hTs_all_mean = np.mean(hTs_all)
        hTs_all_bs = scipy.stats.bootstrap((hTs_all, ), np.mean, confidence_level=.95,
                                           n_resamples=10000, method='BCa')
        hTs_all_bs_l, hTs_all_bs_u = hTs_all_bs.confidence_interval

        k_j_counts_mean = np.mean(k_j_counts)
        w_j_counts_mean = np.mean(w_j_counts)

        row = [inititalStates_label, finalStates_label, hTs_all_mean,
               hTs_all_bs_l, hTs_all_bs_u, n_hTs, k_j_counts_mean, w_j_counts_mean]

        data.append(row)

    df = pd.DataFrame(data,
                      columns=["initial","final", "mean (ns)", "low (ns)",
                               "high (ns)", "n", "k_f", "w_f"])
    return df

def plotCycles(cycles, threshold=0.05, dist=0.2, figsize=(10,10), save=None,
               returnCycleProb=False, returnMainPath=False, returnFullPaths=False):
    """
    Parameters
    ----------
    cycles: array
        occupancy for a trajectory segment

    Returns
    -------

    """
    # flatten nested cycles
    cycles_flattened = [c for cycle in cycles for c in cycle]
    n_cycles = len(cycles_flattened)

    # probs: dict, key=state, value=(target state, %, count)
    # counts: dict, key=state, value=(count, population)
    probs, counts = markov(cycles_flattened, embedded=False, quiet=True)

    probs_individual = []
    counts_individual = []

    for cycle in cycles:
        p, c = markov(cycle, embedded=False, quiet=True)
        probs_individual.append(p)
        counts_individual.append(c)

    # find the backbone of the cyclic graph
    states_all = np.array([k for k in probs.keys()])
    backbone = []

    # assume that probs are already sorted in descending order of population
    state = list(probs)[0]
    backbone.append(state)

    state = list(probs[state].keys())[0]
    backbone.append(state)

    while(backbone[-1] != list(probs)[0]):
        state = list(probs[state].keys())[0]
        backbone.append(state)

    # assign the side branches of the cyclic graph
    non_backbone = states_all[np.in1d(states_all, backbone, invert=True)]
    sidechain = {k:[] for k in backbone}

    #     for state_1 in non_backbone:
    #         scores = [computeSimilarity(state_1, state_2) for state_2 in backbone]
    #         backbone_state = backbone[np.argmax(scores)]
    #         sidechain[backbone_state].append(state_1)

    # put next to the backbone state before the target state which the non-backbone
    # state has highest probability transitioning to
    for state_i in non_backbone:
        j_idx = None
        for state_j in list(probs[state_i]):
            try:
                j_idx = backbone.index(state_j)
                break
            except:
                continue
        sidechain[backbone[j_idx-1]].append(state_i)


    # determine position of nodes
    pos = nx.circular_layout(nx.path_graph(backbone))
    for sel in sidechain.keys():
        distance = np.linalg.norm(np.asarray([v for v in pos.values()]) - pos[sel], axis=1)
        coor = list(pos.values())[np.argmax(distance)]

        c = (1.4 + dist * np.arange(len(sidechain[sel]))).reshape(-1, 1)
        newpos = coor + c * (pos[sel] - coor)

        for k, v in zip(sidechain[sel], newpos):
            pos[k] = v
    # plot graph
    _ = plt.figure(1,figsize=figsize, dpi=200)
    _ = plt.axis('equal')
    G = nx.DiGraph()
    G.add_nodes_from(states_all)
    sizes = np.array([counts[state][0] for state in states_all])


    nx.draw_networkx(G, pos, node_color='orange',
                     node_size=1000*sizes/n_cycles,
                     alpha=1, font_size=12, font_color='k')


    p_dict = {}
    for state_i in states_all:
        p_dict_ = {}
        for state_j in states_all:
            if state_i != state_j:
                _ , _, count = probs[state_i][state_j]
                p = count / n_cycles
                err = 1.96 * np.sqrt(p*(1-p)/ (n_cycles-1))
                alpha = np.tanh(2*count/n_cycles)/np.tanh(2)
                _ = nx.draw_networkx_edges(G, pos,
                                           edgelist=[(state_i, state_j, 2)],
                                           arrowsize=16,
                                           min_source_margin=30, min_target_margin=30,
                                           alpha=alpha,
                                           connectionstyle="arc3,rad=0.2")
                # create edge label
                if p > threshold:
                    label = fr"{p*100:.1f}$\pm${err*100:.1f}%"
                      # make labels lie on outside
#                     edge = pos[state_j] - pos[state_i]
#                     normal = np.array([-edge[1]/edge[0], 1])
#                     normal /= np.linalg.norm(normal)
#                     direction_vec = np.cross(np.array([0, 0, -1]), edge)
#                     direction = np.sign(np.dot(normal, direction_vec[:-1]))
#                     move = direction * normal * np.linalg.norm( .5 * (pos[state_j] - pos[state_i]) ) * np.sin(np.pi/10)
#                     pos_tmp = {state_i:(pos[state_i]+move), state_j:(pos[state_j]+move)}
                    _ = nx.draw_networkx_edge_labels(G, pos, edge_labels={(state_i, state_j):label} ,
                                                     label_pos=0.5, font_size=8,
                                                     font_color='k', font_family='sans-serif',
                                                     font_weight='normal', alpha=alpha, bbox=dict(alpha=0),
                                                     horizontalalignment='center', verticalalignment='center',
                                                     ax=None, rotate=True)
                p_dict_[state_j] = (p, err)
        p_dict[state_i] = p_dict_

    _ = plt.tight_layout()
    if save is not None:
        _ = plt.savefig(saveName, dpi=1000)
        print(f"saved as {saveName}")
    _ = plt.show()

    if returnMainPath:
        transition_pairs = [[k]for k in sidechain.keys()]
        transition_pairs_tmp = [[transition_pairs[i], transition_pairs[i+1]] for i in range(len(transition_pairs)-1)]
        transition_pairs = transition_pairs_tmp + [[transition_pairs[-1], transition_pairs[0]]]
        if returnCycleProb:
            return p_dict, transition_pairs
        else:
            return transition_pairs

    if returnFullPaths:
        transition_pairs = [[k]+sidechain[k] for k in sidechain.keys()]
        transition_pairs_tmp = [[transition_pairs[i], transition_pairs[i+1]] for i in range(len(transition_pairs)-1)]
        transition_pairs = transition_pairs_tmp + [[transition_pairs[-1], transition_pairs[0]]]
        if returnCycleProb:
            return p_dict, transition_pairs
        else:
            return transition_pairs

def loadResults(results_loc):
    jumps_all = []
    occupancy_4_all = []
    occupancy_6_all = []
    total_times = []
    dts = []
    currents = []


    for loc in results_loc:
        data_loc = os.path.join(loc, 'results.csv')
        df = pd.read_csv(data_loc, index_col=0)
        occ = df['occupancy'].to_numpy().astype(str)
        jumps = df[['j_k', 'j_w']].to_numpy().astype(int)

        occupancy_6_all.append(occ)
        occ_4 = np.array([s[1:-1] for s in occ])
        occupancy_4_all.append(occ_4)
        jumps_all.append(jumps)

        log_loc = os.path.join(loc, 'results.log')
        with open(log_loc, 'r') as f:
            log = f.read()
        total_time = float(re.search(r'Total time\D+(\d+\.\d+)', log).group(1))
        dt = float(re.search(r'dt\D+(\d+\.\d+)', log).group(1))
        current = float(re.search(r'Current\D+(\d+\.\d+)', log).group(1))
        total_times.append(total_time)
        dts.append(dt)
        currents.append(current)
    channel = Channel()
    channel.occupancy_4_all = occupancy_4_all
    channel.occupancy_6_all = occupancy_6_all
    channel.jumps_all = jumps_all
    channel.total_times = np.array(total_times)
    channel.dts = np.array(dts)
    channel.currents = np.array(currents)

    return channel

def computeStats(channel):
    current_bs = scipy.stats.bootstrap((channel.currents,), np.mean, confidence_level=.95, n_resamples=10000, method='BCa')
    current_bs_l, current_bs_h = current_bs.confidence_interval
    channel.current = (np.mean(channel.currents), current_bs_l, current_bs_h)
    print(f"Current (pA): {channel.current:.3f}\t{current_bs_l:.3f} - {current_bs_h:.3f}\n")

    states, counts = np.unique(np.concatenate(channel.occupancy_6_all), return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    states = states[sort_idx]
    population = counts[sort_idx] / np.sum(counts)

    stats_dict = {'T (ns)':channel.total_times,
                  'dt (ns)':channel.dts,
                  'current (pA)':channel.currents}

    states_dict = {}
    states_dict['state'] = states
    states_dict['p_mean'] = population

    p_ls = []
    p_hs = []

    for s, p_mean in zip(states, population):
        ps = np.array([np.mean(occupancy == s) for occupancy in channel.occupancy_6_all])
        stats_dict[s] = ps

        p_bs = scipy.stats.bootstrap((ps,), np.mean, confidence_level=.95, n_resamples=10000, method='BCa')
        p_l, p_h = p_bs.confidence_interval
        p_ls.append(p_l)
        p_hs.append(p_h)

    states_dict['p_l'] = p_ls
    states_dict['p_h'] = p_hs

    stats = pd.DataFrame(stats_dict)
    states = pd.DataFrame(states_dict)
    return stats, states

def markov(trajs, threshold=1, embedded=False, return_matrix=False, quiet=False):
    if isinstance(trajs, np.ndarray):
        trajs = [trajs]
    #assert isinstance(trajs, list)
    #assert isinstance(trajs[0], np.ndarray)
    trans_prob_dict = {}
    n_outward_total_dict = {}

    if isinstance(trajs, np.ndarray):
        trajs = [trajs]
    if isinstance(trajs, list) and isinstance(trajs[0], list):
        trajs = [s for traj in trajs for s in traj]

    states, counts = np.unique(np.concatenate(trajs), return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    states = states[sort_idx]
    counts = counts[sort_idx]
    counts_total = np.sum(counts)

    population = {states[i]:counts[i] / counts_total for i in range(len(counts))}
    state_to_idx = {s:i for i, s in enumerate(states)}
    idx_to_state = {i:s for i, s in enumerate(states)}

    n_states = len(states)
    trans_counts = np.zeros((n_states, n_states), dtype=int)

    for traj in trajs:
        for t in range(len(traj)-1):
            i, j = state_to_idx[traj[t]], state_to_idx[traj[t+1]]
            trans_counts[i,j] += 1
    trans_counts[trans_counts < threshold] = 0

    trans_prob = np.nan_to_num(np.asarray([trans_counts[i] / (np.sum(trans_counts[i]) or 1.0) for i in range(n_states)]))

    if embedded:
        _ = np.fill_diagonal(trans_prob, 0)
        trans_prob_embedded = np.nan_to_num(np.asarray([trans_prob[i] / (np.sum(trans_counts[i]) or 1.0) for i in range(n_states)]))


    n_outward_total = np.sum(trans_counts, axis=1)
    n_outward_total_sum = np.sum(n_outward_total)

    for i, n in enumerate(n_outward_total):
        n_outward_total_dict[idx_to_state[i]] = (n, n/n_outward_total_sum)

    for i in range(n_states):
        target = {}
        idx = np.argsort(trans_prob[i])[::-1]
        if quiet is False:
            print(f"\n========= {idx_to_state[i]} {n_outward_total[i]}===========")

        for j in idx:
            targetState = idx_to_state[j]
            prob = trans_prob[i,j]
            n_outward = trans_counts[i,j]
            ci = 2 * np.sqrt(prob*(1-prob)/(n_outward_total[i] or 1e-10)) # or n_outward?

            if j<12 and quiet is False:
                print(targetState,
                      f"\t\t{prob*100:.5f} +- {ci*100:.2f} %",
                      f"\t{n_outward}")

            #target.append((targetState, prob, ci, n_outward))
            target[targetState] = (prob, ci, n_outward)
        trans_prob_dict[idx_to_state[i]] = target
    if return_matrix:
        return trans_prob, state_to_idx, idx_to_state
    else:
        return trans_prob_dict, n_outward_total_dict



import numpy as np

def computeTransProb(trajs, jumps=None, return_matrix=False, quiet=False):
    """ given trajectories, compute transition probabilities

    Parameters
    ----------
    trajs: list of arrays of size N    or   list of lists of arrays
        trajectories, can be occupancy or cycles for all trajectories

    return_matrix: boolean
        if True, return also the transition matrix and mapping between indices and state names

    quiet: boolean
        if True, do not print anything to stdout

    Returns
    -------
    trans_prob_dict: dict
        dict containing count and prob of states, and the transition probs

    trans_prob: narray (optional)
        transition matrix

    state_to_idx: dict
        map from state name to idx

    idx_to_state: dict
        map from idx to state name (reverse of state_to_idx)

    """
    # for list of lists of arrays
    if isinstance(trajs, list) and isinstance(trajs[0], list):
        trajs = [s for traj in trajs for s in traj]

    states, counts = np.unique(np.concatenate(trajs), return_counts=True)
    sort_idx = np.argsort(counts)[::-1]
    states = states[sort_idx]
    counts = counts[sort_idx]
    counts_total = np.sum(counts)

#     trans_prob_dict = {state:{"count":count, "prob":count/counts_total} \
#                        for state, count in zip(states, counts)}
    trans_prob_dict = {state:{} for state in states}

    state_to_idx = {s:i for i, s in enumerate(states)}
    idx_to_state = {i:s for i, s in enumerate(states)}

    n_states = len(states)
    trans_counts = np.zeros((n_states, n_states), dtype=int)
    if jumps:
        # total number of net jumps for each transition
        k_jumps_counts = np.zeros((n_states, n_states), dtype=int)
        w_jumps_counts = np.zeros((n_states, n_states), dtype=int)


    if jumps:
        for traj, jump in zip(trajs, jumps):
            for t in range(len(traj)-1):
                i, j = state_to_idx[traj[t]], state_to_idx[traj[t+1]]
                trans_counts[i,j] += 1
                k_jumps_counts[i,j] += jump[t, 0]
                w_jumps_counts[i,j] += jump[t, 1]
    else:
        for traj in trajs:
            for t in range(len(traj)-1):
                i, j = state_to_idx[traj[t]], state_to_idx[traj[t+1]]
                trans_counts[i,j] += 1

    # trans_counts[trans_counts < threshold] = 0
    trans_prob = np.nan_to_num(np.asarray([trans_counts[i] / (np.sum(trans_counts[i]) or 1.0) for i in range(n_states)]))
    n_outward_total = np.sum(trans_counts, axis=1)

    if jumps:
        k_jump_avg = np.nan_to_num(k_jumps_counts / trans_counts)
        w_jump_avg = np.nan_to_num(w_jumps_counts / trans_counts)

    for i in range(n_states):
        state_i = idx_to_state[i]
        n_total = n_outward_total[i]

        indices = np.argsort(trans_prob[i])[::-1]
        for j in indices:
            state_j = idx_to_state[j]
            p = trans_prob[i, j]
            if n_total < 2:
                err = .0
            else:
                err = 1.96 * np.sqrt(p*(1-p)/(n_total-1))
            n = trans_counts[i, j]
            if jumps:
                trans_prob_dict[state_i][state_j] = (trans_prob[i, j], err, n, k_jump_avg[i,j], w_jump_avg[i,j])
            else:
                trans_prob_dict[state_i][state_j] = (trans_prob[i, j], err, n)

        trans_prob_dict[state_i]['count_out'] = n_total
        trans_prob_dict[state_i]['count'] = counts[i]
        trans_prob_dict[state_i]['prob'] = counts[i]/counts_total

    if not quiet:
        for state_i, state_i_dict in trans_prob_dict.items():
            print(f"\n========= {state_i} {state_i_dict['count_out']}===========")
            for k, v in state_i_dict.items():
                if k not in ['count', 'prob', 'count_out']:
                    state_j = k
                    if jumps:
                        p, err, n, k_j, w_j = v
                        if p != .0:
                            print(f"{state_j:<10}\t{p:<12.5%} ± {err:<12.5%} {n} {k_j} {w_j}")
                    else:
                        p, err, n = v
                        if p != .0:
                            print(f"{state_j:<10}\t{p:<12.5%} ± {err:<12.5%} {n}")

    if return_matrix:
        return trans_prob_dict, trans_prob, state_to_idx, idx_to_state
    else:
        return trans_prob_dict

def MCMC(prob_dict, s0, length=10000):

    n_states = len(prob_dict)
    state_to_idx = {s:i for i, s in enumerate(prob_dict.keys())}
    idx_to_state = {i:s for i, s in enumerate(prob_dict.keys())}

    trans_prob = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            trans_prob[i,j] = float(prob_dict[idx_to_state[i]][idx_to_state[j]][0])

    trans_prob_cumsum = np.cumsum(trans_prob, axis=1)

    mcmc_traj = np.zeros(length, dtype=int)
    mcmc_traj[0] = state_to_idx[s0]
    for i in range(1, length):
        mcmc_traj[i] = np.searchsorted(trans_prob_cumsum[mcmc_traj[i-1]], np.random.random())

    mcmc_traj = np.array([idx_to_state[i] for i in mcmc_traj])

    if len(prob_dict[mcmc_traj[0]][mcmc_traj[1]]) > 3:
        mcmc_jump = np.zeros((length-1, 2))
        for i in range(1, length-1):
            mcmc_jump[i, 0], mcmc_jump[i, 1] = prob_dict[mcmc_traj[i-1]][mcmc_traj[i]][-2:]
        return mcmc_traj, mcmc_jump
    else:
        return mcmc_traj

import pytest
import kchannel

def test_SF_MthK():

    sf_MthK_ref = {'O': {4: {'idx': [756, 2209, 3662, 5115],
                'resid': [59, 59, 59, 59],
                'resname': ['THR', 'THR', 'THR', 'THR'],
                'name': ['OG1', 'OG1', 'OG1', 'OG1']},
                3: {'idx': [759, 2212, 3665, 5118],
                'resid': [59, 59, 59, 59],
                'resname': ['THR', 'THR', 'THR', 'THR'],
                'name': ['O', 'O', 'O', 'O']},
                2: {'idx': [779, 2232, 3685, 5138],
                'resid': [60, 60, 60, 60],
                'resname': ['VAL', 'VAL', 'VAL', 'VAL'],
                'name': ['O', 'O', 'O', 'O']},
                1: {'idx': [786, 2239, 3692, 5145],
                'resid': [61, 61, 61, 61],
                'resname': ['GLY', 'GLY', 'GLY', 'GLY'],
                'name': ['O', 'O', 'O', 'O']},
                0: {'idx': [807, 2260, 3713, 5166],
                'resid': [62, 62, 62, 62],
                'resname': ['TYR', 'TYR', 'TYR', 'TYR'],
                'name': ['O', 'O', 'O', 'O']}},
                'CA': {4: {'idx': [746, 2199, 3652, 5105],
                'resid': [59, 59, 59, 59],
                'resname': ['THR', 'THR', 'THR', 'THR'],
                'name': ['CA', 'CA', 'CA', 'CA']},
                3: {'idx': [762, 2215, 3668, 5121],
                'resid': [60, 60, 60, 60],
                'resname': ['VAL', 'VAL', 'VAL', 'VAL'],
                'name': ['CA', 'CA', 'CA', 'CA']},
                2: {'idx': [782, 2235, 3688, 5141],
                'resid': [61, 61, 61, 61],
                'resname': ['GLY', 'GLY', 'GLY', 'GLY'],
                'name': ['CA', 'CA', 'CA', 'CA']},
                1: {'idx': [789, 2242, 3695, 5148],
                'resid': [62, 62, 62, 62],
                'resname': ['TYR', 'TYR', 'TYR', 'TYR'],
                'name': ['CA', 'CA', 'CA', 'CA']},
                0: {'idx': [810, 2263, 3716, 5169],
                'resid': [63, 63, 63, 63],
                'resname': ['GLY', 'GLY', 'GLY', 'GLY'],
                'name': ['CA', 'CA', 'CA', 'CA']}}}

    sf_MthK = kchannel.detectSF("MthK/start.gro")

    assert sf_MthK == sf_MthK_ref

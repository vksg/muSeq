'''
Created on May 10, 2015

@author: levy
'''

import sys

MY_PYTHON_LIBS = ['/data/safe/vsreeniv/lib/python2.7/site-packages',
            '/data/safe/vsreeniv/python-cpp']
if sys.path[:2] != MY_PYTHON_LIBS:
    sys.path =  MY_PYTHON_LIBS + sys.path


import numpy as np
import tables
import time
import cPickle
import pysam
from representation import *
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from itertools import izip
import networkx as nx
import GCExt


DEFAULT_COLOR_LIST = ["#554236",
                      "gray", 
                      "#F77825",
                      "#D3CE3D",
                      "#F1EFA5",
                      "#60B99A",
                      "#BBB2DC"]

#colorlist = [
#"gray", 
#"#554236",
#"#F77825",
#"#D3CE3D",
#"#F1EFA5",
#"#60B99A"]

NUCS        = [' ', 'A', 'C', 'G', 'T', 'N']
NUCS_RL     = {' ':0, 'A':1, 'C':2, 'G':3, 'T':4, 'N':5}
MAX_PATTERN_NOISE = 0.05

## move this to C
def get_hamming_and_overlap(bbp, em):
    """
    Compute the hamming distance over bbp (binary bit patterns)
    over positions that are covered in both (error mask)
    returning the hamming difference and the amount of overlap
    """
    N = len(bbp)
    overlap  = np.zeros((N, N), dtype="int16")
    mismatch = np.zeros((N, N), dtype="int16")
    for i in xrange(N):
        for j in xrange(i+1, N):
            common_mask = em[i]*em[j]
            x = np.sum(common_mask)
            y = np.sum(common_mask*np.logical_xor(bbp[i], bbp[j]))
            overlap[i,j] = overlap[j,i] = x
            mismatch[i,j] = mismatch[j, i] = y
    return mismatch, overlap

def bit_count(int_type):
    return bin(int_type).count("1")

def to_binary(X):
    return int( (len(X) * '%d') % tuple(X),2)

def load_restriction_fragments_of_interest(interval_bedfile):
    """
    loads the theoretically predicted to be well- 
    mapped and smallish fragments (150-400 bp)
    """
    interval_file = file(interval_bedfile)
    ans = []
    for line in interval_file:
        chrom, start, end = line.strip().split("\t")
        ans.append((chrom, int(start), int(end)))
    
    ## data type declaration
    DT = {'names':['chrom', 'start', 'end'],
          'formats':['S5', 'int', 'int']}
    interval_file.close()
    ans = np.array(ans, dtype=DT)
    return ans


def extract_reads_for_fragment(bamreader, chrom, start, end, buff_in=2):
    """
    Pull all the reads from a bam file over the corresponding interval.
    Identify read pairs that correspond to a restriction fragment,
    within the tolerances of buff_in.
    
    Returns the reads split by original top and original bottom.
    """
    
    pair_dict = defaultdict(lambda: [None, None])
    
    for read in bamreader.fetch(chrom, start, end):
            read_name = read.qname
            if PROJECT == 'next_seq':
                read_name += '.'+read.get_tag('RG')
            pair_dict[read_name][read.is_read2] = read
    
    ## two read lists, for OB and for OT
    read_lists = [[], []]
    
    for x in pair_dict.keys():
        r1, r2 = pair_dict[x]
        ## if both reads not in the interval
        if r1 == None or r2 == None:
            continue    
        ## or not a proper pair, then skip this readpair
        if not (r1.is_reverse ^ r2.is_reverse):
            continue
        ## determine if from the original top strand
        original_top = r2.is_reverse
        ## pair start and end coordinates depend on strand
        if original_top:
            pstart, pend = r1.reference_start, r2.reference_end
        else:
            pstart, pend = r2.reference_start, r1.reference_end
        ## test if within the boundaries
        start_hit = start         <= pstart < start + buff_in
        end_hit   = end - buff_in < pend    <= end
        ## and if not in the boundaries
        if not (start_hit and end_hit):
            continue
        ## then add to the strand-based read list
        read_lists[original_top].append((r1, r2))
    
    return read_lists

## perhaps improve with read.get_blocks()
def read_list_to_referenced_arrays(read_list, start, end, qual_offset=33):
    '''
    returns two arrays of arrays:
    containing the sequence and the quality scores for read1 and read2
    converted into integer arrays according to the nucleotide dictionary.
    stored in uint8 for size.
    '''
    
    ## N is the number of read pairs
    ## L is the length of the interval
    N = len(read_list)
    L = end - start
    
    seq  = [np.zeros(shape=(N, L), dtype="uint8"), 
            np.zeros(shape=(N, L), dtype="uint8")]
    qual = [np.zeros(shape=(N, L), dtype="uint8"), 
            np.zeros(shape=(N, L), dtype="uint8")]
    
    ## for each read pair
    for j, read_pair in enumerate(read_list):
        ## for read1 and read2
        for i in range(2):
            read  = read_pair[i]
            rseq  = read.seq
            rqual = read.qual
            ## go through all the reference positions       
            for rpos, gpos in read.get_aligned_pairs():
                if gpos != None and rpos != None:
                    seq[i][j, gpos-start]  = NUCS_RL[rseq[rpos]]
                    qual[i][j, gpos-start] = ord(rqual[rpos]) - qual_offset
    
    return seq, qual


def get_all_referenced_arrays(read_lists, start, end, conservative=True):
    """
    Convert both the OB and OT read lists into referenced arrays.
    Returns seq, qual and joint_seq as lists of lists.

    The first index specifies OB or OT (0,1)
    and the second index specifies read1 or read2 (0, 1)
    Joint_seq is indexed just by OB or OT
    
    If conservative = True, then bases where read1 and read2 disagree are labeled "N"
    If not, vote from the best quality base preferring read1 over read2 in case of a tie. 
    """
    seqqual = [read_list_to_referenced_arrays(read_lists[i], start, end) for i in [0,1]]
    seq  = [seqqual[i][0] for i in [0,1]]
    qual = [seqqual[i][1] for i in [0,1]]
    # pick the best quality base (prefering read1 over read2 at a tie)
    joint_seq = [np.where(qual[i][0] >= qual[i][1], seq[i][0], seq[i][1]) for i in [0,1]]
    if conservative:
        #hyper-correct, no disagreements
        joint_seq = [np.where((qual[i][0]*qual[i][1] > 0)*(seq[i][0] != seq[i][1]), NUCS_RL["N"], joint_seq[i]) for i in [0,1]]
    return seq, qual, joint_seq

def get_base_ratios(joint_seq):
    """
    from the pair read consensus, compute the base ratio over each position
    for the OB and OT.
    """
    L = joint_seq[0].shape[1]
    base_counts = [np.zeros(shape=(len(NUCS), L)),
                   np.zeros(shape=(len(NUCS), L))]
    for i in range(2):
        for j in range(len(NUCS)):
            base_counts[i][j] = np.sum(joint_seq[i] == j, axis=0)
    return [bc / np.sum(bc, axis=0) for bc in base_counts]


def get_bit_masks(base_ratios, homo_thresh = 0.9):
    """
    Determine which bits are homozygous "C" or "G" based on empirical base count ratios
    for the top and bottom strand
    """
    bottom_bit_OB = base_ratios[0][NUCS_RL['A']] + base_ratios[0][NUCS_RL['G']]
    bottom_bit_OT = base_ratios[1][NUCS_RL['G']]
    
    top_bit_OB = base_ratios[0][NUCS_RL['C']]
    top_bit_OT = base_ratios[1][NUCS_RL['C']] + base_ratios[1][NUCS_RL['T']]
    
    bottom_bit_mask = (bottom_bit_OB > homo_thresh)*(bottom_bit_OT > homo_thresh)
    top_bit_mask    = (top_bit_OB > homo_thresh)*(top_bit_OT > homo_thresh)
    return bottom_bit_mask, top_bit_mask


def bits_to_patterns_and_error(bit_patterns):
    """
    Convert bit_patterns (masked joint_seq by bit_masks)
    to binary_bit_pattern: zero/one for unflipped / flipped matrix
    and a error_mask: zero if nucleotide is neither flipped nor unflipped bit.    
    """
    unflipped_bit = map(NUCS_RL.get, ["G", "C"])
    flipped_bit = map(NUCS_RL.get, ["A", "T"])
    binary_bit_pattern = [(bp == fbit) for bp, ubit, fbit, in izip(bit_patterns, unflipped_bit, flipped_bit)]
    error_mask         = [(bp == fbit)+(bp==ubit) for bp, ubit, fbit, in izip(bit_patterns, unflipped_bit, flipped_bit)]
    return binary_bit_pattern, error_mask


def plot_base_data(ax, chrom, start, end, seq_dic, base_calls, color_list=None, mask=None):
    '''
    plot the base call data over the interval chrom:start-end.
    the color_list specifies "reference match", "no info", A, C, T, G
    '''
    if color_list == None:
        color_list = DEFAULT_COLOR_LIST
    cmap = mpl.colors.ListedColormap(color_list)
    ref_seq       = seq_dic[chrom][start:end]
    ref_seq_array = np.array([NUCS_RL[x] for x in ref_seq])
    
    
    plot_seq = np.array(base_calls, dtype=int)
    plot_seq[base_calls == ref_seq_array] = -1
    if type(mask) != type(None):
        plot_seq = plot_seq[:, mask]
        
    ax.imshow(plot_seq, interpolation="nearest", aspect='auto', vmin=-1, vmax=5, cmap=cmap)


def get_likelihood_difference(flip_rate, error_rate, mismatch, overlap, unique_counts):    
    ## expected hamming difference rates
    flip_hamming  = 2*flip_rate *(1 - flip_rate)
    error_hamming = 2*error_rate*(1 - error_rate)
    
    ## if you want to be very precise, (and we do) should include error in hamming rate
    flip_hamming = error_hamming*(1 - flip_hamming) + flip_hamming*(1-error_hamming) 
    
    logpe   = np.log(error_hamming)
    log1mpe = np.log(1.0 - error_hamming)
    logpf   = np.log(flip_hamming)
    log1mpf = np.log(1.0 - flip_hamming)
    
    ## ds is the difference in log likelihoods 
    ## that i and j are log(same cluster) - log(different cluster)    
    
    d1 = logpe*mismatch + (overlap-mismatch)*log1mpe
    d2 = logpf*mismatch + (overlap-mismatch)*log1mpf
    mult = np.outer(unique_counts, unique_counts)
    ds = (mult*(d1 - d2)).astype("float32")
    return ds


# move edge fraction into GCExt:
# 1. to remove calculation of blue_edges and red_edges from python loop
# 2. to remove ds+log_edge which makes a new matrix (expensive!)
def clustering(ds, lam = 0.5, convergence_goal=100, 
               max_iter = 100, min_iter = 5, edge_fraction=True):
    """
    PROPER COMMENT HERE
    """
    start_time = time.time()
    n = ds.shape[0]
    upper_triangle = np.triu_indices(n, 1)
    A = np.zeros(shape=(n, n, n), dtype="float32")
    A1 = np.zeros(shape=(n, n, n), dtype="float32")    
    A_sum = np.copy(ds)
    
    log_edge = 0.0
    
    for num_iters in xrange(max_iter):
        print num_iters, n
        iter_goal = GCExt.update_Amatrix32(A, ds+log_edge, A_sum, A1, n, lam)
        if edge_fraction:
            blue_edges = np.sum((A_sum[upper_triangle] > 0))
            red_edges = n*(n-1)/2 - blue_edges
            log_edge = np.log(blue_edges) - np.log(red_edges)
        
        if iter_goal >= convergence_goal and num_iters >= min_iter:
            break
    
    ## get connected components
    graph = nx.Graph()
    graph.add_edges_from(zip(*np.where(A_sum >= 0)))
    
    comps =nx.connected_components(graph) 
    return comps, (iter_goal >= convergence_goal), num_iters, (time.time() - start_time)

def filter_patterns_for_noise(bbp, em, max_noise=0.05):
    B = em.shape[1] ## number of bits
    prop_error = np.sum(~em, axis=1) / float(B)
    ## remove patterns with more than max_noise percent of positions showing error
    noisy_pattern_filter = prop_error < max_noise
    return noisy_pattern_filter

def cluster_patterns(binary_bit_pattern, error_mask, strand, flip_rate, 
                     max_noise=MAX_PATTERN_NOISE, max_iter=1000, 
                     dampening=0.5, converge_goal=100, edge_fraction=False):
    """
    cluster patterns for a given strand using transative clustering.
    reads that are too noisy do not get clustered and are assigned a cluster index of -1.
    options include:
    FILL OUT
    """
    ## for a strand load bit pattern and error mask
    bbp = binary_bit_pattern[strand]
    em  = error_mask[strand]
    N, B = bbp.shape
    ## component index specifies the component of each read
    comp_index = -np.ones(N, dtype=int)
    
    ## filter out noisy patterns
    noisy_pattern_filter = filter_patterns_for_noise(bbp, em, max_noise = max_noise)
    bbp = bbp[noisy_pattern_filter]
    em  = em[noisy_pattern_filter]
    
    error_rate = np.sum(~em) / (2. * em.shape[0]*em.shape[1])
     
    ## combine bbp and em into a single integer to collapse identical patterns and masks
    pattern_key = [to_binary(list(x) + list(y)) for x, y in zip(bbp, em)]
    ## and get all unique elements with inverse and multiplicity
    values, unique_index, unique_inverse, multiplicity = np.unique(pattern_key, return_index = True, return_inverse=True, return_counts=True)
    
    ## restrict to unique patterns and masks
    bbp = bbp[unique_index]
    em  = em[unique_index]
    
    ## check that there is something left
    if bbp.shape[0] * bbp.shape[1] == 0:
        comps = []
        converged_flag = True
        num_iterations = 0
        time_taken = 0.
    else:
        ## get hamming distance and overlap and apply transative clustering
        mismatch, overlap = get_hamming_and_overlap(bbp, em)
        initial_ds = get_likelihood_difference(flip_rate, error_rate, mismatch, overlap, multiplicity)
        comps, converged_flag, num_iterations, time_taken = clustering(initial_ds, max_iter = max_iter, lam = dampening, convergence_goal = converge_goal, edge_fraction=False)
        ## sort clusters by size taking into account multiplicity
        comps.sort(key = lambda comp: np.sum([multiplicity[x] for x in comp]), reverse=True)
            
    cluster_lookup = [0]*len(values)
    for ind, comp in enumerate(comps):
        for pattern_index in comp:
            cluster_lookup[pattern_index] = ind
        
    post_filter_map = [cluster_lookup[pattern_index] for pattern_index in unique_inverse] 
    comp_index[noisy_pattern_filter] = post_filter_map
    
    return comp_index, converged_flag, num_iterations, time_taken, error_rate


def double_plot(chrom, start, end, seq_dic, joint_seq, bit_masks, strand, comp_index, color_list=DEFAULT_COLOR_LIST):    
    val, cluster_breaks = np.unique(np.sort(comp_index), return_index=True)
    order = np.argsort(comp_index)
    N = joint_seq[strand].shape[0]
    fig = plt.figure(figsize=(30, 15))
    cmap = mpl.colors.ListedColormap(DEFAULT_COLOR_LIST)
    
    chromstring = "%s:%d-%d" % (chrom, start, end)
    OTstring = ["OB", "OT"][strand]
    plot_title = "%s %s\n%d reads in %d clusters" % (chromstring, OTstring, N, len(cluster_breaks))
    fig.suptitle(plot_title, fontsize=30)
    full_data_shape = [0.035, 0.05, 0.5, 0.85]
    bits_data_shape = [0.56, 0.05, 0.35, 0.85]
    colorbar_shape = [0.925, 0.05, 0.025, 0.85]
    ax1 = fig.add_axes(full_data_shape, frame_on=True)
    ax2 = fig.add_axes(bits_data_shape, sharey=ax1, frame_on=True)
    ax3 = fig.add_axes(colorbar_shape, frame_on=True)
    
    plot_base_data(ax1, chrom, start, end, seq_dic, joint_seq[strand][order], color_list=color_list, mask=None)
    plot_base_data(ax2, chrom, start, end, seq_dic, joint_seq[strand][order], color_list=color_list, mask=bit_masks[strand])
    
    #ax1.set_ylabel("cluster size", fontsize=20)
    ax1.set_xlabel("fragment position", fontsize=20)
    ax2.set_xlabel("bit position", fontsize=20)
    bounds = [-1, 0, 1, 2, 3, 4, 5, 6]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                         norm=norm,
                                         boundaries=bounds,
                                         ticks=np.array(bounds, dtype=float) + 0.5, # optional
                                         spacing='proportional',
                                         orientation='vertical')
    
    cb2.set_ticklabels(["match", "not covered", "A", "C", "G", "T", "N"])
    
    for y in cluster_breaks:
        ax1.axhline(y - 0.5, color="LightGray", lw=2)
        ax2.axhline(y - 0.5, color="LightGray", lw=2)
    
    #    for y, clen in zip(cluster_mid, cluster_size):
    #        ax2.text(-2, y, str(clen), fontsize=20)
    #        ax2.text(L2, y, str(clen), fontsize=20)
    
    #ylabels = [str(clen) for clen in cluster_size]
    #
    #ax1.set_yticks(cluster_mid)
    #ax1.set_yticklabels(ylabels, fontsize=20)
    #ax2.set_yticks(cluster_mid)
    #ax2.set_yticklabels(ylabels, fontsize=20)
    #ax1.set_ylim(-0.5, N-0.5)
    #ax2.set_ylim(-0.5, N-0.5)
    plt.show()


flip_rate = 0.4
error_rate = 0.002
max_iter = 100
dampening = 0.5
converge_goal = 100


seq_pickle = "/data/safe/levy/museq/hg38.genome.pickle"
print 'loading genome sequence from %s' % seq_pickle
startTime = time.time()
seq_dic = cPickle.load(open(seq_pickle))
print 'loaded in', time.time() - startTime

chrom_list = ['chr%s' % x for x in range(1, 23)] + ["chrX", "chrY"]
TABLE_DIR = "/data/safe/vsreeniv/muSeq/rep/tables"
interval_bedfile = os.path.join(TABLE_DIR, "TheoreticalRepresentationFragments.bed")
RFOI_index = load_restriction_fragments_of_interest(interval_bedfile)
PROJECT = 'next_seq'
bamfile   = ('/data/safe/vsreeniv/muSeq/rep/%s/' + 
                          'bowtie2Align.sorted.bam') % PROJECT
bamreader = pysam.AlignmentFile(bamfile, 'rb') 

index = 105511
#index = 6 ## two hom and 1 het
#index = 90000
chrom, start, end = RFOI_index[index]
chrom = str(chrom)
type(chrom)

print chrom, start, end
read_lists = extract_reads_for_fragment(bamreader, chrom, start, end, buff_in=2)

seq, qual, joint_seq = get_all_referenced_arrays(read_lists, start, end, conservative=True)

base_ratios = get_base_ratios(joint_seq)

bit_masks = get_bit_masks(base_ratios, homo_thresh=0.9)

bit_patterns = [js[:, bm] for js, bm in izip(joint_seq, bit_masks)]

binary_bit_pattern, error_mask = bits_to_patterns_and_error(bit_patterns)

np.sum(error_mask[0])

for strand in [0, 1]:
    comp_index, converged_flag, num_iterations, time_taken, error_rate = cluster_patterns(binary_bit_pattern, error_mask, strand, flip_rate)
    print error_rate
    double_plot(chrom, start, end, seq_dic, joint_seq, bit_masks, strand, comp_index, color_list=DEFAULT_COLOR_LIST)
    plt.show()


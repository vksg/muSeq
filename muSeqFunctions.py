#!/usr/bin/python2.7

import sys

MY_PYTHON_LIBS = ['/data/safe/vsreeniv/lib/python2.7/site-packages',
            '/data/safe/vsreeniv/python-cpp']
if sys.path[:2] != MY_PYTHON_LIBS:
    sys.path =  MY_PYTHON_LIBS + sys.path

import pysam
import numpy as np
import time
import pandas as pd
import os
from collections import Counter, defaultdict
import cPickle
import operator
import scipy.optimize as opt
import GCExt
import networkx as nx
import resource
from itertools import imap
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import izip

NE = operator.ne


try:
    SEQ_DIC.keys()[:1]
except:
    print 'loading cPickled genome'
    SEQ_DIC = cPickle.load(open("/data/safe/levy/museq/hg38.genome.pickle"))
    
## which nucleotide does the bit convert to
FLIP_NUC = {'C':'T', 'G':'A'}

## which nucleotides are certainly errors for a certain bit
ERROR_NUC = {'C': ['G', 'A'], 'G':['C','T']}

NUC_INDEX = {'A':0, 'C':1, 'T':2, 'G':3, '-':4, 'N':5, '*':6}
NUCS = ['A', 'C', 'T', 'G', '-', 'N', '*']

TABLE_DIR = '/data/safe/vsreeniv/muSeq/rep/tables/'
PROJECT = 'next_seq'

BAMFILE   = ('/data/safe/vsreeniv/muSeq/rep/%s/' + 
                          'bowtie2Align.sorted.bam') % PROJECT

'''
R 	A or G
Y 	C or T
S 	G or C
W 	A or T
K 	G or T
M 	A or C
'''
'''
I'm making up new ones for deletions
E   A or -
F   C or -
H   T or -
I   G or -
'''
DINUC_DICT = {'AG': 'R', 'CT':'Y', 'GC':'S', 
              'AT':'W', 'GT':'K', 'AC':'M'}
## add equals
DINUC_DICT.update([[x+x, x] for x in NUCS])
## add singles
DINUC_DICT.update([[x, x] for x in NUCS])
## add N's
DINUC_DICT.update([[x+'N', x] for x in NUCS])
## add *'s
DINUC_DICT.update([[x+'*', x] for x in NUCS])
## add 
DINUC_DICT.update([[x+'-', y] for x, y in zip(NUCS, 'EFHI--')])
## add reversed
DINUC_DICT.update([[dn[::-1], DINUC_DICT[dn]] 
                    for dn in DINUC_DICT.keys()])

TOP_STRAND_DICT = {'C':'C', 'T':'T', 'CC': 'C', 'TT':'T'}

BOTTOM_STRAND_DICT = {'G':'G', 'A':'A', 'GG':'G', 'AA':'A'}

translateTop = np.vectorize(lambda y: TOP_STRAND_DICT.get(y, 'N'))
translateBottom = np.vectorize(lambda x: BOTTOM_STRAND_DICT.get(x, 'N'))

def translateDinuc(bit, x):
    if bit == 'C':
        return translateTop(x)
    else:
        return translateBottom(x)

        
PICKLE_PREFIX = '/data/safe/vsreeniv/muSeq/rep/next_seq/intervals/'


FLIP_RATE  = 0.40
ERROR_RATE =  0.002
## these are the categories for data that we will generate
HEADINGS = '''
chrom
start
end
frag_length
strand
totalbits
coveredbits
numreads
numclusters
flip_rate_ML
error_rate_ML
flip_mean
flip_std
intra_min
intra_q25
intra_q50
intra_q95
intra_max
intra_mean
inter_min
inter_q25
inter_q50
inter_q95
inter_max
inter_mean
rpc_mean
rpc_std
maxmem'''

HEADINGS = HEADINGS.strip().split('\n')


def computeConsensus(cluster_seqs):
    """Returns the majority consensus of a list of sequences. 
    Sequences must be of equal length and only contain the nucleotides
    A,C,T,G,N,-.
    """
    ## nothing to do if the cluster is a singleton
    if len(cluster_seqs) == 1:
        return cluster_seqs[0]
    
    base_counts = np.zeros((len(cluster_seqs), len(cluster_seqs[0]), len(NUCS)), dtype=int)
    
    for read_index, pattern in enumerate(cluster_seqs):
        for pattern_index, dinuc in enumerate(pattern):
            for nuc in dinuc:
                base_counts[read_index, pattern_index, NUC_INDEX[nuc]] += 1
    
    return ''.join([NUCS[index] for index in 
                    np.argmax(np.sum(base_counts, axis=0), axis=1)])

def hamming(str1, str2):
    """Returns the hamming distance of two strings of equal length. 
    If lengths are unequal, returns the hamming distance restricted 
    to the overlap."""
    return sum(imap(NE, str1, str2))

def flipErrorFunction(x, num_errors, unflipped, flipped):
    p_e, p_f = x
    
    l1 = np.log(p_e/3.)
    l2 = np.log(1. - p_f - p_e + 4.*p_e*p_f/3.)
    l3 = np.log(p_f + p_e/3. - 4.*p_e*p_f/3.)
    
    return -(l1*num_errors + l2*unflipped + l3*flipped)


def clustering(ds, lam = 0.5, convergence_goal=100, 
               max_iter = 100, min_iter = 5, edge_fraction=False):
    start_time = time.time()
    n = ds.shape[0]
    
    A = np.zeros(shape=(n, n, n), dtype="float32")
    A1 = np.zeros(shape=(n, n, n), dtype="float32")
    
    A_sum = np.copy(ds)
    log_edge = 0.0
    
    for i in xrange(max_iter):
        iter_goal = GCExt.update_Amatrix32(A, ds+log_edge, A_sum, A1, n, lam)
        if edge_fraction:
            blue_edges = np.sum((A_sum[upper_triangle] > 0))
            red_edges = n*(n-1)/2 - blue_edges
            log_edge = np.log(blue_edges) - np.log(red_edges)
        
        if iter_goal >= convergence_goal and i >= min_iter:
            break
    
    ## get connected components
    graph = nx.Graph()
    graph.add_edges_from(zip(*np.where(A_sum >= 0)))
    
    comps =nx.connected_components(graph) 
        
    return comps, (iter_goal >= convergence_goal), i, (time.time() - start_time)
        
def getReads(bamreader, chrom_bin, ot_bits, ob_bits,
             bit_error_tol=0.05, bit_coverage = 0.90):
    """
    Pull all the reads from the dataset that correspond to a given
    genomic region specified by the interval index in REP_DATA.
    
    Creates a cPickle with all the interval data 
    Returns:    (   seq_array, 
                    qual_array, 
                    all_covered_positions, 
                    bit_mask,
                    error_list, 
                    insertions  ) for each strand
    
    """
    ## change type locally (I have checked that it won't affect the global)
    chrom, start, end = chrom_bin
    
    ## grab all reads that are in this interval
    read_pairs = defaultdict(list)
    
    for read in bamreader.fetch(chrom, start, end):
        read_name = read.qname
        if PROJECT == 'next_seq':
            read_name += '.'+read.get_tag('RG')
            
        if (read.opt('BI') == 'C' ) ^ read.is_read2:
            read_pairs[read_name] = [read] + read_pairs[read_name]
        else:
            read_pairs[read_name].append(read)
    
    ## what are the read nucleotides over genome positions
    aligned_pairs_dict    = {'C': [], 'G': []}
    aligned_quals_dict    = {'C': [], 'G': []}
    ## insertions in reads
    insertions_list       = {'C': [], 'G': []}
    
    ## what is the read-coverage at each nucleotide in this bin
    coverage_counter = Counter()
    
    read_index = 0
    for pair in read_pairs.values():
        ## skip unpaired reads
        if len(pair) == 1:
            continue
        r1, r2 = pair
        bit = r1.get_tag('BI')
        
        ## read-pair must align exactly with fragment start and end
        # bit == 'C' and r1.reference_start - 1 == start
        # bit == 'G' and r1.reference_start == start
        start_proper = ( start == (r1.reference_start - 
                        ((PROJECT == 'next_seq')*(bit == 'C'))))
        end_proper   = ( end   == (r2.reference_end + 
                        ((PROJECT == 'next_seq')*(bit == 'G'))))
        
        if not (start_proper and end_proper):
            continue
        
        ## calculate the read coverage and 
        ## read nucleotides over a given genome position
        ## uses PYSAM get_aligned_pairs!!
        read_dict  = {}
        read_qual  = {}
        insertions = defaultdict(str)
        for read in pair:
            prev_gpos = 0
            for rpos, gpos in read.get_aligned_pairs():
                if gpos == None:
                    insertions[prev_gpos] += (read.seq[rpos]+','+
                                              read.qual[rpos]+';')
                    continue
                prev_gpos = gpos
                
                coverage_counter[gpos] += 1
                if rpos == None:
                    dinuc = read_dict.get(gpos, '')+'-'
                    diqual = read_qual.get(gpos, '')+'_'
                else:
                    dinuc = read_dict.get(gpos, '')+read.seq[rpos]
                    diqual = read_qual.get(gpos, '')+read.qual[rpos]
                
                ## use a dinuc for possible overlap between reads
                read_dict.update([[gpos, dinuc]])
                read_qual.update([[gpos, diqual]])
        
        aligned_pairs_dict[bit].append(read_dict)
        aligned_quals_dict[bit].append(read_qual)  
        insertions_list[bit].append(insertions.items())
        read_index += 1
        
    all_covered_positions = coverage_counter.keys()
    all_covered_positions.sort()
    
    both_records = []
    
    bit_list = {'C':ot_bits, 'G':ob_bits}
    
    for bit in ['C', 'G']:
        ## only consider bits that are covered by 90% of reads
        num_reads = len(aligned_pairs_dict[bit])
        min_bit_coverage = float(num_reads)*bit_coverage
        
        bit_mask = []
        for gpos in all_covered_positions:
            if (gpos in bit_list[bit] and 
                coverage_counter[gpos] >= min_bit_coverage):
                bit_mask.append(True)
            else:
                bit_mask.append(False)
        bit_mask = np.array(bit_mask)
        numbits = np.sum(bit_mask)
        
        ## list of aligned sequences
        seq_array  = []
        qual_array = []
        insert_list = []
        total_errors = 0
        for (align_seq, align_qual, insertion) in izip(aligned_pairs_dict[bit], aligned_quals_dict[bit], insertions_list[bit]):
            bit_errors = 0
            seq=[]
            qual=[]
            for i,x in enumerate(all_covered_positions):
                nuc = align_seq.get(x, '*')
                seq.append(nuc)
                qual.append(align_qual.get(x, '*'))
                
                if nuc == '*':
                    continue
                if bit_mask[i]:
                    bit_errors += not(nuc == bit or 
                                      nuc == FLIP_NUC[bit] or
                                      nuc == (bit*2) or
                                      nuc == (FLIP_NUC[bit]*2))
            
            if bit_errors <= bit_error_tol*numbits:
                seq_array.append(seq)
                qual_array.append(qual)
                insert_list.append(insertion)
                total_errors += bit_errors
        
        seq_array = np.array(seq_array, dtype='S2')
        qual_array = np.array(qual_array, dtype='S2')
        record = (  seq_array, 
                    qual_array, 
                    all_covered_positions, 
                    bit_mask, 
                    num_reads,
                    total_errors,
                    [(read_index, insertion) for read_index, insertion in enumerate(insert_list) if len(insertion)>0]
                 )
        both_records.append(record)
        
    return both_records


def getClusters(seq_array, bit_mask, bit, max_iter=100, converge_goal=100, damp=0.5):
    """
    Run the clustering algorithm.
    Input:    seq_array, bit_mask, bit
    Optional: max_iter=100, converge_goal=100, damp=0.5
    
    Returns:  clusters, converged_flag, num_iterations, time_taken
    """
    if len(seq_array) == 0:
        return [], True, 0, 0.0
    elif len(seq_array) == 1:
        return [0], True, 0, 0.0
    
    ## this puts in N's whenever there are ambiguities between read-pairs    
    
    bit_list = [''.join(bits) for bits in 
                            translateDinuc(bit, seq_array[:,bit_mask])]
    
    
    ## sort the patterns alphabetically
    ## heapsort is fastest
    ordering = np.argsort(bit_list, axis=0, kind='heapsort')
    bit_list = [bit_list[sort_index] for sort_index in ordering]
    
    ## collapse down to unique bit patterns
    read_tracker = []
    prev_pattern = bit_list[0]
    read_bunch = []
    for index, pattern in enumerate(bit_list):
        if pattern != prev_pattern:
            read_tracker.append(read_bunch)
            read_bunch = [index]
        else:
            read_bunch.append(index)
        prev_pattern = pattern

    read_tracker.append(read_bunch)
    ## unique pattern list
    uniq_list = np.array([np.fromstring(bit_list[x[0]], dtype='S1')
                              for x in read_tracker])
    ## multiplicity of unique patterns
    multiplicity = [len(x) for x in read_tracker]
    
    
    numreads = len(bit_list)
    print 'clustering %s reads' % len(uniq_list)
    
    numbits  = len(bit_list[0])
    
    ## CHANGE THIS: use a global flip rate and error rate estimate
    
    ## count the number of N's these are all the errors
    num_errors = 0.0
    for pattern in bit_list:
        num_errors += pattern.count('N')
    
    ## divide by 2 since C -> T = (C->G + C->A)/2
    error_vector = num_errors/(numreads*numbits)/2.0
    flip_vector  = 0.4
    
    
    flip_hamming = 2*flip_vector*(1.-flip_vector)
    error_hamming = 2*error_vector*(1 - error_vector)

    logpe = np.log(error_hamming)
    log1mpe = np.log(1.0 - error_hamming)
    logpf = np.log(flip_hamming)
    log1mpf = np.log(1.0-flip_hamming)
    
    num_patterns, num_bits = uniq_list.shape
    
    ## ignore bits that are definitely errors
    ## make a mask of all bits on all reads that we will keep
    keep_mask = uniq_list != 'N'
    
    ds = np.zeros((num_patterns, num_patterns), dtype="float32")
    
    for i in xrange(num_patterns):
        for j in xrange(i+1,num_patterns):
            ## only count mismatches that we are sure are mismatches
            common_mask = np.logical_and(keep_mask[i], keep_mask[j])
            overlap = np.sum(common_mask)
            
            mismatch = np.sum(uniq_list[i][common_mask] != 
                        uniq_list[j][common_mask])
            d1 = logpe*mismatch + (overlap-mismatch)*log1mpe
            d2 = logpf*mismatch + (overlap-mismatch)*log1mpf
            ds[i,j] = multiplicity[i]*multiplicity[j]*(d1-d2)
            ds[j,i] = ds[i,j]

    comps, converged_flag, num_iterations, time_taken = clustering(ds, max_iter = max_iter, lam = damp, convergence_goal = converge_goal)

    clusters = []
    consensus_reads = []
    for comp in comps:
        cluster = []
        for c in comp:
            ## this is because we re-ordered bit_list in the beginning
            cluster.extend([ordering[index] for index in read_tracker[c]])

        clusters.append(cluster)
    
    if len(clusters) == 0:
        return None
    
    return clusters, converged_flag, num_iterations, time_taken
    

def makeTable(interval_index, bit, seq_array, bit_mask, clusters):
    """
    Gather information about the interval and the clustering solution 
    into a list. Initializes the class attributes:
    self.tabledata
    self.tableheadings
    
    strand = 'C'/'G'/None. Default: None, initialize for both strands.
    """
    chrom, start, end = REP_DATA[interval_index]
    
    if clusters == []:
        return None
        
    bit_list = [''.join(bits) for bits in 
                            translateDinuc(bit, seq_array[:,bit_mask])]
    
    fv = FLIP_RATE
    ev = ERROR_RATE
    
    cluster_bits = []
    for cluster in clusters:
        cluster_bits.append(np.fromstring(
        computeConsensus([bit_list[i] for i in cluster]), dtype='S1') == FLIP_NUC[bit])
    
    cluster_bits = np.array(cluster_bits)
    
    flipped_bits = np.sum(cluster_bits == 1, axis=1)
    flits_mean = np.mean(flipped_bits)
    flits_std  = np.std(flipped_bits)
    
    totalbits    = self.totalbits
    coveredbits  = len(bit_list[0])
    numreads = len(bit_list)
    numclusters = len(clusters)
    
    reads_per_cluster = np.sort([len(x) for x in clusters])
    rpc = ",".join(map(str, reads_per_cluster))

    rpc_mean = np.mean(reads_per_cluster)
    rpc_std  = np.std(reads_per_cluster)

    num_singletons = np.sum(reads_per_cluster == 1)
    
    consensus_list = [computeConsensus([bit_list[i] for i in cluster]) for cluster in clusters]

    intrahamming = []
    for i in xrange(numclusters):
        L = len(clusters[i])
        if L > 1:
            for j in xrange(L):
                intrahamming.append(hamming(''.join(consensus_list[i]), ''.join(bit_list[clusters[i][j]])))

    interhamming = []
    singleton_hamming = []

    for i in xrange(numclusters):
        for j in xrange(i+1, numclusters):
            h = hamming(consensus_list[i], consensus_list[j])
            if len(clusters[i]) == 1 or len(clusters[j]) == 1:
                singleton_hamming.append( h )
            else:
                interhamming.append( h )

    if len(interhamming) == 0:
        interhamming = [0]

    if len(intrahamming) == 0:
        intrahamming = [0]
        
    interquart = np.percentile(interhamming, [0, 25, 50, 95, 100])
    intermean  = np.mean(interhamming)
    intraquart = np.percentile(intrahamming, [0, 25, 50, 95, 100])
    intramean  = np.mean(intrahamming)
        
    maxmem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dataout = ([chrom, start, end, end-start, bit,
               totalbits,coveredbits, numreads, numclusters, 
               fv, ev, flits_mean, flits_std] + 
               list(intraquart) + [intramean] + 
               list(interquart) + [intermean] +
               [rpc_mean, rpc_std, maxmem/1000.0])
    
    return dataout

def visualize(chrom_bin, bit, seq_array, bit_mask, coveredpos, clusters):
    """
    visualize clustering for one strand or both strands.
    """
        
    if len(seq_array) <= 1:
        return
    
    chrom, start, end = chrom_bin
    colorlist = [
    "gray", 
    "#554236",
    "#F77825",
    "#D3CE3D",
    "#F1EFA5",
    "#60B99A"]

    N = len(seq_array)
    
    cluster_size  = [len(x) for x in clusters]
    ## sort by size
    cluster_order    = np.argsort(cluster_size)[::-1]
    clusters = [clusters[i] for i in cluster_order]
    cluster_size     = [cluster_size[i] for i in cluster_order]
    cluster_index = np.zeros(N, dtype=int)
    cluster_breaks = np.cumsum(cluster_size)
    for i, rlist in enumerate(clusters):
        cluster_index[rlist] = i

    num_bits = np.sum(bit_mask)
    num_bases = len(seq_array[0])
    
    read_order = np.array([item for sublist in clusters for item in sublist])
    
    covered_bases = np.array(coveredpos) - start
    covered_bits  = np.array(coveredpos)[bit_mask] - start
    refbase = np.array([NUC_INDEX[SEQ_DIC[chrom][x]] for x in coveredpos])
    
    basecalls = np.array([[NUC_INDEX.get(DINUC_DICT[b], NUC_INDEX['N']) for b in seq] for seq in seq_array])
    basecalls = basecalls[read_order]
    
    ## if you match the reference 
    mask = basecalls == refbase[covered_bases]
    basecalls[mask] = -1
    
    ## N or - mean that it is not a covered position
    not_covered_mask = np.logical_or(np.logical_or(basecalls == NUC_INDEX['-'], basecalls == NUC_INDEX['N']), basecalls == NUC_INDEX['*'])
    basecalls[not_covered_mask] = -2

    cmap = mpl.colors.ListedColormap(colorlist)

    fig = plt.figure(figsize=(30, 15))
    chromstring = "%s:%d-%d" % (chrom, start, end)
    plot_title = "%s bit=%s\n%d reads in %d clusters" % (chromstring, bit, N, len(cluster_size))
    fig.suptitle(plot_title, fontsize=30)
    full_data_shape = [0.035, 0.05, 0.5, 0.85]
    bits_data_shape = [0.56, 0.05, 0.35, 0.85]
    colorbar_shape = [0.925, 0.05, 0.025, 0.85]
    ax1 = fig.add_axes(full_data_shape, frame_on=True)
    ax2 = fig.add_axes(bits_data_shape, sharey=ax1, frame_on=True)
    ax3 = fig.add_axes(colorbar_shape, frame_on=True)
    
    ax1.imshow(basecalls, interpolation="nearest", aspect='auto', vmin=-2, vmax=4, cmap=cmap)
    ax2.imshow(basecalls[:, bit_mask], interpolation="nearest", aspect='auto', vmin=-2, vmax=4, cmap=cmap)
    
    ax1.set_ylabel("cluster size", fontsize=20)
    ax1.set_xlabel("fragment position", fontsize=20)
    ax2.set_xlabel("bit position", fontsize=20)
    
    bounds = [-2, -1, 0, 1, 2, 3, 4]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                         norm=norm,
                                         boundaries=bounds,
                                         ticks=np.array(bounds, dtype=float) + 0.5, # optional
                                         spacing='proportional',
                                         orientation='vertical')

    cb2.set_ticklabels(["not covered", "match", "A", "C", "T", "G"])

    for y in cluster_breaks:
        ax1.axhline(y - 0.5, color="white", lw=2)
        ax2.axhline(y - 0.5, color="white", lw=2)

    ax1.set_ylim(-0.5, N - 0.5)


def mismatchStatistics(pickle_path='mismatch.cPickle.v2'):
    mismatch_data = []
    num_samples = 1000
    sample_intervals=np.random.choice(100000, size=num_samples)

    sample_count = 0
    for index in sample_intervals:
        print 'interval', index, ', ', REP_DATA[index]
        sample_count += 1
        print 'sample', sample_count
        
        read_record = getReads(BAMREADER, index)
        
        for bit, reads in zip(['C', 'G'], read_record):
            print bit
            (seq_array, qual_array, all_covered_positions, 
             bit_mask, error_list, insertions_list) = reads
            numbits = float(np.sum(bit_mask))
            if numbits < 30:
                continue
            data = [Counter(error_list).items(), numbits]
            good_reads = np.sum((np.array(error_list)/numbits) <= 0.05)
            data.append(float(good_reads)/seq_array.shape[0])
            
            good_reads = np.sum((np.array(error_list)/numbits) <= 0.02)
            data.append(float(good_reads)/seq_array.shape[0])
            
            good_reads = np.sum((np.array(error_list)/numbits) <= 0.01)
            data.append(float(good_reads)/seq_array.shape[0])
            
            good_reads = np.sum(np.array(error_list) <= 5)
            data.append(float(good_reads)/seq_array.shape[0])
            
            good_reads = np.sum(np.array(error_list) <= 3)
            data.append(float(good_reads)/seq_array.shape[0])
            
            mismatch_data.append(data)
            
        sys.stdout.flush()

    fo = open(pickle_path, 'w')
    cPickle.dump(mismatch_data, fo)
    fo.close()

def readIntervalsAndBits():
    rep_interval_file = os.path.join(TABLE_DIR,
                              'TheoreticalRepresentationFragments.bed')
    ot_pos_file = os.path.join(TABLE_DIR, 'ot_bits.txt.v3')
    ob_pos_file = os.path.join(TABLE_DIR, 'ob_bits.txt.v3')

    def getBits(bit_file, rep_data):
        print 'reading', bit_file
        bit_dict = defaultdict(list)
        interval_index = 0
        for line in open(bit_file, 'r'):
            chrom, pos = line.strip().split('\t')
            pos = int(pos)-1

            ichrom, istart, iend = rep_data[interval_index]
            
            ## first get the right chrom
            while ichrom != chrom:
                interval_index += 1
                ichrom, istart, iend = rep_data[interval_index]
            
            if pos < istart:
                continue
            
            ## next get to the right start, end
            while pos >= iend:
                interval_index += 1
                ichrom, istart, iend = rep_data[interval_index]
            
            ## now pos >= istart
            if pos < iend:
                bit_dict[interval_index].append(pos)
        return bit_dict
    
    rep_data = []
    for line in open(rep_interval_file, 'r'):
        c,s,e = line.strip().split('\t')
        rep_data.append((c,int(s), int(e)))
    
    ot_bits = getBits(ot_pos_file, rep_data)
    ob_bits = getBits(ob_pos_file, rep_data)
    
    return (rep_data, ot_bits, ob_bits)
        
###



## TEST CODE BELOW

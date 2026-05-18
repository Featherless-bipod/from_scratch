import torch
import re
from torch import nn

_restypes = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",]
_restypes_with_x = _restypes + ["X"]
_restypes_with_x_and_gap = _restypes_with_x + ["-"]

restype_order_with_x = None
restype_order_with_x_and_gap = None

restype_order_with_x = {res: i for i, res in enumerate(_restypes_with_x)}
restype_order_with_x_and_gap = {res:i for i, res in enumerate(restype_order_with_x_and_gap)}

def load_a3m_file(file_name: str):

    seqs = None

    with open(file_name,'r') as file:
        lines = file.readlines()

    description_line_indicies = [i for i,l in enumerate(lines) if l.startswith('>') ]
    seqs = [lines[i+1].strip() for i in description_line_indicies]

    return seqs



def onehot_encode_aa_type(seq, include_gap_token=False):

    restype_order = restype_order_with_x if not include_gap_token else restype_order_with_x_and_gap

    sequence = torch.tensor([restype_order[a] for a in seq])
    encoding = nn.functional.one_hot(sequence, num_classes = len(restype_order))

    return encoding



def initial_data_from_seqs(seqs):

    unique_seqs = None
    deletion_count_matrix = None
    aa_distribution = None
    deletion_count_matrix = []
    unique_seqs = []

    for seq in seqs:
        deletion_count_list = []
        deletion_count = 0
        for letter in seq:
            if letter.islower():
                deletion_count += 1
            else:
                deletion_count_list.append(deletion_count)
                deletion_count = 0
        seq_without_deletion = re.sub('[a-z]','',seq)
        
        if seq_without_deletion in unique_seqs:
            continue
        
        unique_seqs.append(seq_without_deletion)
        deletion_count_matrix.append(deletion_count_list)

    unique_seqs = torch.stack([onehot_encode_aa_type(seq, include_gap_token=True) for seq in unique_seqs], dim=0) #convert into a tesnor from list
    unique_seqs = unique_seqs.float()
    aa_distribution = unique_seqs.mean(dim=0) #convert to tensor
    deletion_count_matrix = torch.tensor(deletion_count_matrix).float()

    return { 'msa_aatype': unique_seqs, 'msa_deletion_count': deletion_count_matrix, 'aa_distribution': aa_distribution}

def select_cluster_centers(features, max_msa_clusters=512, seed=None):

    N_seq, N_res = features['msa_aatype'].shape[:2]
    MSA_FEATURE_NAMES = ['msa_aatype', 'msa_deletion_count']
    max_msa_clusters = min(max_msa_clusters, N_seq)

    gen = None
    if seed is not None:
        gen = torch.Generator(features['msa_aatype'].device)
        gen.manual_seed(seed)

    shuffled = torch.randperm(N_seq-1, generator=gen) + 1
    shuffled = torch.cat((torch.tensor[0],shuffled),dim=0)

    for key in MSA_FEATURE_NAMES:
        extra_key = f'extra_{key}'
        value = features[key]
        features[extra_key] = value[shuffled[max_msa_clusters:]]
        features[key] = value [shuffled[:max_msa_clusters]]

    return features

def mask_cluster_centers(features, mask_probability=0.15, seed=None):

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_aa_categories = 23 # 20 Amino Acids, Unknown AA, Gap, masked_msa_token
    odds = {
        'uniform_replacement': 0.1,
        'replacement_from_distribution': 0.1,
        'no_replacement': 0.1,
        'masked_out': 0.7,
    }
    gen = None
    if seed is not None:
        gen = torch.Generator(features['msa_aatype'].device)
        gen.manual_seed(seed)
        torch.manual_seed(seed)

    #prepare the possibiliites
    uniform_replacement = torch.tensor([1/20]*20+[0,0]) * odds['uniform_replacement'] #(22,)
    no_replacement = features['msa_aatype'] * odds['no_replacement']#[N_clust,N_res,22] how does this do one line at a time meow
    replacement_from_distribution = features['aa_distribution'] * odds['replacement_from_distribution'] #[N_res,22]
    masked_out = torch.ones[(N_clust,N_res,1)] * odds['masked_out']

    #reshape and add together
    uniform_replacement = uniform_replacement[None,None,...].broadcast_to(no_replacement.shape)
    replacement_from_distribution = replacement_from_distribution[None,...].broadcast_to(no_replacement.shape)
    mask_without_token = uniform_replacement + no_replacement + replacement_from_distribution
    mask_with_token = torch.cat((mask_without_token,masked_out),dim = -1)
    mask_with_token = mask_with_token.reshape((-1,N_aa_categories)) #needed for the sampling

    #selecting and one-hotting
    replace_with = torch.distributions.Categorical(mask_with_token).sample()
    replace_with = nn.functional.one_hot(replace_with, num_classes = N_aa_categories)
    replace_with = replace_with.reshape(N_clust,N_res,N_aa_categories)
    replace_with = replace_with.float()

    #create the random mask
    rand_mask = torch.rand((N_clust,N_res),generator = gen) < mask_probability

    features['true_msa_aatype'] = features['msa_aatype'].clone()
    mask_padding = torch.zeros((N_clust, N_res, 1)) #create exrtra tensor space for mask token
    features['msa_aatype'] = torch.cat((features['msa_aatype'], mask_padding), dim=-1)
    features['msa_aatype'][rand_mask] = replace_with[rand_mask] #replaces via indexing

    return features

def cluster_assignment(features):

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    msa_slice = features['msa_aatype'][...,:21]
    extra_msa_slice = features['extra_msa_aatpe'][...,:21]

    agreement = torch.einsum('cra,era—>ce',msa_slice,extra_msa_slice) #works because of one_hot
    assignment = torch.argmax(agreement, dim = 0 ) #collapses down to one row
    features['cluster_assignment'] = assignment

    assignment_counts = torch.bincount(assignment, minlength=N_clust)
    features['assignment_counts'] = assignment_counts
            
    return features

def cluster_average(feature, extra_feature, cluster_assignment, cluster_assignment_count):

    N_clust, N_res = feature.shape[:2]
    N_extra = extra_feature.shape[0]

    extra_shape = (N_extra,) + (1,) * (extra_feature.dim()-1)
    cluster_assignment = cluster_assignment.view(extra_shape).broadcast_to(extra_feature.shape) #need this cuz broadcast too broad, also no mathmatical reason, we just need shape to fit the scatteradd funciton
    cluster_shape = (N_clust,) + (1,) * (feature.dim()-1)
    cluster_assignment_count = cluster_assignment_count.view(cluster_shape).broadcast_to(feature.shape)

    cluster_sum = torch.scatter_add(feature, dim = 0, index = cluster_assignment, src=extra_feature) #essentially adds everything to the feature so then divided later to get average
    
    cluster_average = cluster_sum/(cluster_assignment_count+1)

    return cluster_average



def summarize_clusters(features):
    """
    Calculates cluster summaries by applying cluster averaging to the MSA amino acid 
    representations and deletion counts.

    Args:
        features: A dictionary containing feature representations of the MSA.

    Modifies:
        The 'features' dictionary in-place by adding the following:
            * cluster_deletion_mean: Average deletion counts for each cluster center, 
                                     scaled for numerical stability.
            * cluster_profile: Average amino acid representations for each cluster center.
    """

    N_clust, N_res = features['msa_aatype'].shape[:2]
    N_extra = features['extra_msa_aatype'].shape[0]

    cluster_deletion_mean = cluster_average(
        features['msa_deletion_count'],
        features['extra_msa_deletion_count'],
        features['cluster_assignment'],
        features['cluster_assignment_count']
        )
    cluster_deletion_mean = 2/torch.pi * torch.arctan(cluster_deletion_mean/3)

    zero_pad = torch.zeros(features['extra_msa_aatype'].shape[:-1]+(1,),dtype = features['extra_msa_aatype'].dtype, device=features['extra_msa_aatype'].device)
    extra_msa_aatype_pad = torch.cat((features['extra_msa_aatype'],zero_pad),dim =-1)
    cluster_aa_avg = cluster_average(
        features['msa_aatype'],
        extra_msa_aatype_pad,
        features['cluster_assignment'],
        features['cluster_assignment_count']
    )

    features['cluster_deletion_mean'] = cluster_deletion_mean
    features['cluster_profile'] = cluster_aa_avg

    return features

def crop_extra_msa(features, max_extra_msa_count=5120, seed=None):

    N_extra = features['extra_msa_aatype'].shape[0]
    gen = None
    if seed is not None:
        gen = torch.Generator(features['extra_msa_aatype'].device)
        gen.manual_seed(seed)

    max_extra_msa_count = min(max_extra_msa_count, N_extra)

    rand_index = torch.randperm(N_extra, generator = gen)[:max_extra_msa_count]
    for k in features.keys():
        if k.startswith('extra_'):
            features[k] = features[k][rand_index]

    return features

def calculate_msa_feat(features):
    
    N_clust, N_res = features['msa_aatype'].shape[:2]
    msa_feat = None

    cluster_msa = features['cluster_msa']
    msa_deletion_count = features['msa_deletion_count']
    cluster_deletion_mean = features['cluster_deletion_mean']
    cluster_profile = features['cluster_profile']

    cluster_has_deletion = (msa_deletion_count > 0).float().unsqueezed(-1)
    cluster_deletion_value = 2/torch.pi * torch.arctan(msa_deletion_count/ 3)

    msa_feat = torch.cat((cluster_msa,cluster_has_deletion,cluster_deletion_value,cluster_profile,cluster_deletion_mean))

    return msa_feat

def calculate_extra_msa_feat(features):

    N_extra, N_res = features['extra_msa_aatype'].shape[:2]
    extra_msa_feat = None

    extra_msa_aatype = features['extra_msa_aatype']
    extra_msa_deletion_count = features['extra_msa_deletion_count']

    padding =torch.zeros((N_extra,N_res,1))
    extra_msa_aatype = torch.cat((extra_msa_aatype,padding),dim=-1)

    extra_msa_has_deletion = (extra_msa_deletion_count > 0).float().unsqueezed(-1)
    extra_msa_deletion_value = 2/torch.pi*torch.arctan(extra_msa_has_deletion/3)

    extra_msa_feat = torch.cat((extra_msa_aatype,extra_msa_has_deletion,extra_msa_deletion_value),dim=-1)

    return extra_msa_feat



def create_features_from_a3m(file_name, seed=None):

    msa_feat = None
    extra_msa_feat = None
    target_feat = None
    residue_index = None
    select_clusters_seed = None
    mask_clusters_seed = None
    crop_extra_seed = None
    if seed is not None:
        select_clusters_seed = seed
        mask_clusters_seed = seed+1
        crop_extra_seed = seed+2

    seqs = load_a3m_file(file_name)
    features = initial_data_from_seqs(seqs)

    transforms = [
        lambda x: select_cluster_centers(x, seed=select_clusters_seed),
        lambda x: mask_cluster_centers(x, seed=mask_clusters_seed),
        cluster_assignment,
        summarize_clusters,
        lambda x: crop_extra_msa(x,seed=crop_extra_seed)
    ]

    for transform in transforms:
        features = transform(features)

    msa_feat = calculate_msa_feat(features)
    extra_msa_feat = calculate_extra_msa_feat(features)

    target_feat = onehot_encode_aa_type(seqs[0],include_gap_token = False).float()
    residue_index = torch.arange(len(seqs[0]))

    return {
        'msa_feat': msa_feat,
        'extra_msa_feat': extra_msa_feat,
        'target_feat': target_feat,
        'residue_index': residue_index
    }

def create_control_values(base_folder):
    file_name = f'{base_folder}/alignment_tautomerase.a3m'
    control = f'{base_folder}/control_values'

    seqs = load_a3m_file(file_name)

    initial_data = initial_data_from_seqs(seqs)
    torch.save(initial_data, f'{control}/initial_data.pt')
    clusters_selected = select_cluster_centers(initial_data, seed=0)
    torch.save(clusters_selected, f'{control}/clusters_selected.pt')
    clusters_masked = mask_cluster_centers(clusters_selected, seed=1)
    torch.save(clusters_masked, f'{control}/clusters_masked.pt')
    clusters_assigned = cluster_assignment(clusters_masked)
    torch.save(clusters_assigned, f'{control}/clusters_assigned.pt')
    clusters_summarized = summarize_clusters(clusters_assigned)
    torch.save(clusters_summarized, f'{control}/clusters_summarized.pt')
    extra_msa_cropped = crop_extra_msa(clusters_summarized, seed=2)
    torch.save(extra_msa_cropped, f'{control}/extra_msa_cropped.pt')

    msa_feat = calculate_msa_feat(extra_msa_cropped)
    extra_msa_feat = calculate_extra_msa_feat(extra_msa_cropped)
    torch.save(msa_feat, f'{control}/msa_feat.pt')
    torch.save(extra_msa_feat, f'{control}/extra_msa_feat.pt')


    full_batch = create_features_from_a3m(file_name, seed=0)
    torch.save(full_batch, f'{control}/full_batch.pt')


    
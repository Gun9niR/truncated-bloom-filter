from skippingbaselines import *
# from skippingbaselinesjointbloom import *
from relationalquerygeneration import RelationalLoader
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 300
plt.style.use("seaborn-v0_8-paper")
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}

plt.rcParams.update(params)


x, y, z = ('data/access_frequency_skipping_data/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet',
           'data/access_frequency_skipping_data/Real_Estate/queries',
              'data/access_frequency_skipping_data/Real_Estate/utilities/utilities.npy')

rl = RelationalLoader(x, y, z)

group_keys_all, column_dtypes, column_names, rg_size  = rl.extract_group_keys()

# get address column only
column_names = column_names[1:]
column_dtypes = column_dtypes[1:]
group_keys_all = [group_keys_all[1]]

# group_keys_all, column_dtypes, column_names  = rl.extract_group_keys()
alpha_queries, non_alpha_queries = rl.load_queries()
utilities = rl.load_utilities()



crs = np.arange(0.1, 1.0, 0.2)
fpr = 0.0001

# RE
k = 22
# NASA
# k = 12
# EV
# k = 5

index_sizes_bb = []
index_sizes_elas_bf = []
filter_lengths_bb_li = []
filter_lengths_elas_bf_li = []

for cr in crs:
    print("CR:", cr)
    # bs = BloomSkipping(column_names, column_dtypes, len(group_keys_all[0]),
    #                utilities, "data/skipping_data_processed/Real_Estate/parquet/Real_Estate_Sales_2001-2020.parquet", len(group_keys_all[0][0]))
    # bs = BloomSkipping(column_names, column_dtypes, len(group_keys_all[0]),
    #                utilities, x, len(group_keys_all[0][0]))

    
    # bs_index_size = bs.index_size()
    
    # imet = InMemEqualTrunc(column_names, column_dtypes, len(group_keys_all[0]),
    #                utilities, x, rg_size)
    # imet.construct_indexes(group_keys_all, fpr, cr)
            
    # tu = TopUtility(column_names, column_dtypes, len(group_keys_all[0]),
    #                utilities, x, rg_size)
    # tu.construct_indexes(group_keys_all, fpr, cr)
    
    # hybloom = HybridBloom(column_names, column_dtypes, len(group_keys_all[0]),
    #                  utilities, x, rg_size)
    # hybloom.construct_indexes(group_keys_all, fpr, cr)
    
    # tuh = TopUtilityHybrid(column_names, column_dtypes, len(group_keys_all[0]),
    #                     utilities, x, rg_size)
    # tuh.construct_indexes(group_keys_all, fpr, cr)
    

    
    bs = BloomSkipping(column_names, column_dtypes, len(group_keys_all[0]),
                   utilities, x, rg_size)
    bs.construct_indexes(group_keys_all, fpr, cr)
    
    row_group_util = bs.row_group_utilities
    
    
    
    filter_lengths_bb = [len(bf) for bf in list(bs.column_indexes.values())[0].bloom_filters]
    
    index_size_bb = list(bs.column_indexes.values())[0].index_size()
    
    
    elas_bf = ElasticBF(column_names, column_dtypes, len(group_keys_all[0]),
                        utilities, x, rg_size)
    elas_bf.construct_indexes(group_keys_all, fpr, cr)
    
    filter_lengths_elas_bf = [len(bf) for bf in list(elas_bf.column_indexes.values())[0].bloom_filters]
    
    index_size_elas_bf = list(elas_bf.column_indexes.values())[0].index_size()
    
    # append to lists
    index_sizes_bb.append(index_size_bb)
    index_sizes_elas_bf.append(index_size_elas_bf)
    
    filter_lengths_bb_li.append(filter_lengths_bb)
    filter_lengths_elas_bf_li.append(filter_lengths_elas_bf)
    
# make 2 horizontal subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
alphas = np.arange(0.1, 1.0, 0.2)
line_styles = ['-', '--', '-.', ':', '-', 'dashdotdotted']
# plot results with different transparency for each index size
for i, (fl_bb, fl_ebf) in enumerate(zip(filter_lengths_bb_li, filter_lengths_elas_bf_li)):
    if i == 0:
        ax1.plot(range(1, len(fl_bb)+1), fl_bb, label="BB: " + str(int(round(crs[i], 2)*100)) + '%', marker='*', linestyle=line_styles[i], alpha=alphas[i])
        # get color
        color = ax1.get_lines()[-1].get_color()
    else:
        ax1.plot(range(1, len(fl_bb)+1), fl_bb, label="BB: " + str(int(round(crs[i], 2)*100)) + '%' , marker='*', linestyle=line_styles[i], alpha=alphas[i], color=color)
    ax2.plot(range(1, len(fl_bb)+1), fl_ebf, label="EBF: " + str(int(round(crs[i], 2)*100)) + '%', marker='s', linestyle=line_styles[i], alpha=alphas[i], color='lightcoral')

# label each axis 
ax1.set_xlabel("Row group")
ax2.set_xlabel("Row group")

# shared y axis label for both plots
ax1.set_ylabel("Filter length (bits)")

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')


plt.savefig("REVISION_RESULTS/filter_lengths_real_estate.png", bbox_inches='tight')

plt.clf()

# plot row groups vs utility values
plt.plot(range(1, len(row_group_util)+1), row_group_util, color = 'grey')

plt.xlabel("Row group")
plt.ylabel("Utility value")

plt.savefig("REVISION_RESULTS/row_group_utility_real_estate.png", bbox_inches='tight')

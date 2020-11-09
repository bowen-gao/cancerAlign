# cancerAlign
cancerAlign: use adversarial learning to align different cancer types
# Introduction
cancerAlign is an unsupervised approach to stratify tumors through learning tumor alignment across cancer types. It utilizes the adversarial learning to learn a shared latent space between the somatic mutation profile of a target cancer type and a source cancer type. Tumors of the target cancer type are then clustered in this shared latent space so that common patterns across cancer types can be exploited to assist the stratification. 
# Dataset
patients' mutation data are in raw_survival/\*\_mut.txt &nbsp; * are cancer type names\
patients' survival data are in raw_survival/\*\_surv.txt_clean &nbsp; * are cancer type names\
A preprocessed file that contains the cancer type, mutatated genes for each patient: data.csv 
# How to run
To run the cancerAlign for a specific target cancer type:
 <pre><code>python3 run.py  --target="cancer type name"</code></pre>
To run and store the model of the mapping of a pair of cancer types target->source:
<pre><code>python3 pair_model.py --target="cancer type name for target cancer type" --source="cancer type name for source cancer type"</code></pre>
# How to see output
To generate the final clustering labels of a target cancer type when number of clusters is k by cancerAlign:
 <pre><code>python3 final_cluster.py  --target="cancer type name" --num_clusters=k</code></pre>
To generate cancer genes produced by cancerAlign when number of clusters is k:
 <pre><code>python3 cancer_genes.py  --target="cancer type name" --num_clusters=k</code></pre>

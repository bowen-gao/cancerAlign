# cancerAlign
<img src="https://s3.ax1x.com/2020/11/29/D67tHI.md.png" width="10%">[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pytorch implementation for cancerAlign: use adversarial learning to align different cancer types

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@article{cancerAlign,
  title={cancerAlign: Stratifying tumors by unsupervised alignment across cancer types},
  author={Gao, Bowen and Luo, Yunan and Ma, Jianzhu and Wang, Sheng},
  journal={arXiv preprint arXiv:2011.xxxxx},
  year={2020}
}
</pre>

# Abstract
Tumor stratification, which aims at clustering tumors into biologically meaningful subtypes, is the key step towards personalized treatment. Large-scale profiled cancer genomics data enables us to develop computational methods for tumor stratification. However, most of the existing approaches only considered tumors from an individual cancer type during clustering, leading to the overlook of common patterns across cancer types and the vulnerability to the noise within that cancer type. To address these challenges, we proposed cancerAlign to map tumors of the target cancer type into latent spaces of other source cancer types. These tumors were then clustered in each latent space rather than the original space in order to exploit shared patterns across cancer types. Due to the lack of aligned tumor samples across cancer types, cancerAlign used adversarial learning to learn the mapping at the population level. It then used consensus clustering to integrate cluster labels from different source cancer types. We evaluated cancerAlign on 7,134 tumors spanning 24 cancer types from TCGA and observed substantial improvement on tumor stratification and cancer gene prioritization. We further revealed the transferability across cancer types, which reflected the similarity among them based on the somatic mutation profile. cancerAlign is an unsupervised approach that provides deeper insights into the heterogeneous and rapidly accumulating somatic mutation profile and can be also applied to other genome-scale molecular information.

# Model Architecture
<p align='center'>
<img src="https://i.loli.net/2020/11/29/akTn1zlf27HhRx5.png" height="300"/>
</p>

# Dataset
download from http://gdac.broadinstitute.org/ \
patients' mutation data are in raw_survival/\*\_mut.txt &nbsp; * are cancer type names\
patients' survival data are in raw_survival/\*\_surv.txt_clean &nbsp; * are cancer type names\
A preprocessed file that contains the cancer type, mutatated genes for each patient: data.csv\
A known cancer gene list known_cancer_genes.csv

# Experiments
To run the cancerAlign for a specific target cancer type:
 <pre><code>python3 run.py  --target="cancer type name"</code></pre>
To generate the final clustering labels of a target cancer type when number of clusters is k by cancerAlign:
 <pre><code>python3 final_cluster.py  --target="cancer type name" --num_clusters=k</code></pre>
It would generate a file (target cancer type)\_k.txt. For example, for BLCA and k=2, the file name is BLCA\_2.txt. Inside the file, first column is patient names, second column is corresponding labels.

To generate cancer genes produced by cancerAlign:
 <pre><code>python3 cancer_genes.py  --target="cancer type name"</code></pre>
 It would print top 10 genes generated by cancerAlign for the target cancer type.
 
 # Questions

For questions about the data and code, please contact bgao@caltech.edu. We will do our best to provide support and address any issues. We appreciate your feedback!

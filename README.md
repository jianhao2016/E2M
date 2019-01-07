# E2M
source code and sample data for E2M

## Requirements
  * python 3.0+
  * [numpy](http://www.numpy.org/), [TensorFlow r1.2](https://www.tensorflow.org/versions/r1.2/api_docs/), [matplotlib](https://matplotlib.org/)

## Data Description
Raw methylation and expression files are available for download via [GDC data portal](https://portal.gdc.cancer.gov/).

Here we provide a small dataset of GBM in folder **/data/GBM/** for testing and validation of our method. Due to the space constraint, other dataset used in paper are available upon request
In folder **/data/download_from_TCGA**, we also provide sample codes on downloading/extracting other genes/cancer of interest. All dataset are stored in a Python dictionary format. 

## How to Use the Software
Folder **/data/** includes codes on downloading/extracting/preprocessing of data. More detail can be found in the *README* inside.

Folder **/script/** contains codes on MR (Multiclass Regression), FCNN (Fully Connected Neural Network), E2M and visualization.

### Install
```
# download METHCOMP, go to terminal and enter:
git clone https://github.com/jianhao2016/E2M.git
```
### Run
* To run E2M on the test file, first make sure that all the packages have been installed in the correct version of Python. 
  Then go to folder **/script/inception_with_diff_type_of_error/** and use the following command:

  `python multi_gene_multi_cancer_inception.py --cancer_type gbm --gene mgmt --quantization 0`

  This command will train an unquantized E2M on the provided GBM dataset and provides the accuracy on test set. You can also use 
  `python multi_gene_multi_cancer_inception.py --help` to see all available configuration options.

* To run MR or FCNN, go to folder **/script/MR_and_FCNN/** and use the following command:

  `python multi_class_regression.py --cancer_type gbm --gene mgmt`

  or 

  `python fcnn_multi_gene_multi_cancer.py --cancer_type gbm --gene mgmt`

  respectively
 
 * PCA/histogram/Heatmap visualization can be run in a similar way with correpsonding files 
  *PCA_visualization.py*, *plot_histogram.py* and *selected_gene_heatmap.py*


## Output Files

Prediction accuracy and final error will be printed on screen. 
All trained models will be stored in **/data/** in TensorFlow SavedModel format

# E2M
E2M source code and sample dataset.

## Requirements
  * python 3.0+
  * [numpy](http://www.numpy.org/), [TensorFlow r1.2](https://www.tensorflow.org/versions/r1.2/api_docs/), [matplotlib](https://matplotlib.org/)

## Data Description
Raw methylation and expression files are available for download via the [GDC data portal](https://portal.gdc.cancer.gov/).

A small dataset for GBM may be found in the folder **/data/GBM/** for testing and validation purposes. Additional larger datasets are available upon request.
The folder **/data/download_from_TCGA** also contains sample codes for downloading/extracting other genes/cancer datasets of interest. All datasets are stored in a Python dictionary format. 

## How to Use the Software
Folder **/data/** includes codes for downloading/extracting/preprocessing of the data. More details can be found inside the data folder under the *README* heading.

The folder **/script/** contains codes for MR (Multiclass Regression), FCNN (Fully Connected Neural Network), E2M and visualization.

### Install
```
# download METHCOMP, go to terminal and enter:
git clone https://github.com/jianhao2016/E2M.git
```
### Run
* To run E2M on the test file, first ensure that all the packages have been installed with the correct version of Python. 
  Then go to folder **/script/inception_with_diff_type_of_error/** and use the following command:

  `python multi_gene_multi_cancer_inception.py --cancer_type gbm --gene mgmt --quantization 0`

  This command will train an unquantized E2M on the provided GBM dataset and list the accuracy of the given test set. You can also use 
  `python multi_gene_multi_cancer_inception.py --help` to see all available configuration options.

* To run MR or FCNN, go to folder **/script/MR_and_FCNN/** and use the following command:

  `python multi_class_regression.py --cancer_type gbm --gene mgmt`

  or 

  `python fcnn_multi_gene_multi_cancer.py --cancer_type gbm --gene mgmt`

  respectively.
 
 * The PCA/histogram/Heatmap visualization codes may be run in a similar way, with the corresponding files 
  *PCA_visualization.py*, *plot_histogram.py* and *selected_gene_heatmap.py*


## Output Files

The prediction accuracy and error will be printed on the screen. 
All trained models will be stored in **/data/** in TensorFlow SavedModel format.

# [Authorship Identification using Recurrent Neural Networks](https://dl.acm.org/doi/abs/10.1145/3325917.3325935)
[Shriya T.P. Gupta](https://scholar.google.com/citations?user=3RRL_0QAAAAJ&hl=en&oi=ao), [Jajati K. Sahoo](https://scholar.google.co.in/citations?user=luGgl5EAAAAJ&hl=en&oi=ao) and [Rajendra K. Roul](https://scholar.google.co.in/citations?user=uWs8xfwAAAAJ&hl=en&oi=ao)

This repository contains code for the models presented in the [paper](https://dl.acm.org/doi/abs/10.1145/3325917.3325935).

This code base is built using the [Tensorflow](https://www.tensorflow.org/) library. The auth_identification.ipynb file is also provided in the form of a jupyter notebook which can be run on [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb).

The RNN_average_model.py provides the Sentence Level Module whereas the RNN_sent_model.py provides the code for the Article Level Networks with the corresponding GRU and LSTM models in the proj_gru_cell.py and proj_rnn_cell.py files respectively.

The experiments have been carried out as described in the paper for the following datasets:

1) The Reuters C50 dataset
2) BBC New classification dataset

## Citation

If you find this project useful in your research, please consider citing:

```
@inproceedings{gupta2019authorship,
  title={Authorship Identification using Recurrent Neural Networks},
  author={Gupta, Shriya TP and Sahoo, Jajati Keshari and Roul, Rajendra Kumar},
  booktitle={Proceedings of the 2019 3rd International Conference on Information System and Data Mining},
  pages={133--137},
  year={2019}
}
```

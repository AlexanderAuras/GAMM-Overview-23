# Robustness and Exploration of Variational and Machine Learning Approaches to Inverse Problems: An Overview
<font size="3">Alexander Auras<sup>1</sup>, Kanchana Vaishnavi Gandikota<sup>1</sup>, Hannah Droege<sup>2</sup>and Michael Moeller<sup>1</sup></font>

<font size="2">
<sup>1</sup>Institute for Vision and Graphics, University of Siegen<br/>
<sup>2</sup>Bonn?<br/>
</font><br/>

Official implementation of *Robustness and Exploration of Variational and Machine Learning Approaches to Inverse Problems: An Overview*.

Based off of the paper ["Solving Inverse Problems With Deep Neural Networks - Robustness Included?"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9705105) by M. Genzel, J. Macdonald, and M. März. <br/>
For their implementation see [https://github.com/jmaces/robust-nets](https://github.com/jmaces/robust-nets).

<!--[![arXiv](https://img.shields.io/badge/arXiv-0000.00000-b31b1b.svg)](https://arxiv.org/abs/0000.00000)-->

## Installation

    git clone https://github.com/AlexanderAuras/GAMM-Overview-23.git
    cd GAMM-Overview-23
    pip install .

Then simply run the jupyter notebook.

The used operator, data and model weights are available at [https://drive.google.com/drive/folders/1nL_Z6gKyRRp36E58KUwNSZbO7Yj58chL?usp=drive_link](https://drive.google.com/drive/folders/1nL_Z6gKyRRp36E58KUwNSZbO7Yj58chL?usp=drive_link).<br/>
The downloaded files location must be specified in the jupyter notebook. Alternatively you can use the *generate_op.py* and *generate_data.py* scripts (in *src/gamm23/operators* and *src/gamm23/data*) to generate a new operator/new data and train the models using the *config.yaml* file and the *train.py* script.
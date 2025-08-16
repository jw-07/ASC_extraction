# ASC_extraction

 Our paper, "Efficient Attributed Scattering Center Extraction Using Gradient-Based Optimization for SAR Image", has been accepted for publication in **IEEE Antennas and Wireless Propagation Letters (AWPL)**.

This repository contains the official PyTorch implementation of our work. 

## Overview
![Flowchart of our method](fig/GBASC.png)

We introduce a fully gradient-based optimization Attributed Scattering Centers (ASCs) extrection method. By leveraging a novel **joint image-frequency domain initialization** and using gradient-based optimization in PyTorch, 
our approach significantly enhances the extraction speed. For the T72 target, the estimation of 40 ASCs can be completed in approximately 10 seconds.



## Installation

 **Install dependencies:**
 > * PyTorch == 1.9.1 （supports complex number operations.）
 > * scipy
 > * hdf5storage



## Usage

1.  **Data Preparation:**
    Place your SAR echo data files into the `./data/` directory. The input should be `.mat` files.

    For the MSTAR/SAMPLE datasets, echo signals should be obtained by performing zero-padding removal and Taylor window processing, and stored in variable `specRmZeroAndWin`.

2.  **Run Extraction:**
    Execute the main script to start the ASC extraction process.
    ```bash
    python ASC_extract_multi_file_loop_zh.py
    ```
 3.  **Result:**
    The extracted results are saved in the `./result/` directory.  


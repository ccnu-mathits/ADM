This folder contains the code related to the real-world dataset (Lomusity) application in the paper.

####Dependent packages
The main dependence packages of the code include _numpy_, _pytorch_, _pandas_, _pysr_ with their specific versions as follows:  
`numpy==1.23.3`  
`pytorch==1.10.0`  
`pandas==1.5.0`  
`pysr==0.11.2`  

####Runs
In order to replicate the Lomusity experiments described in this paper, first open the terminal and navigate to the directory where the code is located. Afterwards, run "`python main.py`" in the terminal.

During the execution,  
the generated processed data will be saved in the `/ProcessedData` folder.   
The optimized parameters of the deep learning regressor will be saved in the `/SavedModels` folder.  
The estimated learner's cognitive state and behavior encoding data by the deep learning regressor will be saved in the `/StateMLPData` folder.  
The symbolic regression-generated regular equations will be saved in the `/PYSRres` folder.

####Experimental parameters
All parameters can be modified in the main function of the `main.py` file. Once the modifications are completed, simply run the file to begin the experiment. Please refer to the relevant comments in the `main.py` file for instructions on how to modify specific parameters.
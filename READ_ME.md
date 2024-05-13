# Training

Script python to train the model.


## Installation

Python 3.7.* is necessary to launch the python script.      
It is necessary to install all python packages listed in requirements.txt.       
Quick install: pip install -r requirements.txt        


## Execution

To execute the script, please follow these steps:

1. Navigate to the root directory 
2. Locate the script named *train.py*
3. Run the script using the following command: python3 [script_name].py [optional_parameters]
4. Optionally, you can pass some parameters to set all training options. See next section for details

Ensure you have Python installed and that the necessary dependencies are met before running the script.

### Usage

<pre>
usage: train.py [-h]        
                [--data_path DATA_PATH]          
                [--split SPLIT]              
                [--transform TRANSFORM]          
                [--normalize]       
                [--mean MEAN [MEAN ...]] 
                [--std STD [STD ...]] 
                [--resize] 
                [--height HEIGHT] 
                [--width WIDTH] 
                [--one_hot_encoding] 
                [--name NAME] 
                [--format FORMAT] 
                [--buffer_size BUFFER_SIZE] 
                [--batch_size BATCH_SIZE] 
                [--model_path MODEL_PATH] 
                [--domain_randomization] 
                [--dom_rand__mode DOM_RAND__MODE] 
                [--dom_rand__seed DOM_RAND__SEED] 
                [--dom_rand__factors DOM_RAND__FACTORS [DOM_RAND__FACTORS ...]] 
                [--dom_rand__dist_pars DOM_RAND__DIST_PARS [DOM_RAND__DIST_PARS ...]] 
                [--dom_rand__dist_pars___uniform DOM_RAND__DIST_PARS___UNIFORM [DOM_RAND__DIST_PARS___UNIFORM ...]] 
                [--dom_rand__dist_pars___triangular DOM_RAND__DIST_PARS___TRIANGULAR [DOM_RAND__DIST_PARS___TRIANGULAR ...]] 
                [--dom_rand__dist_pars___univariatenormal DOM_RAND__DIST_PARS___UNIVARIATENORMAL [DOM_RAND__DIST_PARS___UNIVARIATENORMAL ...]] 
                [--dom_rand__dist_pars___multivariatenormal DOM_RAND__DIST_PARS___MULTIVARIATENORMAL [DOM_RAND__DIST_PARS___MULTIVARIATENORMAL ...]] 
                [--adaptive_dom_rand] 
                [--adaptive_dom_rand__dist_ranges____low ADAPTIVE_DOM_RAND__DIST_RANGES____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES____LOW ...]] 
                [--adaptive_dom_rand__dist_ranges____up ADAPTIVE_DOM_RAND__DIST_RANGES____UP [ADAPTIVE_DOM_RAND__DIST_RANGES____UP ...]] 
                [--adaptive_dom_rand__dist_ranges___uniform____low ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____LOW ...]] 
                [--adaptive_dom_rand__dist_ranges___uniform____up ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____UP ...]] 
                [--adaptive_dom_rand__dist_ranges___triangular____low ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____LOW ...]] 
                [--adaptive_dom_rand__dist_ranges___triangular____up ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____UP ...]] 
                [--adaptive_dom_rand__dist_ranges___univariatenormal____low ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____LOW ...]] 
                [--adaptive_dom_rand__dist_ranges___univariatenormal____up ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____UP ...]] 
                [--adaptive_dom_rand__dist_ranges___multivariatenormal____low ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____LOW ...]] 
                [--adaptive_dom_rand__dist_ranges___multivariatenormal____up ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____UP ...]] 
                [--adaptive_dom_rand__dist_initials ADAPTIVE_DOM_RAND__DIST_INITIALS [ADAPTIVE_DOM_RAND__DIST_INITIALS ...]] 
                [--adaptive_dom_rand__dist_initials___uniform ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIFORM [ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIFORM ...]] 
                [--adaptive_dom_rand__dist_initials___triangular ADAPTIVE_DOM_RAND__DIST_INITIALS___TRIANGULAR [ADAPTIVE_DOM_RAND__DIST_INITIALS___TRIANGULAR ...]] 
                [--adaptive_dom_rand__dist_initials___univariatenormal ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIVARIATENORMAL [ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIVARIATENORMAL ...]] 
                [--adaptive_dom_rand__dist_initials___multivariatenormal ADAPTIVE_DOM_RAND__DIST_INITIALS___MULTIVARIATENORMAL [ADAPTIVE_DOM_RAND__DIST_INITIALS___MULTIVARIATENORMAL ...]] 
                [--neural_network NEURAL_NETWORK] 
                [--gradient_based__optimizer GRADIENT_BASED__OPTIMIZER] 
                [--gradient_based__optimizer___learning_rate GRADIENT_BASED__OPTIMIZER___LEARNING_RATE] 
                [--gradient_based__loss GRADIENT_BASED__LOSS] 
                [--epochs EPOCHS] 
                [--gradient_free__optimizer GRADIENT_FREE__OPTIMIZER]
</pre>

train.py

optional arguments:

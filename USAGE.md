# Training

Script python to train the model.


## Installation

Python $3.7.*$ is necessary to launch the python script.      
It is necessary to install all python packages listed in requirements.txt.       
Quick install: 
<pre>  
pip install -r requirements.txt        
</pre>

## Execution

To execute the script, please follow these steps:

1. Navigate to the root directory 
2. Locate the script named *train.py*
3. Run the script using the following command: python3 [script_name].py [optional_parameters]
4. Optionally, you can pass some parameters to set all training options. See <a href="#Usage">next section</a> for details.

Ensure you have Python installed and that the necessary dependencies are met before running the script.

## <span id="Usage">Usage</span>


<pre>   
       train.py [-h]        
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

<pre>
  -h, --help            
                        Show this help message and exit
  --data_path DATA_PATH
                        Path of the directory in which input data are stored.
                        (default: 'Input\dataset')
  --split SPLIT         
                        Percentage to split the dataset into train/validation sets. 
                        It can be a float between 0 and 1 (representing the percentage), or 'auto' (to use 
                        the standard split: 0.2) or 'train only' (to use all the dataset for training. Use 
                        this option if only training is needed).
                        (default: 'train only')
  --transform TRANSFORM
                        Transformations to be applied to images of the dataset. 
                        (default: None, i.e.: no transformations are applied.)
  --normalize           
                        Normalize option to be applied to images of the dataset. (default: True)
  --mean MEAN [MEAN ...]
                        Mean values for normalization. 
                        They can be multiple floats separated by a spaces (list representing the values, 
                        whose cardinality depends on the image format) or 'auto' (to apply standard values)
                        or None (no normalization).
                        Standard values. See the <a href="#Supported-Color-Modes-and-Standard-Normalization-Parameters">section</a>.
                        (default: 'auto')
  --std STD [STD ...]   
                        Standard deviation values for normalization. 
                        They can be multiple floats separated by a spaces (list representing the values, 
                        whose cardinality depends on the image format) or 'auto' (to apply standard values)
                        or None (no standard deviation, i.e.: sd = 0).
                        Standard values. See the <a href="#Supported-Color-Modes-and-Standard-Normalization-Parameters">section</a>.
                        (default: None)
  --resize              
                        Resize option to be applied to images of the dataset.
                        (default: True)
  --height HEIGHT       
                        Height value for resize. It can be a float or 'auto' (to apply standard values) or 
                        None (no resize).
                        Standard values. See the <a href="#Standard-Resize-Parameters">section</a>.
                        (default: 'auto')
  --width WIDTH         
                        Width value for resize. It can be a float or 'auto' (to apply standard values) or 
                        None (no resize).
                        Standard values. See the <a href="#Standard-Resize-Parameters">section</a>.
                        (default: 'auto')
  --one_hot_encoding    
                        One-hot encoding option to be applied to labels.
                        (default: True)
  --name NAME           
                        Name of the dataset.
                        (default: 'Electronic components dataset')
  --format FORMAT       
                        Image format (e.g.: RGB, Grayscale, RGBA). 
                        If not present, automatically deduced from images. 
                        Images of homegeneus formats are suggested to improve algorithm performances. 
                        (default: None)
  --buffer_size BUFFER_SIZE
                        Buffer size.
                        (default:500)
  --batch_size BATCH_SIZE
                        Batch size. 
                        Also possibile to use 'no batches'.
                        (default:16)
  --model_path MODEL_PATH
                        Path of the directory in which the trained model will be saved. 
                        (default: 'out\model')
  --domain_randomization
                        Enable image parameters domain randomization. 
                        Parameters. See the <a href="#Parameters-names">section</a>.
                        (default: True)
  --dom_rand__mode DOM_RAND__MODE
                        Domain randomization mode. 
                        Distributions. See the <a href="#Distributions">section</a>.
                        (default: 'multivariate normal')
  --dom_rand__seed DOM_RAND__SEED
                        Domain randomization random seed. 
                        (default: None)
  --dom_rand__factors DOM_RAND__FACTORS [DOM_RAND__FACTORS ...]
                        Domain randomization factor parameters. 
                        Multiple floats separated by a space (list representing the values, whose 
                        cardinality depends on the number of parameters). 
                        If None, standard factors are used.
                        Standard values. See the <a href="#Domain-Randomization-Standard-Factor-Parameters">section</a>.
                        (default: None)
  --dom_rand__dist_pars DOM_RAND__DIST_PARS [DOM_RAND__DIST_PARS ...]
                        Domain randomization distribution parameters. 
                        Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
                        If None, standard distributions parameters are used. 
                        Standard distributions parameters. See the <a href="#Domain-Randomization-Standard-Distribution-Parameters">section</a>.
                        (default: None)
  --dom_rand__dist_pars___uniform DOM_RAND__DIST_PARS___UNIFORM [DOM_RAND__DIST_PARS___UNIFORM ...]
                        Domain randomization uniform distribution parameters. 
                        Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
                        If None, standard distributions parameters are used. 
                        Standard distributions parameters. See the <a href="#Domain-Randomization-Standard-Distribution-Parameters">section</a>.
                        (default: None)
  --dom_rand__dist_pars___triangular DOM_RAND__DIST_PARS___TRIANGULAR [DOM_RAND__DIST_PARS___TRIANGULAR ...]
                        Domain randomization triangular distribution parameters. 
                        Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
                        If None, standard distributions parameters are used. 
                        Standard distributions parameters. See the <a href="#Domain-Randomization-Standard-Distribution-Parameters">section</a>.
                        (default: None)
  --dom_rand__dist_pars___univariatenormal DOM_RAND__DIST_PARS___UNIVARIATENORMAL [DOM_RAND__DIST_PARS___UNIVARIATENORMAL ...]
                        Domain randomization univariate normal distribution parameters. 
                        Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
                        If None, standard distributions parameters are used. 
                        Standard distributions parameters. See the <a href="#Domain-Randomization-Standard-Distribution-Parameters">section</a>.
                        (default: None)
  --dom_rand__dist_pars___multivariatenormal DOM_RAND__DIST_PARS___MULTIVARIATENORMAL [DOM_RAND__DIST_PARS___MULTIVARIATENORMAL ...]
                        Domain randomization multivariate normal distribution parameters. 
                        Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
                        If None, standard distributions parameters are used. 
                        Standard distributions parameters. See the <a href="#Domain-Randomization-Standard-Distribution-Parameters">section</a>.
                        (default: None)
  --adaptive_dom_rand   
			Optimize domain randomization distribution parameters.
                        (default: False)
  --adaptive_dom_rand__dist_ranges____low ADAPTIVE_DOM_RAND__DIST_RANGES____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES____LOW ...]
                        Adaptive domain randomization distribution parameters low ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Low-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges____up ADAPTIVE_DOM_RAND__DIST_RANGES____UP [ADAPTIVE_DOM_RAND__DIST_RANGES____UP ...]
                        Adaptive domain randomization distribution parameters up ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters up ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Up-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges___uniform____low ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____LOW ...]
                        Adaptive domain randomization uniform distribution parameters low ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Low-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges___uniform____up ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____UP ...]
                        Adaptive domain randomization uniform distribution parameters up ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters up ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Up-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges___triangular____low ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____LOW ...]
                        Adaptive domain randomization triangular distribution parameters low ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Low-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges___triangular____up ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____UP ...]
                        Adaptive domain randomization triangular distribution parameters up ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters up ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Up-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges___univariatenormal____low ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____LOW ...]
                        Adaptive domain randomization univariate normal distribution parameters low ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Low-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges___univariatenormal____up ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____UP ...]
                        Adaptive domain randomization univariate normal distribution parameters up ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters up ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Up-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges___multivariatenormal____low ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____LOW ...]
                        Adaptive domain randomization distribution parameters low ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Low-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_ranges___multivariatenormal____up ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____UP ...]
                        Adaptive domain randomization multivariate normal distribution parameters up ranges. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters up ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Up-Ranges">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_initials ADAPTIVE_DOM_RAND__DIST_INITIALS [ADAPTIVE_DOM_RAND__DIST_INITIALS ...]
                        Adaptive domain randomization distribution parameters initial values. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Initial-Values">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_initials___uniform ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIFORM [ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIFORM ...]
                        Adaptive domain randomization uniform distribution parameters initial values. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Initial-Values">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_initials___triangular ADAPTIVE_DOM_RAND__DIST_INITIALS___TRIANGULAR [ADAPTIVE_DOM_RAND__DIST_INITIALS___TRIANGULAR ...]
                        Adaptive domain randomization triangular distribution parameters initial values. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Initial-Values">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_initials___univariatenormal ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIVARIATENORMAL [ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIVARIATENORMAL ...]
                        Adaptive domain randomization univariate normal distribution parameters initial values. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Initial-Values">section</a>.
                        (default: None)
  --adaptive_dom_rand__dist_initials___multivariatenormal ADAPTIVE_DOM_RAND__DIST_INITIALS___MULTIVARIATENORMAL [ADAPTIVE_DOM_RAND__DIST_INITIALS___MULTIVARIATENORMAL ...]
                        Adaptive domain randomization multivariate normal distribution parameters initial values. 
			Use the linearized version, i.e. Parameters list, whose cardinality depends 
                        on the number of parameters: single parameters separated by 1 single space. 
                        A single parameter can be a scalar, a vector or a a matrix. 
                        Inside a vector, elements are separated by 1 single space. 
                        Inside a matrix, rows are representd as a (transpose) vector and inserted separated
                        by 1 single space. 
			If None, standard distribution parameters ranges are used. 
			Standard distribution parameters low ranges. See the <a href="#Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Initial-Values">section</a>.
                        (default: None)
  --neural_network NEURAL_NETWORK
                        Neural Network to be used. 
			Possible choices. See the <a href="#Neural-Networks">section</a>.
			(default: 'ResNet 2.0')
  --gradient_based__optimizer GRADIENT_BASED__OPTIMIZER
                        Optimizer to be used for gradient based optimization.
                        Possible choices. See the <a href="#Gradient-Based-Optimizers">section</a>.
			(default: 'Adam')
  --gradient_based__optimizer___learning_rate GRADIENT_BASED__OPTIMIZER___LEARNING_RATE
                        Learning rate of the optimizer to be used for gradient based optimization. 
			It has to be a float. 
			(default: 1e-03)
  --gradient_based__loss GRADIENT_BASED__LOSS
                        Loss to be used for gradient based optimization.
                        Possible choices. See the <a href="#Gradient-Based-Optimizers-Losses">section</a>. 
			(default: 'CategoricalCrossentropy')
  --epochs EPOCHS       
			Training epochs. 
			It has to be an int. 
			(default: 1000)
  --gradient_free__optimizer GRADIENT_FREE__OPTIMIZER
                        Optimizer to be used for gradient free optimization.
			Possible choices. See the <a href="#Gradient-Free-Optimizers">section</a>.
			(default: 'CMA')
</pre>


### Standard values

#### <span id="Supported-Color-Modes-and-Standard-Normalization-Parameters">Supported Color Modes and Standard Normalization Parameters</span>


1. RGB:
    - Mean: [0.00392157, 0.00392157, 0.00392157]
    - Standard Deviation (Std): [0.229, 0.224, 0.225]

2. Grayscale:
    - Mean: [0.5]
    - Std: [0.5]

3. RGBA:
    - Mean: [0.485, 0.456, 0.406, 0.0]
    - Std: [0.229, 0.224, 0.225, 1.0]

Note: 
- The mean and standard deviation (std) values are used for normalization in the respective color modes.
- RGB and RGBA mean values are scaled to the range [0, 1] by dividing by 255.


#### <span id="Standard-Resize-Parameters">Standard Resize Parameters</span>

- Height: 128
- Width: 128

#### <span id="Domain-Randomization-Standard-Factor-Parameters">Domain Randomization, Standard Factor Parameters</span>

Probability that for a single (batch of) image the corresponding parameter is randomized:
- brightness, 0.9
- contrast, 0.9
- horizontally flip, 0.9
- vertically flip, 0.9
- hue, 0.9
- jpeg quality, 0.9
- saturation, 0.9

#### <span id="Domain-Randomization-Standard-Distribution-Parameters">Domain Randomization, Standard Distribution Parameters</span>

The meaning of the parameter depends on the distribution(s) considered.  
If univariate distributions are used, the order reflects the corresponding image parameter to be randomized:
- Univariate uniform distribution,   
    $$\mathcal{U}(\text{{lower brightness}}, \text{{upper brightness}});\quad  \mathcal{U}(\text{{lower contrast}}, \text{{upper contrast}});\quad  \mathcal{U}(\text{{lower horizontal filp}}, \text{{upper horizontal flip}});\quad  \mathcal{U}(\text{{lower vertical flip}}, \text{{upper vertical flip}});  \mathcal{U}(\text{{lower hue}}, \text{{upper hue}});\quad  \mathcal{U}(\text{{lower jpeg quality}}, \text{{upper jpeg quality}});\quad  \mathcal{U}(\text{{lower saturation}}, \text{{upper saturation}})$$
    standard distribution parameters,
  $$\qquad \mathcal{U}(-0.2, 0.2);\quad  \mathcal{U}(0, 2.5);\quad  \mathcal{U}(0, 1);\quad  \mathcal{U}(0, 1);\quad  \mathcal{U}(-0.2, 0.2);\quad  \mathcal{U}(20, 100);\quad  \mathcal{U}(0, 2)$$
	- Linearized version:
	<pre>
	Given lowers=[-0.2, 0, 0, 0, -0.2, 20, 0], uppers=[0.2, 2.5, 1, 1, 0.2, 100, 2] -> -0.2 0 0 0 -0.2 20 0 0.2 2.5 1 1 0.2 100 2
	</pre>
  &nbsp;
- Univariate triangular distribution,
    $$\mathcal{T}(\text{{lower brightness}}, \text{{mode brightness}}, \text{{upper brightness}});\quad  \mathcal{T}(\text{{lower contrast}}, \text{{mode contrast}}, \text{{upper contrast}});\quad  \mathcal{T}(\text{{lower horizontal filp}}, \text{{mode horizontal flip}}, \text{{upper horizontal flip}});\quad  \mathcal{T}(\text{{lower vertical flip}}, \text{{mode vertical flip}}, \text{{upper vertical flip}}); \quad \mathcal{T}(\text{{lower hue}}, \text{{mode hue}}, \text{{upper hue}});\quad  \mathcal{T}(\text{{lower jpeg quality}}, \text{{mode jpeg quality}}, \text{{upper jpeg quality}});\quad  \mathcal{T}(\text{{lower saturation}}, \text{{mode saturation}}, \text{{upper saturation}})$$
    standard distribution parameters,
  $$\qquad \mathcal{T}(-0.2, 0, 0.2) ;\quad \mathcal{T}(0, 1.25, 2.5) ;\quad \mathcal{T}(0, 0.5, 1) ;\quad \mathcal{T}(0, 0.5, 1) ;\quad \mathcal{T}(-0.2, 0, 0.2) ;\quad \mathcal{T}(20, 60, 100) ;\quad \mathcal{T}(0, 1, 2)$$
	- Linearized version:
	<pre>
	Given lowers=[-0.2, 0, 0, 0, -0.2, 20, 0], modes=[0, 1.25, 0.5, 0.5, 0, 60, 1], uppers=[0.2, 2.5, 1, 1, 0.2, 100, 2] -> -0.2 0 0 0 -0.2 20 0 0 1.25 0.5 0.5 0 60 1 0.2 2.5 1 1 0.2 100 2
	</pre>
  &nbsp;
- Univariate normal distribution,
        $$\mathcal{N}(\text{{mean brightness}}, \text{{variance brightness}});\quad  \mathcal{N}(\text{{mean contrast}}, \text{{variance contrast}});\quad  \mathcal{N}(\text{{mean horizontal flip}}, \text{{variance horizontal flip}});\quad  \mathcal{N}(\text{{mean vertical flip}}, \text{{variance vertical flip}});  \mathcal{N}(\text{{mean hue}}, \text{{variance hue}});\quad  \mathcal{N}(\text{{mean jpeg quality}}, \text{{variance jpeg quality}});\quad  \mathcal{N}(\text{{mean saturation}}, \text{{variance saturation}})$$
standard distribution parameters,
  $\qquad \mathcal{N}(0, 0.15);\quad  \mathcal{N}(1.25, 1);\quad  \mathcal{N}(0.5, 0.1);\quad  \mathcal{N}(0.5, 0.1);\quad  \mathcal{N}(0, 0.15);\quad  \mathcal{N}(60, 25);\quad  \mathcal{N}(1.25, 1.125)$
	- Linearized version:
	<pre>
	Given means=[0, 1.25, 0.5, 0.5, 0, 60, 1.25], variances=[0.15, 1, 0.1, 0.1, 0.15, 25, 1.125] -> 0 1.25 0.5 0.5 0 60 1.25 0.15 1 0.1 0.1 0.15 25 1.125
	</pre>
 &nbsp;       
- Multivariate normal distribution,
	 
	```math
		\mathcal{N}_7(\begin{bmatrix}\mu_\text{brightness}\\\mu_\text{contrast}\\\mu_\text{horizontal flip}\\\mu_\text{vertical flip}\\\mu_\text{hue}\\\mu_\text{jpeg quality}\\\mu_\text{saturation}\end{bmatrix}, \\
		\begin{bmatrix}
		\sigma^2_{\text{brightness}} & \sigma_{\text{brightness}, \text{contrast}} & \cdots & \sigma_{\text{brightness}, \text{saturation}} \\
		\sigma_{\text{contrast}, \text{brightness}} & \sigma^2_{\text{contrast}} & \cdots & \sigma_{\text{contrast}, \text{saturation}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\sigma_{\text{saturation}, \text{brightness}} & \sigma_{\text{saturation}, \text{contrast}} & \cdots & \sigma^2_{\text{saturation}}
		\end{bmatrix}) \\,
 	```

	```math
 		\mu: \text{mean}, \quad \sigma^2: \text{variance} \quad and \quad \sigma: \text{covariance}
 	```

	standard distribution parameters,

 	```math
		\mathcal{N}_7(\begin{bmatrix}
		0 \\
		1.25 \\
		0.5 \\
		0.5 \\
		0 \\
		60 \\
		1.25
		\end{bmatrix}, \\
		\begin{bmatrix}
		0.15 & 0 & 0 & 0 & 0 & 0 & 0 \\
		0 & 1 & 0 & 0 & 0 & 0 & 0 \\
		0 & 0 & 0.1 & 0 & 0 & 0 & 0 \\
		0 & 0 & 0 & 0.1 & 0 & 0 & 0 \\
		0 & 0 & 0 & 0 & 0.15 & 0 & 0 \\
		0 & 0 & 0 & 0 & 0 & 25 & 0 \\
		0 & 0 & 0 & 0 & 0 & 0 & 1.125
		\end{bmatrix})
	```

	- Linearized version
	<pre>
	Given mean_vector=[0, 1.25, 0.5, 0.5, 0, 60, 1.25], variance_covariance_matrix=[[0.15, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0.1, 0, 0, 0, 0], [0, 0, 0, 0.1, 0, 0, 0], [0, 0, 0, 0, 0.15, 0, 0], [0, 0, 0, 0, 0, 25, 0], [0, 0, 0, 0, 0, 0, 1.125]] -> 0.15, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 1.125
	</pre>

#### <span id="Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Low-Ranges">Adaptive Domain Randomization, Standard Distribution Parameters Low Ranges</span>

The meaning of the parameter depends on the distribution(s) considered.  
If univariate distributions are used, the order reflects the corresponding image parameter to be randomized:
- Univariate uniform distribution,   
    $$\mathcal{U}(\text{{lower brightness}}, \text{{upper brightness}});\quad  \mathcal{U}(\text{{lower contrast}}, \text{{upper contrast}});\quad  \mathcal{U}(\text{{lower horizontal filp}}, \text{{upper horizontal flip}});\quad  \mathcal{U}(\text{{lower vertical flip}}, \text{{upper vertical flip}});  \mathcal{U}(\text{{lower hue}}, \text{{upper hue}});\quad  \mathcal{U}(\text{{lower jpeg quality}}, \text{{upper jpeg quality}});\quad  \mathcal{U}(\text{{lower saturation}}, \text{{upper saturation}})$$
    standard distribution parameters low ranges,
    $$\qquad \mathcal{U}(a, b),\text{lower range for }a = -1, \text{lower range for }b = -1;\quad  \mathcal{U}(a, b),\text{lower range for }a = -\infty, \text{lower range for }b = -\infty;\quad  \mathcal{U}(a, b),\text{lower range for }a = -\infty, \text{lower range for }b = -\infty;\quad  \mathcal{U}(a, b),\text{lower range for }a = -\infty, \text{lower range for }b = -\infty;\quad  \mathcal{U}(a, b),\text{lower range for }a = -1, \text{lower range for }b = -1;\quad  \mathcal{U}(a, b),\text{lower range for }a = 0, \text{lower range for }b = 0;\quad  \mathcal{U}(a, b),\text{lower range for }a = -0, \text{lower range for }b = 0$$
	- Linearized version:
	<pre>
	Given lowers=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], uppers=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1 float(-inf) float(-inf) float(-inf) -1 0 0
	</pre>
  &nbsp;
- Univariate triangular distribution,
    $$\mathcal{T}(\text{{lower brightness}}, \text{{mode brightness}}, \text{{upper brightness}});\quad  \mathcal{T}(\text{{lower contrast}}, \text{{mode contrast}}, \text{{upper contrast}});\quad  \mathcal{T}(\text{{lower horizontal filp}}, \text{{mode horizontal flip}}, \text{{upper horizontal flip}});\quad  \mathcal{T}(\text{{lower vertical flip}}, \text{{mode vertical flip}}, \text{{upper vertical flip}}); \quad \mathcal{T}(\text{{lower hue}}, \text{{mode hue}}, \text{{upper hue}});\quad  \mathcal{T}(\text{{lower jpeg quality}}, \text{{mode jpeg quality}}, \text{{upper jpeg quality}});\quad  \mathcal{T}(\text{{lower saturation}}, \text{{mode saturation}}, \text{{upper saturation}})$$
    standard distribution parameters low ranges,
        $$\qquad \mathcal{T}(a, m, b),\text{lower range for }a = -1, \text{lower range for }m = -1, \text{lower range for }b = -1;\quad  \mathcal{T}(a, m, b),\text{lower range for }a = -\infty, \text{lower range for }m = -\infty, \text{lower range for }b = -\infty;\quad  \mathcal{T}(a, m, b),\text{lower range for }a = -\infty, \text{lower range for }m = -\infty, \text{lower range for }b = -\infty;\quad  \mathcal{T}(a, m, b),\text{lower range for }a = -\infty, \text{lower range for }m = -\infty, \text{lower range for }b = -\infty;\quad  \mathcal{T}(a, m, b),\text{lower range for }a = -1, \text{lower range for }m = -1, \text{lower range for }b = -1;\quad  \mathcal{T}(a, m, b),\text{lower range for }a = 0, \text{lower range for }m = 0, \text{lower range for }b = 0;\quad  \mathcal{T}(a, m, b),\text{lower range for }a = -0, \text{lower range for }m = -0, \text{lower range for }b = 0$$
	- Linearized version:
	<pre>
	Given lowers=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], modes=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], uppers=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1 float(-inf) float(-inf) float(-inf) -1 0 0
	</pre>
  &nbsp;
- Univariate normal distribution,
        $$\mathcal{N}(\text{{mean brightness}}, \text{{variance brightness}});\quad  \mathcal{N}(\text{{mean contrast}}, \text{{variance contrast}});\quad  \mathcal{N}(\text{{mean horizontal flip}}, \text{{variance horizontal flip}});\quad  \mathcal{N}(\text{{mean vertical flip}}, \text{{variance vertical flip}});  \mathcal{N}(\text{{mean hue}}, \text{{variance hue}});\quad  \mathcal{N}(\text{{mean jpeg quality}}, \text{{variance jpeg quality}});\quad  \mathcal{N}(\text{{mean saturation}}, \text{{variance saturation}})$$
    standard distribution parameters low ranges,
    $$\qquad \mathcal{N}(\mu, \sigma^2),\text{lower range for }\mu = -1, \text{lower range for }\sigma^2 = 0;\quad  \mathcal{N}(\mu, \sigma^2),\text{lower range for }\mu = -\infty, \text{lower range for }\sigma^2 = 0;\quad  \mathcal{N}(\mu, \sigma^2),\text{lower range for }\mu = -\infty, \text{lower range for }\sigma^2 = 0;\quad  \mathcal{N}(\mu, \sigma^2),\text{lower range for }\mu = -\infty, \text{lower range for }\sigma^2 = 0;\quad  \mathcal{N}(\mu, \sigma^2),\text{lower range for }\mu = -1, \text{lower range for }\sigma^2 = 0;\quad  \mathcal{N}(\mu, \sigma^2),\text{lower range for }\mu = 0, \text{lower range for }\sigma^2 = 0;\quad  \mathcal{N}(\mu, \sigma^2),\text{lower range for }\mu = 0, \text{lower range for }\sigma^2 = 0$$
	- Linearized version:
	<pre>
	Given means=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], variances=[0, 0, 0, 0, 0, 0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0 0 0 0 0 0 0 0
	</pre>
 &nbsp;       
- Multivariate normal distribution,
	 
	```math
		\mathcal{N}_7(\begin{bmatrix}\mu_\text{brightness}\\\mu_\text{contrast}\\\mu_\text{horizontal flip}\\\mu_\text{vertical flip}\\\mu_\text{hue}\\\mu_\text{jpeg quality}\\\mu_\text{saturation}\end{bmatrix}, \\
		\begin{bmatrix}
		\sigma^2_{\text{brightness}} & \sigma_{\text{brightness}, \text{contrast}} & \cdots & \sigma_{\text{brightness}, \text{saturation}} \\
		\sigma_{\text{contrast}, \text{brightness}} & \sigma^2_{\text{contrast}} & \cdots & \sigma_{\text{contrast}, \text{saturation}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\sigma_{\text{saturation}, \text{brightness}} & \sigma_{\text{saturation}, \text{contrast}} & \cdots & \sigma^2_{\text{saturation}}
		\end{bmatrix}) \\,
 	```

	```math
 		\mu: \text{mean}, \quad \sigma^2: \text{variance} \quad and \quad \sigma: \text{covariance}
 	```

	standard distribution parameters low ranges,

 	```math
		\mathcal{N}_7(\begin{bmatrix}
		\mu_1 \\
		\mu_2 \\
		\mu_3 \\
		\mu_4 \\
		\mu_5 \\
		\mu_6 \\
		\mu_7
		\end{bmatrix}, \\
		\begin{bmatrix}
		\sigma^2_1 & \sigma_{12} & \sigma_{13} & \sigma_{14} & \sigma_{15} & \sigma_{16} & \sigma_{17} \\
		\sigma_{21} & \sigma^2_2 & \sigma_{23} & \sigma_{24} & \sigma_{25} & \sigma_{26} & \sigma_{27} \\
		\sigma_{31} & \sigma_{32} & \sigma^2_3 & \sigma_{34} & \sigma_{35} & \sigma_{36} & \sigma_{37} \\
		\sigma_{41} & \sigma_{42} & \sigma_{43} & \sigma^2_4 & \sigma_{45} & \sigma_{46} & \sigma_{47} \\
		\sigma_{51} & \sigma_{52} & \sigma_{53} & \sigma_{54} & \sigma^2_5 & \sigma_{56} & \sigma_{57} \\
		\sigma_{61} & \sigma_{62} & \sigma_{63} & \sigma_{64} & \sigma_{65} & \sigma^2_6 & \sigma_{67} \\
		\sigma_{71} & \sigma_{72} & \sigma_{73} & \sigma_{74} & \sigma_{75} & \sigma_{76} & \sigma^2_7
		\end{bmatrix}),
	```

	```math
		\text{lower range for }\mu_1 = -1,
 	```
 
 	```math
  		\text{lower range for }\mu_2 = -\infty, 
  	```

  	```math
  		\text{lower range for }\mu_3 = -\infty, 
  	```

   	```math
  		\text{lower range for }\mu_4 = -\infty, 
  	```

   	```math
  		\text{lower range for }\mu_5 = -1, 
  	```

   	```math
  		\text{lower range for }\mu_6 = 0, 
  	```

   	```math
  		\text{lower range for }\mu_7 = 0,
  	```

	```math
  		\text{lower range for }\sigma^2_i = 0, i \in \{1, 2, 3, 4, 5, 6, 7\},
  	```

	```math
  		\text{lower range for }\sigma_{ij} = -100, i, j \in \{1, 2, 3, 4, 5, 6, 7\} \quad \text{s.t.} \quad i \neq j
  	```

	- Linearized version
   		(note that the variance covariance matrix is a _positive definite matrix_: only elements of diagonal and elements below diagonal has to be specified):
	<pre>
	Given mean_vector=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], variance_covariance_matrix=[[0], [-100, 0], [-100, -100, 0], [-100, -100, -100, 0], [-100, -100, -100, -100, 0], [-100, -100, -100, -100, -100, 0], [-100, -100, -100, -100, -100, -100, 0]] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0 0 -100 0 -100 -100 0 -100 -100 -100 0 -100 -100 -100 -100 0 -100 -100 -100 -100 -100 0 -100 -100 -100 -100 -100 -100 0 
	</pre>


 #### <span id="Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Up-Ranges">Adaptive Domain Randomization, Standard Distribution Parameters Up Ranges</span>

The meaning of the parameter depends on the distribution(s) considered.  
If univariate distributions are used, the order reflects the corresponding image parameter to be randomized:
- Univariate uniform distribution,   
    $$\mathcal{U}(\text{{lower brightness}}, \text{{upper brightness}});\quad  \mathcal{U}(\text{{lower contrast}}, \text{{upper contrast}});\quad  \mathcal{U}(\text{{lower horizontal filp}}, \text{{upper horizontal flip}});\quad  \mathcal{U}(\text{{lower vertical flip}}, \text{{upper vertical flip}});  \mathcal{U}(\text{{lower hue}}, \text{{upper hue}});\quad  \mathcal{U}(\text{{lower jpeg quality}}, \text{{upper jpeg quality}});\quad  \mathcal{U}(\text{{lower saturation}}, \text{{upper saturation}})$$
    standard distribution parameters up ranges,
    $$\qquad \mathcal{U}(a, b),\text{upper range for }a = 1, \text{upper range for }b = 1;\quad  \mathcal{U}(a, b),\text{upper range for }a = \infty, \text{upper range for }b = \infty;\quad  \mathcal{U}(a, b),\text{upper range for }a = \infty, \text{upper range for }b = \infty;\quad  \mathcal{U}(a, b),\text{upper range for }a = \infty, \text{upper range for }b = \infty;\quad  \mathcal{U}(a, b),\text{upper range for }a = 1, \text{upper range for }b = 1;\quad  \mathcal{U}(a, b),\text{upper range for }a = 100, \text{upper range for }b = 100;\quad  \mathcal{U}(a, b),\text{upper range for }a = \infty, \text{upper range for }b = \infty$$
	- Linearized version:
	<pre>
	Given lowers=[1, float(inf), float(inf), float(inf), 1, 100, float(inf)], uppers= [1, float(inf), float(inf), float(inf), 1, 100, float(inf)] -> 1 float(inf) float(inf) float(inf) 1 100 float(inf) 1 float(inf) float(inf) float(inf) 1 100 float(inf)
	</pre>
  &nbsp;
- Univariate triangular distribution,
    $$\mathcal{T}(\text{{lower brightness}}, \text{{mode brightness}}, \text{{upper brightness}});\quad  \mathcal{T}(\text{{lower contrast}}, \text{{mode contrast}}, \text{{upper contrast}});\quad  \mathcal{T}(\text{{lower horizontal filp}}, \text{{mode horizontal flip}}, \text{{upper horizontal flip}});\quad  \mathcal{T}(\text{{lower vertical flip}}, \text{{mode vertical flip}}, \text{{upper vertical flip}}); \quad \mathcal{T}(\text{{lower hue}}, \text{{mode hue}}, \text{{upper hue}});\quad  \mathcal{T}(\text{{lower jpeg quality}}, \text{{mode jpeg quality}}, \text{{upper jpeg quality}});\quad  \mathcal{T}(\text{{lower saturation}}, \text{{mode saturation}}, \text{{upper saturation}})$$
    standard distribution parameters up ranges,
        $$\qquad \mathcal{T}(a, m, b),\text{upper range for }a = 1, \text{upper range for }m = 1, \text{upper range for }b = 1;\quad  \mathcal{T}(a, m, b),\text{upper range for }a = \infty, \text{upper range for }m = \infty, \text{upper range for }b = \infty;\quad  \mathcal{T}(a, m, b),\text{upper range for }a = \infty, \text{upper range for }m = \infty, \text{upper range for }b = \infty;\quad  \mathcal{T}(a, m, b),\text{upper range for }a = \infty, \text{upper range for }m = \infty, \text{upper range for }b = \infty;\quad  \mathcal{T}(a, m, b),\text{upper range for }a = 1, \text{upper range for }m = 1, \text{upper range for }b = 1;\quad  \mathcal{T}(a, m, b),\text{upper range for }a = 100, \text{upper range for }m = 100, \text{upper range for }b = 100;\quad  \mathcal{T}(a, m, b),\text{upper range for }a = \infty, \text{upper range for }m = \infty, \text{upper range for }b = \infty$$
	- Linearized version:
	<pre>
	Given lowers=[1, float(inf), float(inf), float(inf), 1, 100, float(inf)], modes= [1, float(inf), float(inf), float(inf), 1, 100, float(inf)], uppers[1, float(inf), float(inf), float(inf), 1, 100, float(inf)] -> 1 float(inf) float(inf) float(inf) 1 100 float(inf) 1 float(inf) float(inf) float(inf) 1 100 float(inf) 1 float(inf) float(inf) float(inf) 1 100 float(inf)
	</pre>
  &nbsp;
- Univariate normal distribution,
        $$\mathcal{N}(\text{{mean brightness}}, \text{{variance brightness}});\quad  \mathcal{N}(\text{{mean contrast}}, \text{{variance contrast}});\quad  \mathcal{N}(\text{{mean horizontal flip}}, \text{{variance horizontal flip}});\quad  \mathcal{N}(\text{{mean vertical flip}}, \text{{variance vertical flip}});  \mathcal{N}(\text{{mean hue}}, \text{{variance hue}});\quad  \mathcal{N}(\text{{mean jpeg quality}}, \text{{variance jpeg quality}});\quad  \mathcal{N}(\text{{mean saturation}}, \text{{variance saturation}})$$
    standard distribution parameters up ranges,
    $$\qquad \mathcal{N}(\mu, \sigma^2),\text{upper range for }\mu = 1, \text{upper range for }\sigma^2 = 0.4;\quad  \mathcal{N}(\mu, \sigma^2),\text{upper range for }\mu = \infty, \text{upper range for }\sigma^2 = 10;\quad  \mathcal{N}(\mu, \sigma^2),\text{upper range for }\mu = \infty, \text{upper range for }\sigma^2 = 10;\quad  \mathcal{N}(\mu, \sigma^2),\text{upper range for }\mu = \infty, \text{upper range for }\sigma^2 = 10;\quad  \mathcal{N}(\mu, \sigma^2),\text{upper range for }\mu = 1, \text{upper range for }\sigma^2 = 0.4;\quad  \mathcal{N}(\mu, \sigma^2),\text{upper range for }\mu = 100, \text{upper range for }\sigma^2 = 25;\quad  \mathcal{N}(\mu, \sigma^2),\text{upper range for }\mu = \infty, \text{upper range for }\sigma^2 = 10$$
	- Linearized version:
	<pre>
	Given means=[1, float(inf), float(inf), float(inf), 1, 100, float(inf)], variances=[0.4, 10, 10, 10, 0.4, 25, 10] -> 1 float(inf) float(inf) float(inf) 1 100 float(inf) 0.4 10 10 10 0.4 25 10
	</pre>
 &nbsp;       
- Multivariate normal distribution,
	 
	```math
		\mathcal{N}_7(\begin{bmatrix}\mu_\text{brightness}\\\mu_\text{contrast}\\\mu_\text{horizontal flip}\\\mu_\text{vertical flip}\\\mu_\text{hue}\\\mu_\text{jpeg quality}\\\mu_\text{saturation}\end{bmatrix}, \\
		\begin{bmatrix}
		\sigma^2_{\text{brightness}} & \sigma_{\text{brightness}, \text{contrast}} & \cdots & \sigma_{\text{brightness}, \text{saturation}} \\
		\sigma_{\text{contrast}, \text{brightness}} & \sigma^2_{\text{contrast}} & \cdots & \sigma_{\text{contrast}, \text{saturation}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\sigma_{\text{saturation}, \text{brightness}} & \sigma_{\text{saturation}, \text{contrast}} & \cdots & \sigma^2_{\text{saturation}}
		\end{bmatrix}) \\,
 	```

	```math
 		\mu: \text{mean}, \quad \sigma^2: \text{variance} \quad and \quad \sigma: \text{covariance}
 	```

	standard distribution parameters up ranges,

 	```math
		\mathcal{N}_7(\begin{bmatrix}
		\mu_1 \\
		\mu_2 \\
		\mu_3 \\
		\mu_4 \\
		\mu_5 \\
		\mu_6 \\
		\mu_7
		\end{bmatrix}, \\
		\begin{bmatrix}
		\sigma^2_1 & \sigma_{12} & \sigma_{13} & \sigma_{14} & \sigma_{15} & \sigma_{16} & \sigma_{17} \\
		\sigma_{21} & \sigma^2_2 & \sigma_{23} & \sigma_{24} & \sigma_{25} & \sigma_{26} & \sigma_{27} \\
		\sigma_{31} & \sigma_{32} & \sigma^2_3 & \sigma_{34} & \sigma_{35} & \sigma_{36} & \sigma_{37} \\
		\sigma_{41} & \sigma_{42} & \sigma_{43} & \sigma^2_4 & \sigma_{45} & \sigma_{46} & \sigma_{47} \\
		\sigma_{51} & \sigma_{52} & \sigma_{53} & \sigma_{54} & \sigma^2_5 & \sigma_{56} & \sigma_{57} \\
		\sigma_{61} & \sigma_{62} & \sigma_{63} & \sigma_{64} & \sigma_{65} & \sigma^2_6 & \sigma_{67} \\
		\sigma_{71} & \sigma_{72} & \sigma_{73} & \sigma_{74} & \sigma_{75} & \sigma_{76} & \sigma^2_7
		\end{bmatrix}),
	```

	```math
		\text{upper range for }\mu_1 = 1,
 	```
 
 	```math
  		\text{upper range for }\mu_2 = \infty, 
  	```

  	```math
  		\text{upper range for }\mu_3 = \infty, 
  	```

   	```math
  		\text{upper range for }\mu_4 = \infty, 
  	```

   	```math
  		\text{upper range for }\mu_5 = 1, 
  	```

   	```math
  		\text{upper range for }\mu_6 = 100, 
  	```

   	```math
  		\text{upper range for }\mu_7 = \infty,
  	```

	```math
  		\text{upper range for }\sigma^2_i = 0.4, i \in \{1, 5\},
  	```
	
	```math
  		\text{upper range for }\sigma^2_i = 10, i \in \{2, 3, 4, 7\},
  	```

	```math
  		\text{upper range for }\sigma^2_i = 25, i = 6,
  	```

	```math
  		\text{upper range for }\sigma_{ij} = 100, i, j \in \{1, 2, 3, 4, 5, 6, 7\} \quad \text{s.t.} \quad i \neq j
  	```

	- Linearized version
   		(note that the variance covariance matrix is a _positive definite matrix_: only elements of diagonal and elements below diagonal has to be specified):
	<pre>
	Given mean_vector=[1, float(inf), float(inf), float(inf), 1, 100, float(inf)], variance_covariance_matrix=[[0.4], [100, 10], [100, 100, 10], [100, 100, 100, 10], [100, 100, 100, 100, 0.4], [100, 100, 100, 100, 100, 25], [100, 100, 100, 100, 100, 100, 10]] -> 1 float(inf) float(inf) float(inf) 1 100 float(inf) 0.4 100 10 100 100 10 100 100 100 10 100 100 100 100 0.4 100 100 100 100 100 25 100 100 100 100 100 100 10
	</pre>


 #### <span id="Adaptive-Domain-Randomization-Standard-Distribution-Parameters-Initial-Values">Adaptive Domain Randomization, Standard Distribution Parameters Initial Values</span>

The meaning of the parameter depends on the distribution(s) considered.  
If univariate distributions are used, the order reflects the corresponding image parameter to be randomized:
- Univariate uniform distribution,   
    $$\mathcal{U}(\text{{lower brightness}}, \text{{upper brightness}});\quad  \mathcal{U}(\text{{lower contrast}}, \text{{upper contrast}});\quad  \mathcal{U}(\text{{lower horizontal filp}}, \text{{upper horizontal flip}});\quad  \mathcal{U}(\text{{lower vertical flip}}, \text{{upper vertical flip}});  \mathcal{U}(\text{{lower hue}}, \text{{upper hue}});\quad  \mathcal{U}(\text{{lower jpeg quality}}, \text{{upper jpeg quality}});\quad  \mathcal{U}(\text{{lower saturation}}, \text{{upper saturation}})$$
    standard distribution parameters initial values,
    $$\qquad \mathcal{U}(a, b),\text{initial value for }a = 0, \text{initial value for }b = 0;\quad  \mathcal{U}(a, b),\text{initial value for }a = 0, \text{initial value for }b = 0;\quad  \mathcal{U}(a, b),\text{initial value for }a = 0, \text{initial value for }b = 0;\quad  \mathcal{U}(a, b),\text{initial value for }a = 0, \text{initial value for }b = 0;\quad  \mathcal{U}(a, b),\text{initial value for }a = 0, \text{initial value for }b = 0;\quad  \mathcal{U}(a, b),\text{initial value for }a = 50, \text{initial value for }b = 50;\quad  \mathcal{U}(a, b),\text{initial value for }a = 1.25, \text{initial value for }b = 1.25$$
	- Linearized version:
	<pre>
	Given lowers=[0, 0, 0, 0, 0, 50, 1.25], uppers= [0, 0, 0, 0, 0, 50, 1.25] -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25
	</pre>
  &nbsp;
- Univariate triangular distribution,
    $$\mathcal{T}(\text{{lower brightness}}, \text{{mode brightness}}, \text{{upper brightness}});\quad  \mathcal{T}(\text{{lower contrast}}, \text{{mode contrast}}, \text{{upper contrast}});\quad  \mathcal{T}(\text{{lower horizontal filp}}, \text{{mode horizontal flip}}, \text{{upper horizontal flip}});\quad  \mathcal{T}(\text{{lower vertical flip}}, \text{{mode vertical flip}}, \text{{upper vertical flip}}); \quad \mathcal{T}(\text{{lower hue}}, \text{{mode hue}}, \text{{upper hue}});\quad  \mathcal{T}(\text{{lower jpeg quality}}, \text{{mode jpeg quality}}, \text{{upper jpeg quality}});\quad  \mathcal{T}(\text{{lower saturation}}, \text{{mode saturation}}, \text{{upper saturation}})$$
    standard distribution parameters initial values,
        $$\qquad \mathcal{T}(a, m, b),\text{initial value for }a = 0, \text{initial value for }m = 0, \text{initial value for }b = 0;\quad  \mathcal{T}(a, m, b),\text{initial value for }a = 0, \text{initial value for }m = 0, \text{initial value for }b = 0;\quad  \mathcal{T}(a, m, b),\text{initial value for }a = 0, \text{initial value for }m = 0, \text{initial value for }b = 0;\quad  \mathcal{T}(a, m, b),\text{initial value for }a = 0, \text{initial value for }m = 0, \text{initial value for }b = 0;\quad  \mathcal{T}(a, m, b),\text{initial value for }a = 0, \text{initial value for }m = 0, \text{initial value for }b = 0;\quad  \mathcal{T}(a, m, b),\text{initial value for }a = 50, \text{initial value for }m = 50, \text{initial value for }b = 50;\quad  \mathcal{T}(a, m, b),\text{initial value for }a = 1.25, \text{initial value for }m = 1.25, \text{initial value for }b = 1.25$$
	- Linearized version:
	<pre>
	Given lowers=[0, 0, 0, 0, 0, 50, 1.25], modes= [0, 0, 0, 0, 0, 50, 1.25], uppers=[0, 0, 0, 0, 0, 50, 1.25] -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25
	</pre>
  &nbsp;
- Univariate normal distribution,
        $$\mathcal{N}(\text{{mean brightness}}, \text{{variance brightness}});\quad  \mathcal{N}(\text{{mean contrast}}, \text{{variance contrast}});\quad  \mathcal{N}(\text{{mean horizontal flip}}, \text{{variance horizontal flip}});\quad  \mathcal{N}(\text{{mean vertical flip}}, \text{{variance vertical flip}});  \mathcal{N}(\text{{mean hue}}, \text{{variance hue}});\quad  \mathcal{N}(\text{{mean jpeg quality}}, \text{{variance jpeg quality}});\quad  \mathcal{N}(\text{{mean saturation}}, \text{{variance saturation}})$$
    standard distribution parameters initial values,
    $$\qquad \mathcal{N}(\mu, \sigma^2),\text{initial value for }\mu = 0, \text{initial value for }\sigma^2 = 0.2;\quad  \mathcal{N}(\mu, \sigma^2),\text{initial value for }\mu = 0, \text{initial value for }\sigma^2 = 5;\quad  \mathcal{N}(\mu, \sigma^2),\text{initial value for }\mu = 0, \text{initial value for }\sigma^2 = 5;\quad  \mathcal{N}(\mu, \sigma^2),\text{initial value for }\mu = 0, \text{initial value for }\sigma^2 = 5;\quad  \mathcal{N}(\mu, \sigma^2),\text{initial value for }\mu = 0, \text{initial value for }\sigma^2 = 0.2;\quad  \mathcal{N}(\mu, \sigma^2),\text{initial value for }\mu = 50, \text{initial value for }\sigma^2 = 12.5;\quad  \mathcal{N}(\mu, \sigma^2),\text{initial value for }\mu = 1.25, \text{initial value for }\sigma^2 = 5$$
	- Linearized version:
	<pre>
	Given means=[0, 0, 0, 0, 0, 50, 1.25], variances=[0.2, 5, 5, 5, 0.2, 12.5, 5] -> 0 0 0 0 0 50 1.25 0.2 5 5 5 0.2 12.5 5
	</pre>
 &nbsp;       
- Multivariate normal distribution,
	 
	```math
		\mathcal{N}_7(\begin{bmatrix}\mu_\text{brightness}\\\mu_\text{contrast}\\\mu_\text{horizontal flip}\\\mu_\text{vertical flip}\\\mu_\text{hue}\\\mu_\text{jpeg quality}\\\mu_\text{saturation}\end{bmatrix}, \\
		\begin{bmatrix}
		\sigma^2_{\text{brightness}} & \sigma_{\text{brightness}, \text{contrast}} & \cdots & \sigma_{\text{brightness}, \text{saturation}} \\
		\sigma_{\text{contrast}, \text{brightness}} & \sigma^2_{\text{contrast}} & \cdots & \sigma_{\text{contrast}, \text{saturation}} \\
		\vdots & \vdots & \ddots & \vdots \\
		\sigma_{\text{saturation}, \text{brightness}} & \sigma_{\text{saturation}, \text{contrast}} & \cdots & \sigma^2_{\text{saturation}}
		\end{bmatrix}) \\,
 	```

	```math
 		\mu: \text{mean}, \quad \sigma^2: \text{variance} \quad and \quad \sigma: \text{covariance}
 	```

	standard distribution parameters initial values,

 	```math
		\mathcal{N}_7(\begin{bmatrix}
		\mu_1 \\
		\mu_2 \\
		\mu_3 \\
		\mu_4 \\
		\mu_5 \\
		\mu_6 \\
		\mu_7
		\end{bmatrix}, \\
		\begin{bmatrix}
		\sigma^2_1 & \sigma_{12} & \sigma_{13} & \sigma_{14} & \sigma_{15} & \sigma_{16} & \sigma_{17} \\
		\sigma_{21} & \sigma^2_2 & \sigma_{23} & \sigma_{24} & \sigma_{25} & \sigma_{26} & \sigma_{27} \\
		\sigma_{31} & \sigma_{32} & \sigma^2_3 & \sigma_{34} & \sigma_{35} & \sigma_{36} & \sigma_{37} \\
		\sigma_{41} & \sigma_{42} & \sigma_{43} & \sigma^2_4 & \sigma_{45} & \sigma_{46} & \sigma_{47} \\
		\sigma_{51} & \sigma_{52} & \sigma_{53} & \sigma_{54} & \sigma^2_5 & \sigma_{56} & \sigma_{57} \\
		\sigma_{61} & \sigma_{62} & \sigma_{63} & \sigma_{64} & \sigma_{65} & \sigma^2_6 & \sigma_{67} \\
		\sigma_{71} & \sigma_{72} & \sigma_{73} & \sigma_{74} & \sigma_{75} & \sigma_{76} & \sigma^2_7
		\end{bmatrix}),
	```

	```math
		\text{initial value for }\mu_1 = 0,
 	```
 
 	```math
  		\text{initial value for }\mu_2 = 0, 
  	```

  	```math
  		\text{initial value for }\mu_3 = 0, 
  	```

   	```math
  		\text{initial value for }\mu_4 = 0, 
  	```

   	```math
  		\text{initial value for }\mu_5 = 0, 
  	```

   	```math
  		\text{initial value for }\mu_6 = 1.25, 
  	```

   	```math
  		\text{initial value for }\mu_7 = 50,
  	```

	```math
  		\text{initial value for }\sigma^2_i = 0.2, i \in \{1, 5\},
  	```
	
	```math
  		\text{initial value for }\sigma^2_i = 5, i \in \{2, 3, 4, 7\},
  	```

	```math
  		\text{initial value for }\sigma^2_i = 12.5, i = 6,
  	```

	```math
  		\text{initial value for }\sigma_{ij} = 0, i, j \in \{1, 2, 3, 4, 5, 6, 7\} \quad \text{s.t.} \quad i \neq j
  	```

	- Linearized version
   		(note that the variance covariance matrix is a _positive definite matrix_: only elements of diagonal and elements below diagonal has to be specified):
	<pre>
	Given mean_vector=[0, 0, 0, 0, 0, 1.25, 50], variance_covariance_matrix=[[0.2], [0, 5], [0, 0, 5], [0, 0, 0, 5], [0, 0, 0, 0, 0.2], [0, 0, 0, 0, 0, 12.5], [0, 0, 0, 0, 0, 0, 5]] -> 0 0 0 0 0 1.25 500.2 0 5 0 0 5 0 0 0 5 0 0 0 0 0.2 0 0 0 0 0 12.5 0 0 0 0 0 0 5
	</pre>


### Image parameters

#### <span id="Parameters-names">Parameters names</span>

- brightness
- contrast
- horizontally flip
- vertically flip
- hue
- jpeg quality
- saturation

### Domain randomization modes

#### <span id="Distributions">Distributions</span>

- multivariate normal
- univariate normal
- uniform
- triangular

### Neural Networks

#### <span id="Neural-Networks">Networks</span>

1. ResNet 1 (See <a href="#ResNet-1">info</a>)
2. ResNet 2.0 (See <a href="#resnet-20">info</a>)
3. ResNet 2.1 (See <a href="#resnet-21">info</a>)
4. ResNet 2.0.1 (See <a href="#resnet-201">info</a>)
5. ResNet 2.1.1 (See <a href="#resnet-211">info</a>)
   

##### <span id="ResNet-1">ResNet 1</span>

Very fast training, potential low performances.

##### <span id="resnet-20">ResNet 2.0</span>

Slowest training, able to reach the best performances.

##### <span id="resnet-21">ResNet 2.1</span>

Fast training, potential low performances, generally better than ResNet 1.

##### <span id="resnet-201">ResNet 2.0.1</span>

Deeper version of ResNet 2.0, useful for large datasets with many classes.

##### <span id="resnet-211">ResNet 2.1.1</span>

Deeper version of ResNet 2.1, useful for large datasets with many classes. 
  
### Optimizers

#### <span id="Gradient-Based-Optimizers">Gradient Based Optimizers</span>

1. SGD
2. RMSprop
3. Adagrad
4. Adadelta
5. Adafactor
6. Adam
7. Adamax
8. AdamW
9. Lion
10. LossScale
11. Nadam
12. FTRL
13. ProximalGradientDescent
14. ProximalAdagrad
15. Schedules

##### <span id="Gradient-Based-Optimizers-Losses">Gradient Based Optimizers Losses</span>

1. MeanSquaredError
2. MeanAbsoluteError
3. MeanAbsolutePercentageError
4. MeanSquaredLogarithmicError
5. SquaredHinge
6. Hinge
7. CategoricalHinge
8. LogCosh
9. Huber
10. CategoricalCrossentropy
11. SparseCategoricalCrossentropy
12. BinaryCrossentropy
13. KLDivergence
14. Poisson
15. CosineSimilarity
16. serialize
17. deserialize

#### <span id="Gradient-Free-Optimizers">Gradient Free Optimizers</span>

1. RandomSearch
2. QORandomSearch
3. ORandomSearch
4. RandomSearchPlusMiddlePoint
5. MetaRecentering
6. MetaTuneRecentering
7. HullAvgMetaTuneRecentering
8. HullAvgMetaRecentering
9. AvgMetaRecenteringNoHull
10. HaltonSearch
11. HaltonSearchPlusMiddlePoint
12. LargeHaltonSearch
13. ScrHaltonSearch
14. ScrHaltonSearchPlusMiddlePoint
15. HammersleySearch
16. HammersleySearchPlusMiddlePoint
17. ScrHammersleySearchPlusMiddlePoint
18. ScrHammersleySearch
19. QOScrHammersleySearch
20. OScrHammersleySearch
21. CauchyScrHammersleySearch
22. LHSSearch
23. CauchyLHSSearch
24. DE
25. TwoPointsDE
26. VoronoiDE
27. RotatedTwoPointsDE
28. LhsDE
29. QrDE
30. QODE
31. SPQODE
32. QOTPDE
33. LQOTPDE
34. LQODE
35. SODE
36. NoisyDE
37. AlmostRotationInvariantDE
38. RotationInvariantDE
39. DiscreteDE
40. RecES
41. RecMixES
42. RecMutDE
43. ES
44. MixES
45. MutDE
46. NonNSGAIIES
47. AX
48. BOBYQA
49. NelderMead
50. CmaFmin2
51. Powell
52. RPowell
53. BFGS
54. RBFGS
55. LBFGSB
56. Cobyla
57. RCobyla
58. SQP
59. RSQP
60. NLOPT_LN_SBPLX
61. NLOPT_LN_PRAXIS
62. NLOPT_GN_DIRECT
63. NLOPT_GN_DIRECT_L
64. NLOPT_GN_CRS2_LM
65. NLOPT_GN_AGS
66. NLOPT_GN_ISRES
67. NLOPT_GN_ESCH
68. NLOPT_LN_COBYLA
69. NLOPT_LN_BOBYQA
70. NLOPT_LN_NEWUOA_BOUND
71. NLOPT_LN_NELDERMEAD
72. SMAC3
73. PymooCMAES
74. PymooBIPOP
75. PymooNSGA2
76. pysot
77. DSbase
78. DS3p
79. DSsubspace
80. DSproba
81. DSproba2
82. DSproba3
83. DSproba4
84. DSproba5
85. DSproba6
86. DSproba7
87. DSproba8
88. DSproba9
89. OnePlusOne
90. OnePlusLambda
91. NoisyOnePlusOne
92. DiscreteOnePlusOne
93. SADiscreteLenglerOnePlusOneExp09
94. SADiscreteLenglerOnePlusOneExp099
95. SADiscreteLenglerOnePlusOneExp09Auto
96. SADiscreteLenglerOnePlusOneLinAuto
97. SADiscreteLenglerOnePlusOneLin1
98. SADiscreteLenglerOnePlusOneLin100
99. SADiscreteOnePlusOneExp099
100. SADiscreteOnePlusOneLin100
101. SADiscreteOnePlusOneExp09
102. DiscreteOnePlusOneT
103. PortfolioDiscreteOnePlusOne
104. PortfolioDiscreteOnePlusOneT
105. DiscreteLenglerOnePlusOne
106. DiscreteLengler2OnePlusOne
107. DiscreteLengler3OnePlusOne
108. DiscreteLenglerHalfOnePlusOne
109. DiscreteLenglerFourthOnePlusOne
110. DiscreteLenglerOnePlusOneT
111. AdaptiveDiscreteOnePlusOne
112. LognormalDiscreteOnePlusOne
113. AnisotropicAdaptiveDiscreteOnePlusOne
114. DiscreteBSOOnePlusOne
115. DiscreteDoerrOnePlusOne
116. CauchyOnePlusOne
117. OptimisticNoisyOnePlusOne
118. OptimisticDiscreteOnePlusOne
119. NoisyDiscreteOnePlusOne
120. DoubleFastGADiscreteOnePlusOne
121. RLSOnePlusOne
122. SparseDoubleFastGADiscreteOnePlusOne
123. RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
124. RecombiningPortfolioDiscreteOnePlusOne
125. ChoiceBase
126. OldCMA
127. LargeCMA
128. LargeDiagCMA
129. TinyCMA
130. CMAbounded
131. CMAsmall
132. CMAstd
133. CMApara
134. CMAtuning
135. MetaCMA
136. DiagonalCMA
137. SDiagonalCMA
138. FCMA
139. CMA
140. EDA
141. TBPSA
142. NaiveTBPSA
143. NoisyBandit
144. RealSpacePSO
145. PSO
146. QOPSO
147. QORealSpacePSO
148. SQOPSO
149. SOPSO
150. SQORealSpacePSO
151. SPSA
152. RescaledCMA
153. TinyLhsDE
154. TinyQODE
155. TinySQP
156. MicroSQP
157. TinySPSA
158. MicroSPSA
159. VastLengler
160. VastDE
161. Portfolio
162. ParaPortfolio
163. ASCMADEthird
164. MultiCMA
165. MultiDS
166. TripleCMA
167. PolyCMA
168. MultiScaleCMA
169. LPCMA
170. VLPCMA
171. MetaModel
172. NeuralMetaModel
173. SVMMetaModel
174. RFMetaModel
175. MetaModelOnePlusOne
176. MetaModelDSproba
177. RFMetaModelOnePlusOne
178. MetaModelPSO
179. RFMetaModelPSO
180. SVMMetaModelPSO
181. MetaModelDE
182. MetaModelQODE
183. NeuralMetaModelDE
184. SVMMetaModelDE
185. RFMetaModelDE
186. MetaModelTwoPointsDE
187. NeuralMetaModelTwoPointsDE
188. SVMMetaModelTwoPointsDE
189. RFMetaModelTwoPointsDE
190. MultiBFGSPlus
191. LogMultiBFGSPlus
192. SqrtMultiBFGSPlus
193. MultiCobylaPlus
194. MultiSQPPlus
195. BFGSCMAPlus
196. LogBFGSCMAPlus
197. SqrtBFGSCMAPlus
198. SQPCMAPlus
199. LogSQPCMAPlus
200. SqrtSQPCMAPlus
201. MultiBFGS
202. LogMultiBFGS
203. SqrtMultiBFGS
204. MultiCobyla
205. ForceMultiCobyla
206. MultiSQP
207. BFGSCMA
208. LogBFGSCMA
209. SqrtBFGSCMA
210. SQPCMA
211. LogSQPCMA
212. SqrtSQPCMA
213. FSQPCMA
214. F2SQPCMA
215. F3SQPCMA
216. MultiDiscrete
217. CMandAS2
218. CMandAS3
219. CM
220. BO
221. BOSplit
222. PCABO
223. BayesOptimBO
224. GeneticDE
225. MemeticDE
226. QNDE
227. ChainDE
228. OpoDE
229. OpoTinyDE
230. Carola1
231. Carola2
232. DS2
233. Carola4
234. DS4
235. Carola5
236. DS5
237. Carola6
238. DS6
239. PCarola6
240. pCarola6
241. Carola7
242. Carola8
243. DS8
244. Carola9
245. DS9
246. Carola10
247. Carola3
248. BAR
249. BAR2
250. BAR3
251. discretememetic
252. ChainCMAPowell
253. ChainDSPowell
254. ChainMetaModelSQP
255. ChainMetaModelDSSQP
256. ChainMetaModelPowell
257. ChainDiagonalCMAPowell
258. ChainNaiveTBPSAPowell
259. ChainNaiveTBPSACMAPowell
260. BAR4
261. cGA
262. NaiveIsoEMNA
263. NGOptBase
264. NGOptDSBase
265. Shiwa
266. NGO
267. NGOpt4
268. NGOpt8
269. NGOpt10
270. NGOpt15
271. NGOpt16
272. NGOpt36
273. NGOpt39
274. NGOptRW
275. NGOptF
276. NGOptF2
277. NGOptF3
278. NGOptF5
279. NGOpt
280. Wiz
281. NgIoh
282. NgIoh2
283. NgIoh3
284. NgIoh4
285. NgIohRW2
286. NgIoh5
287. NgIoh6
288. SmoothDiscreteOnePlusOne
289. SmoothPortfolioDiscreteOnePlusOne
290. SmoothDiscreteLenglerOnePlusOne
291. SmoothDiscreteLognormalOnePlusOne
292. SuperSmoothDiscreteLenglerOnePlusOne
293. UltraSmoothDiscreteLenglerOnePlusOne
294. SmoothLognormalDiscreteOnePlusOne
295. SmoothAdaptiveDiscreteOnePlusOne
296. SmoothRecombiningPortfolioDiscreteOnePlusOne
297. SmoothRecombiningDiscreteLanglerOnePlusOne
298. UltraSmoothRecombiningDiscreteLanglerOnePlusOne
299. UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne
300. UltraSmoothElitistRecombiningDiscreteLanglerOnePlusOne
301. SuperSmoothElitistRecombiningDiscreteLanglerOnePlusOne
302. SuperSmoothRecombiningDiscreteLanglerOnePlusOne
303. SuperSmoothRecombiningDiscreteLognormalOnePlusOne
304. SmoothElitistRecombiningDiscreteLanglerOnePlusOne
305. SmoothElitistRandRecombiningDiscreteLanglerOnePlusOne
306. SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne
307. RecombiningDiscreteLanglerOnePlusOne
308. RecombiningDiscreteLognormalOnePlusOne
309. MaxRecombiningDiscreteLenglerOnePlusOne
310. MinRecombiningDiscreteLenglerOnePlusOne
311. OnePtRecombiningDiscreteLenglerOnePlusOne
312. TwoPtRecombiningDiscreteLenglerOnePlusOne
313. RandRecombiningDiscreteLenglerOnePlusOne
314. RandRecombiningDiscreteLognormalOnePlusOne
315. NgIoh7
316. NgDS11
317. NgIoh11
318. NgIoh14
319. NgIoh13
320. NgIoh15
321. NgIoh12
322. NgIoh16
323. NgIoh17
324. NgDS
325. NgIoh21
326. NgDS2
327. NGDSRW
328. NgIoh20
329. NgIoh19
330. NgIoh18
331. NgIoh10
332. NgIoh9
333. NgIoh8
334. MixDeterministicRL
335. SpecialRL
336. NoisyRL1
337. NoisyRL2
338. NoisyRL3
339. OnePointDE
340. ParametrizationDE
341. MiniDE
342. MiniLhsDE
343. MiniQrDE
344. AlmostRotationInvariantDEAndBigPop
345. BPRotationInvariantDE
346. MilliCMA
347. MicroCMA
348. FCMAs03
349. FCMAp13
350. ECMA
351. MetaModelDiagonalCMA
352. MetaModelFmin2
353. LSCMA
354. HSCMA
355. HSNeuralCMA
356. HSSVMCMA
357. HSRFCMA
358. HSMetaModel
359. FastGADiscreteOnePlusOne
360. DoubleFastGAOptimisticNoisyDiscreteOnePlusOne
361. RecombiningGA
362. RotatedRecombiningGA
363. FastGAOptimisticNoisyDiscreteOnePlusOne
364. FastGANoisyDiscreteOnePlusOne
365. PortfolioOptimisticNoisyDiscreteOnePlusOne
366. PortfolioNoisyDiscreteOnePlusOne
367. RecombiningOptimisticNoisyDiscreteOnePlusOne
368. RBO
369. QRBO
370. MidQRBO
371. LBO
372. IsoEMNA
373. NaiveAnisoEMNA
374. AnisoEMNA
375. IsoEMNATBPSA
376. NaiveIsoEMNATBPSA
377. AnisoEMNATBPSA
378. NaiveAnisoEMNATBPSA
379. MetaCauchyRecentering
380. ChainCMASQP
381. ChainDEwithR
382. ChainDEwithRsqrt
383. ChainDEwithRdim
384. ChainDEwithR30
385. ChainDEwithLHS
386. ChainDEwithLHSsqrt
387. ChainDEwithLHSdim
388. ChainDEwithLHS30
389. ChainDEwithMetaRecentering
390. ChainDEwithMetaRecenteringsqrt
391. ChainDEwithMetaRecenteringdim
392. ChainDEwithMetaRecentering30
393. ChainBOwithMetaTuneRecentering
394. ChainBOwithMetaTuneRecenteringsqrt
395. ChainBOwithMetaTuneRecenteringdim
396. ChainBOwithMetaTuneRecentering30
397. ChainDEwithMetaTuneRecentering
398. ChainDEwithMetaTuneRecenteringsqrt
399. ChainDEwithMetaTuneRecenteringdim
400. ChainDEwithMetaTuneRecentering30
401. ChainBOwithR
402. ChainBOwithRsqrt
403. ChainBOwithRdim
404. ChainBOwithR30
405. ChainBOwithLHS30
406. ChainBOwithLHSsqrt
407. ChainBOwithLHSdim
408. ChainBOwithLHS
409. ChainBOwithMetaRecentering30
410. ChainBOwithMetaRecenteringsqrt
411. ChainBOwithMetaRecenteringdim
412. ChainBOwithMetaRecentering
413. ChainPSOwithR
414. ChainPSOwithRsqrt
415. ChainPSOwithRdim
416. ChainPSOwithR30
417. ChainPSOwithLHS30
418. ChainPSOwithLHSsqrt
419. ChainPSOwithLHSdim
420. ChainPSOwithLHS
421. ChainPSOwithMetaRecentering30
422. ChainPSOwithMetaRecenteringsqrt
423. ChainPSOwithMetaRecenteringdim
424. ChainPSOwithMetaRecentering
425. ChainCMAwithR
426. ChainCMAwithRsqrt
427. ChainCMAwithRdim
428. ChainCMAwithR30
429. ChainCMAwithLHS30
430. ChainCMAwithLHSsqrt
431. ChainCMAwithLHSdim
432. ChainCMAwithLHS
433. ChainCMAwithMetaRecentering30
434. ChainCMAwithMetaRecenteringsqrt
435. ChainCMAwithMetaRecenteringdim
436. ChainCMAwithMetaRecentering
437. Zero
438. StupidRandom
439. CauchyRandomSearch
440. RandomScaleRandomSearch
441. RandomScaleRandomSearchPlusMiddlePoint
442. RescaleScrHammersleySearch
443. AvgHammersleySearch
444. AvgHammersleySearchPlusMiddlePoint
445. HullCenterHullAvgRandomSearch
446. AvgRandomSearch
447. TEAvgScrHammersleySearchPlusMiddlePoint
448. TEAvgScrHammersleySearch
449. TEAvgRandomSearch
450. TEAvgCauchyScrHammersleySearch
451. TEAvgLHSSearch
452. TEAvgCauchyLHSSearch
453. HullCenterHullAvgScrHaltonSearch
454. HullCenterHullAvgScrHaltonSearchPlusMiddlePoint
455. HullCenterHullAvgScrHammersleySearchPlusMiddlePoint
456. HullCenterHullAvgLargeHammersleySearch
457. HullCenterHullAvgScrHammersleySearch
458. HullCenterHullAvgCauchyScrHammersleySearch
459. HullCenterHullAvgLHSSearch
460. HullCenterHullAvgCauchyLHSSearch
461. MetaNGOpt10
462. NGOptSingle9
463. NGOptSingle16
464. NGOptSingle25
465. Noisy13Splits
466. NoisyInfSplits
467. DiscreteNoisy13Splits
468. DiscreteNoisyInfSplits
469. PCABO80
470. PCABO95DoE20
471. SparseDiscreteOnePlusOne
472. HSDE
473. LhsHSDE
474. FCarola6
475. Carola11
476. Carola14
477. DS14
478. Carola13
479. Carola15
480. NgIoh12b
481. NgIoh13b
482. NgIoh14b
483. NgIoh15b
484. NgDS3
485. NgLn
486. CSEC
487. CSEC4
488. CSEC5
489. CSEC6
490. CSEC7
491. CSEC8
492. CSEC9
493. CSEC10

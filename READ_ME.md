# Object detection

Script python to detect objects based on the trained model.


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
2. Locate the script named *object_detection.py*
3. Run the script using the following command: python3 [script_name].py [optional_parameters]
4. Optionally, you can pass some parameters to set all object detection options. See <a href="#Usage">next section</a> for details.

Ensure you have Python installed and that the necessary dependencies are met before running the script.

## <span id="Usage">Usage</span>

<pre>   
       object_detection.py  [-h]        
                            [--data_path DATA_PATH]          
                            [--name NAME] 
                            [--format FORMAT]
                            [--model_path MODEL_PATH]
</pre>   

test.py

optional arguments:

<pre>
  -h, --help            
                        Show this help message and exit
  --data_path DATA_PATH
                        Path of the directory in which input data are stored.
                        (default: 'Input\dataset')
  --name NAME           
                        Name of the dataset.
                        (default: 'Electronic components dataset')
  --format FORMAT       
                        Image format (e.g.: RGB, Grayscale, RGBA). 
                        If not present, automatically deduced from images. 
                        (default: None)
  --model_path MODEL_PATH
                        Path of the directory from which the trained model has to be loaded.
                        (default: 'default: in\model')
</pre>

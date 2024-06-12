# Object detection and image classification

python libraries to detect objects and classify images based on the trained model.


## Installation

Python $3.7.*$ is necessary to launch the use the python functions.      
It is necessary to install all python packages listed in requirements.txt.       
Quick install: 
<pre>  
pip install -r requirements.txt        
</pre>

## Execution

To import the functions inside a client python script (ex: *client.py*), please follow these steps:

1. Navigate to the root directory (ex: *aidet*)
2. Locate the directory named *interface*
3. import the available functions with the following code:
   
   ```
          from aidet.interface import detect_objects, classify
   ```
   inside your client python script {client.py}
4. Optionally, you can pass some parameters to the functions to set all object detection/image classification options. See <a href="#Usage">next section</a> for details.

Ensure you have Python installed and that the necessary dependencies are met before running the script.

## <span id="Usage">Usage</span>

### detect_objects

Perform object detections on the given image(s).

```   
detect_objects(
    data_path: string = "Input\\regions",
    name: string = "Electronic components dataset",
    format: string = None, 
    model_path: string = "in\\model", 
    classes_path: string = "Input\\regions", 
    gpu: bool = False, 
    verbose: bool = False, 
    info_dir: string = None
)
```

- optional arguments:

   1. data_path
   
       Path of the directory in which input data are stored (default: 'Input\\regions')
   2. name
      
       Name of the dataset (default: 'Electronic components dataset')
   3. format
      
       Image format (e.g.: RGB, Grayscale, RGBA). If not present, automatically deduced from images. (default: None)
   4. model_path
      
       Path of the directory from which the trained model has to be loaded. (default: 'in\\model')
   5. classes_path
      
       Path of the directory in which classes names are stored (default: 'Input\\regions')
   6. gpu
     
       Use GPU for inference
   7. verbose
 
       Print infos to files into directory {info_dir}
   8. info_dir
      
       Directory in which info files will be stored (if {verbose} is False, use the default value: None)


- returns:

   A list of tuples (x, y, w, h, centroid_x, centroid_y, label, confidence), one for each detected object
  
     1. x

         A float representing the x coordinate of the top-left corner of the rectangular area of the detected_object
     2. y

         A float representing the y coordinate of the top-left corner of the rectangular area of the detected_object
     3. w

         A float representing the width of the rectangular area of the detected_object
     4. h

         A float representing the height of the rectangular area of the detected_object
     5. centroid_x

         A float representing the x coordinate of the centroid of the detected_object
     6. centroid_y

         A float representing the y coordinate of the centroid of the detected_object
     7. label
      
         A string describing the classification related to the object
     8. confidence

         A float (between 0 and 1) describing the confidence (i.e., probability of the prediction) associated with the detection

### classify

Perform classification on the given image(s).

```   
detect_objects(
    data_path: string = "Input\\regions",
    name: string = "Electronic components dataset",
    format: string = None, 
    model_path: string = "in\\model", 
    classes_path: string = "Input\\regions", 
    gpu: bool = False, 
    verbose: bool = False, 
    info_dir: string = None
)
```

- optional arguments:

   1. data_path
   
       Path of the directory in which input data are stored (default: 'Input\\regions')
   2. name
      
       Name of the dataset (default: 'Electronic components dataset')
   3. format
      
       Image format (e.g.: RGB, Grayscale, RGBA). If not present, automatically deduced from images. (default: None)
   4. model_path
      
       Path of the directory from which the trained model has to be loaded. (default: 'in\\model')
   5. classes_path
      
       Path of the directory in which classes names are stored (default: 'Input\\regions')
   6. gpu
     
       Use GPU for inference
   7. verbose
 
       Print infos to files into directory {info_dir}
   8. info_dir
      
       Directory in which info files will be stored (if {verbose} is False, use the default value: None)


- returns:

   A tuples (label, confidence), one for the image classified
  
     1. label
      
         A string describing the classification related to the image
     2. confidence

         A float (between 0 and 1) describing the confidence (i.e., probability of the prediction) associated with the image
        


> **Author:** Giovanni Cadau
>
> **Project:** Electronic Components AI Detection
>
> **Version:** Object detection local version to be run on the local machine.
>
> **Description:** Artificial Intelligence for electronic components detection. Master thesis project. 
>
> **Msc Data Science and Engineering @Politecnico di Torino**
>
> **Held at Seica S.P.A.**

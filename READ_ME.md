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
                        (default: Input\dataset)
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
                        Adaptive domain randomization distribution parameters
                        up ranges. Use the linearized version (i.e. Parameters
                        list, whose cardinality depends on the number of
                        parameters: single parameters separated by 1 single
                        space. A single parameter can be a scalar, a vector or
                        a a matrix. Inside a vector, elements are separated by
                        1 single space. Inside a matrix, rows are representd
                        as a (transpose) vector and inserted separated by 1
                        single space). If None, standard distribution
                        parameters ranges are used. Adaptive domain
                        randomization, standard distribution parameters up
                        ranges. The meaning of the parameter depends on the
                        distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters up
                        ranges, U(a, b), upper range for a = 1, upper range
                        for b = 1; U(a, b), upper range for a = float('inf'),
                        upper range for b = float('inf'); U(a, b), upper range
                        for a = float('inf'), upper range for b =
                        float('inf'); U(a, b), upper range for a =
                        float('inf'), upper range for b = float('inf'); U(a,
                        b), upper range for a = 1, upper range for b = 1; U(a,
                        b), upper range for a = 100, upper range for b = 100;
                        U(a, b), upper range for a = float('inf'), upper range
                        for b = float('inf') Linearized version: Given
                        lowers=[1, float(inf), float(inf), float(inf), 1, 100,
                        float(inf)], uppers=[1, float(inf), float(inf),
                        float(inf), 1, 100, float(inf)] -> 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters up ranges, Tr(a, m, b), upper
                        range for a = 1, upper range for m = 1, upper range
                        for b = 1; Tr(a, m, b), upper range for a =
                        float('inf'), upper range for m = float('inf'), upper
                        range for b = float('inf'); Tr(a, m, b), upper range
                        for a = float('inf'), upper range for m =
                        float('inf'), upper range for b = float('inf'); Tr(a,
                        m, b), upper range for a = float('inf'), upper range
                        for m = float('inf'), upper range for b =
                        float('inf'); Tr(a, m, b), upper range for a = 1,
                        upper range for m = 1, upper range for b = 1; Tr(a, m,
                        b), upper range for a = 100, upper range for m = 100,
                        upper range for b = 100; Tr(a, m, b), upper range for
                        a = float('inf'), upper range for m = float('inf'),
                        upper range for b = float('inf') Linearized version:
                        Given lowers=[1, float(inf), float(inf), float(inf),
                        1, 100, float(inf)], modes= [1, float(inf),
                        float(inf), float(inf), 1, 100, float(inf)], uppers[1,
                        float(inf), float(inf), float(inf), 1, 100,
                        float(inf)] -> 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        up ranges, N(mu, sigma), upper range for mu = 1, upper
                        range for sigma = 0.4; N(mu, sigma), upper range for
                        mu = float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = float('inf'), upper range
                        for sigma = 10; N(mu, sigma), upper range for mu =
                        float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = 1, upper range for sigma
                        = 0.4; N(mu, sigma), upper range for mu = 100, upper
                        range for sigma = 25; N(mu, sigma), upper range for mu
                        = float('inf'), upper range for sigma = 10 Linearized
                        version: Given means=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')], variances=[0.4,
                        10, 10, 10, 0.4, 25, 10] -> 1 float('inf')
                        float('inf') float('inf') 1 100 float('inf') 0.4 10 10
                        10 0.4 25 10 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters up ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), upper range for mu_1 = 1, upper range
                        for mu_2 = float('inf'), upper range for mu_3 =
                        float('inf'), upper range for mu_4 = float('inf'),
                        upper range for mu_5 = 1, upper range for mu_6 = 100,
                        upper range for mu_7 = float('inf'), upper range for
                        sigma_ij = 0.4 if i = j = {1, 5}, upper range for
                        sigma_ij = 10 if i = j = {2, 3, 4, 7}, upper range for
                        sigma_ij = 25 if i = j = 6, upper range for sigma_ij =
                        100 if i != j Linearized version: Given
                        mean_vector=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')],
                        variance_covariance_matrix=[[0.4], [100, 10], [100,
                        100, 10], [100, 100, 100, 10], [100, 100, 100, 100,
                        0.4], [100, 100, 100, 100, 100, 25], [100, 100, 100,
                        100, 100, 100, 10]] -> 1 float('inf') float('inf')
                        float('inf') 1 100 float('inf') 0.4 100 10 100 100 10
                        100 100 100 10 100 100 100 100 0.4 100 100 100 100 100
                        25 100 100 100 100 100 100 10 -- triangular matrix
                        required -- (default: None)
  --adaptive_dom_rand__dist_ranges___uniform____low ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____LOW ...]
                        Adaptive domain randomization uniform distribution
                        parameters low ranges. Use the linearized version
                        (i.e. Parameters list, whose cardinality depends on
                        the number of parameters: single parameters separated
                        by 1 single space. A single parameter can be a scalar,
                        a vector or a a matrix. Inside a vector, elements are
                        separated by 1 single space. Inside a matrix, rows are
                        representd as a (transpose) vector and inserted
                        separated by 1 single space). If None, standard
                        distribution parameters ranges are used. Adaptive
                        domain randomization, standard distribution parameters
                        low ranges. The meaning of the parameter depends on
                        the distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters low
                        ranges, U(a, b), lower range for a = -1, lower range
                        for b = -1; U(a, b), lower range for a =
                        float('-inf'), lower range for b = float('-inf'); U(a,
                        b), lower range for a = float('-inf'), lower range for
                        b = float('-inf'); U(a, b), lower range for a =
                        float('-inf'), lower range for b = float('-inf'); U(a,
                        b), lower range for a = -1, lower range for b = -1;
                        U(a, b), lower range for a = 0, lower range for b = 0;
                        U(a, b), lower range for a = 0, lower range for b = 0
                        Linearized version: Given lowers=[-1, float(-inf),
                        float(-inf), float(-inf), -1, 0, 0], uppers= [-1,
                        float(-inf), float(-inf), float(-inf), -1, 0, 0] -> -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -
                        Univariate triangular distribution,
                        Tr(lower_brightness, mode_brightness,
                        upper_brightness); Tr(lower_contrast, mode_contrast,
                        upper_contrast); Tr(lower_horizontal_flip,
                        mode_horizontal_flip, upper_horizontal_flip);
                        Tr(lower_vertical_flip, mode_vertical_flip,
                        upper_vertical_flip); Tr(lower_hue, mode_hue,
                        upper_hue); Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters low ranges, Tr(a, m, b), lower
                        range for a = -1, lower range for m = -1, lower range
                        for b = -1; Tr(a, m, b), lower range for a =
                        float('-inf'), lower range for m = float('-inf'),
                        lower range for b = float('-inf'); Tr(a, m, b), lower
                        range for a = float('-inf'), lower range for m =
                        float('-inf'), lower range for b = float('-inf');
                        Tr(a, m, b), lower range for a = float('-inf'), lower
                        range for m = float('-inf'), lower range for b =
                        float('-inf'); Tr(a, m, b), lower range for a = -1,
                        lower range for m = -1, lower range for b = -1; Tr(a,
                        m, b), lower range for a = 0, lower range for m = 0,
                        lower range for b = 0; Tr(a, m, b), lower range for a
                        = 0, lower range for m = 0, lower range for b = 0
                        Linearized version: Given lowers=[-1, float(-inf),
                        float(-inf), float(-inf), -1, 0, 0], modes= [-1,
                        float(-inf), float(-inf), float(-inf), -1, 0, 0],
                        uppers[-1, float(-inf), float(-inf), float(-inf), -1,
                        0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0
                        -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -
                        Univariate normal distribution, N(mean_brightness,
                        variance_brightness); N(mean_contrast,
                        variance_contrast); N(mean_horizontal_flip,
                        variance_horizontal_flip); N(mean_vertical_flip,
                        variance_vertical_flip); N(mean_hue, variance_hue);
                        N(mean_jpeg_quality, variance_jpeg_quality);
                        N(mean_saturation, variance_saturation) standard
                        distribution parameters low ranges, N(mu, sigma),
                        lower range for mu = -1, lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = float('-inf'),
                        lower range for sigma = 0; N(mu, sigma), lower range
                        for mu = float('-inf'), lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = float('-inf'),
                        lower range for sigma = 0; N(mu, sigma), lower range
                        for mu = -1, lower range for sigma = 0; N(mu, sigma),
                        lower range for mu = 0, lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = 0, lower range for
                        sigma = 0 Linearized version: Given means=[-1,
                        float('-inf'), float('-inf'), float('-inf'), -1, 0,
                        0], variances=[0, 0, 0, 0, 0, 0, 0] -> -1
                        float('-inf') float('-inf') float('-inf') -1 0 0 0 0 0
                        0 0 0 0 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters low ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), lower range for mu_1 = -1, lower range
                        for mu_2 = float('-inf'), lower range for mu_3 =
                        float('-inf'), lower range for mu_4 = float('-inf'),
                        lower range for mu_5 = -1, lower range for mu_6 = 0,
                        lower range for mu_7 = 0, lower range for sigma_ij = 0
                        if i = j, lower range for sigma_ij = -100 if i != j
                        Linearized version: Given mean_vector=[-1,
                        float('-inf'), float('-inf'), float('-inf'), -1, 0,
                        0], variance_covariance_matrix=[[0], [-100, 0], [-100,
                        -100, 0], [-100, -100, -100, 0], [-100, -100, -100,
                        -100, 0], [-100, -100, -100, -100, -100, 0], [-100,
                        -100, -100, -100, -100, -100, 0]] -> -1 float('-inf')
                        float('-inf') float('-inf') -1 0 0 0 -100 0 -100 -100
                        0 -100 -100 -100 0 -100 -100 -100 -100 0 -100 -100
                        -100 -100 -100 0 -100 -100 -100 -100 -100 -100 0 --
                        triangular matrix required -- (default: None)
  --adaptive_dom_rand__dist_ranges___uniform____up ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIFORM____UP ...]
                        Adaptive domain randomization uniform distribution
                        parameters up ranges. Use the linearized version (i.e.
                        Parameters list, whose cardinality depends on the
                        number of parameters: single parameters separated by 1
                        single space. A single parameter can be a scalar, a
                        vector or a a matrix. Inside a vector, elements are
                        separated by 1 single space. Inside a matrix, rows are
                        representd as a (transpose) vector and inserted
                        separated by 1 single space). If None, standard
                        distribution parameters ranges are used. Adaptive
                        domain randomization, standard distribution parameters
                        up ranges. The meaning of the parameter depends on the
                        distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters up
                        ranges, U(a, b), upper range for a = 1, upper range
                        for b = 1; U(a, b), upper range for a = float('inf'),
                        upper range for b = float('inf'); U(a, b), upper range
                        for a = float('inf'), upper range for b =
                        float('inf'); U(a, b), upper range for a =
                        float('inf'), upper range for b = float('inf'); U(a,
                        b), upper range for a = 1, upper range for b = 1; U(a,
                        b), upper range for a = 100, upper range for b = 100;
                        U(a, b), upper range for a = float('inf'), upper range
                        for b = float('inf') Linearized version: Given
                        lowers=[1, float(inf), float(inf), float(inf), 1, 100,
                        float(inf)], uppers= [1, float(inf), float(inf),
                        float(inf), 1, 100, float(inf)] -> 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters up ranges, Tr(a, m, b), upper
                        range for a = 1, upper range for m = 1, upper range
                        for b = 1; Tr(a, m, b), upper range for a =
                        float('inf'), upper range for m = float('inf'), upper
                        range for b = float('inf'); Tr(a, m, b), upper range
                        for a = float('inf'), upper range for m =
                        float('inf'), upper range for b = float('inf'); Tr(a,
                        m, b), upper range for a = float('inf'), upper range
                        for m = float('inf'), upper range for b =
                        float('inf'); Tr(a, m, b), upper range for a = 1,
                        upper range for m = 1, upper range for b = 1; Tr(a, m,
                        b), upper range for a = 100, upper range for m = 100,
                        upper range for b = 100; Tr(a, m, b), upper range for
                        a = float('inf'), upper range for m = float('inf'),
                        upper range for b = float('inf') Linearized version:
                        Given lowers=[1, float(inf), float(inf), float(inf),
                        1, 100, float(inf)], modes= [1, float(inf),
                        float(inf), float(inf), 1, 100, float(inf)], uppers[1,
                        float(inf), float(inf), float(inf), 1, 100,
                        float(inf)] -> 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        up ranges, N(mu, sigma), upper range for mu = 1, upper
                        range for sigma = 0.4; N(mu, sigma), upper range for
                        mu = float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = float('inf'), upper range
                        for sigma = 10; N(mu, sigma), upper range for mu =
                        float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = 1, upper range for sigma
                        = 0.4; N(mu, sigma), upper range for mu = 100, upper
                        range for sigma = 25; N(mu, sigma), upper range for mu
                        = float('inf'), upper range for sigma = 10 Linearized
                        version: Given means=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')], variances=[0.4,
                        10, 10, 10, 0.4, 25, 10] -> 1 float('inf')
                        float('inf') float('inf') 1 100 float('inf') 0.4 10 10
                        10 0.4 25 10 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters up ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), upper range for mu_1 = 1, upper range
                        for mu_2 = float('inf'), upper range for mu_3 =
                        float('inf'), upper range for mu_4 = float('inf'),
                        upper range for mu_5 = 1, upper range for mu_6 = 100,
                        upper range for mu_7 = float('inf'), upper range for
                        sigma_ij = 0.4 if i = j = {1, 5}, upper range for
                        sigma_ij = 10 if i = j = {2, 3, 4, 7}, upper range for
                        sigma_ij = 25 if i = j = 6, upper range for sigma_ij =
                        100 if i != j Linearized version: Given
                        mean_vector=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')],
                        variance_covariance_matrix=[[0.4], [100, 10], [100,
                        100, 10], [100, 100, 100, 10], [100, 100, 100, 100,
                        0.4], [100, 100, 100, 100, 100, 25], [100, 100, 100,
                        100, 100, 100, 10]] -> 1 float('inf') float('inf')
                        float('inf') 1 100 float('inf') 0.4 100 10 100 100 10
                        100 100 100 10 100 100 100 100 0.4 100 100 100 100 100
                        25 100 100 100 100 100 100 10 -- triangular matrix
                        required -- (default: None)
  --adaptive_dom_rand__dist_ranges___triangular____low ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____LOW ...]
                        Adaptive domain randomization triangular distribution
                        parameters low ranges. Use the linearized version
                        (i.e. Parameters list, whose cardinality depends on
                        the number of parameters: single parameters separated
                        by 1 single space. A single parameter can be a scalar,
                        a vector or a a matrix. Inside a vector, elements are
                        separated by 1 single space. Inside a matrix, rows are
                        representd as a (transpose) vector and inserted
                        separated by 1 single space). If None, standard
                        distribution parameters ranges are used. Adaptive
                        domain randomization, standard distribution parameters
                        low ranges. The meaning of the parameter depends on
                        the distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters low
                        ranges, U(a, b), lower range for a = -1, lower range
                        for b = -1; U(a, b), lower range for a =
                        float('-inf'), lower range for b = float('-inf'); U(a,
                        b), lower range for a = float('-inf'), lower range for
                        b = float('-inf'); U(a, b), lower range for a =
                        float('-inf'), lower range for b = float('-inf'); U(a,
                        b), lower range for a = -1, lower range for b = -1;
                        U(a, b), lower range for a = 0, lower range for b = 0;
                        U(a, b), lower range for a = 0, lower range for b = 0
                        Linearized version: Given lowers=[-1, float(-inf),
                        float(-inf), float(-inf), -1, 0, 0], uppers= [-1,
                        float(-inf), float(-inf), float(-inf), -1, 0, 0] -> -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -
                        Univariate triangular distribution,
                        Tr(lower_brightness, mode_brightness,
                        upper_brightness); Tr(lower_contrast, mode_contrast,
                        upper_contrast); Tr(lower_horizontal_flip,
                        mode_horizontal_flip, upper_horizontal_flip);
                        Tr(lower_vertical_flip, mode_vertical_flip,
                        upper_vertical_flip); Tr(lower_hue, mode_hue,
                        upper_hue); Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters low ranges, Tr(a, m, b), lower
                        range for a = -1, lower range for m = -1, lower range
                        for b = -1; Tr(a, m, b), lower range for a =
                        float('-inf'), lower range for m = float('-inf'),
                        lower range for b = float('-inf'); Tr(a, m, b), lower
                        range for a = float('-inf'), lower range for m =
                        float('-inf'), lower range for b = float('-inf');
                        Tr(a, m, b), lower range for a = float('-inf'), lower
                        range for m = float('-inf'), lower range for b =
                        float('-inf'); Tr(a, m, b), lower range for a = -1,
                        lower range for m = -1, lower range for b = -1; Tr(a,
                        m, b), lower range for a = 0, lower range for m = 0,
                        lower range for b = 0; Tr(a, m, b), lower range for a
                        = 0, lower range for m = 0, lower range for b = 0
                        Linearized version: Given lowers=[-1, float(-inf),
                        float(-inf), float(-inf), -1, 0, 0], modes= [-1,
                        float(-inf), float(-inf), float(-inf), -1, 0, 0],
                        uppers[-1, float(-inf), float(-inf), float(-inf), -1,
                        0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0
                        -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -
                        Univariate normal distribution, N(mean_brightness,
                        variance_brightness); N(mean_contrast,
                        variance_contrast); N(mean_horizontal_flip,
                        variance_horizontal_flip); N(mean_vertical_flip,
                        variance_vertical_flip); N(mean_hue, variance_hue);
                        N(mean_jpeg_quality, variance_jpeg_quality);
                        N(mean_saturation, variance_saturation) standard
                        distribution parameters low ranges, N(mu, sigma),
                        lower range for mu = -1, lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = float('-inf'),
                        lower range for sigma = 0; N(mu, sigma), lower range
                        for mu = float('-inf'), lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = float('-inf'),
                        lower range for sigma = 0; N(mu, sigma), lower range
                        for mu = -1, lower range for sigma = 0; N(mu, sigma),
                        lower range for mu = 0, lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = 0, lower range for
                        sigma = 0 Linearized version: Given means=[-1,
                        float('-inf'), float('-inf'), float('-inf'), -1, 0,
                        0], variances=[0, 0, 0, 0, 0, 0, 0] -> -1
                        float('-inf') float('-inf') float('-inf') -1 0 0 0 0 0
                        0 0 0 0 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters low ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), lower range for mu_1 = -1, lower range
                        for mu_2 = float('-inf'), lower range for mu_3 =
                        float('-inf'), lower range for mu_4 = float('-inf'),
                        lower range for mu_5 = -1, lower range for mu_6 = 0,
                        lower range for mu_7 = 0, lower range for sigma_ij = 0
                        if i = j, lower range for sigma_ij = -100 if i != j
                        Linearized version: Given mean_vector=[-1,
                        float('-inf'), float('-inf'), float('-inf'), -1, 0,
                        0], variance_covariance_matrix=[[0], [-100, 0], [-100,
                        -100, 0], [-100, -100, -100, 0], [-100, -100, -100,
                        -100, 0], [-100, -100, -100, -100, -100, 0], [-100,
                        -100, -100, -100, -100, -100, 0]] -> -1 float('-inf')
                        float('-inf') float('-inf') -1 0 0 0 -100 0 -100 -100
                        0 -100 -100 -100 0 -100 -100 -100 -100 0 -100 -100
                        -100 -100 -100 0 -100 -100 -100 -100 -100 -100 0 --
                        triangular matrix required -- (default: None)
  --adaptive_dom_rand__dist_ranges___triangular____up ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___TRIANGULAR____UP ...]
                        Adaptive domain randomization triangular distribution
                        parameters up ranges. Use the linearized version (i.e.
                        Parameters list, whose cardinality depends on the
                        number of parameters: single parameters separated by 1
                        single space. A single parameter can be a scalar, a
                        vector or a a matrix. Inside a vector, elements are
                        separated by 1 single space. Inside a matrix, rows are
                        representd as a (transpose) vector and inserted
                        separated by 1 single space). If None, standard
                        distribution parameters ranges are used. Adaptive
                        domain randomization, standard distribution parameters
                        up ranges. The meaning of the parameter depends on the
                        distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters up
                        ranges, U(a, b), upper range for a = 1, upper range
                        for b = 1; U(a, b), upper range for a = float('inf'),
                        upper range for b = float('inf'); U(a, b), upper range
                        for a = float('inf'), upper range for b =
                        float('inf'); U(a, b), upper range for a =
                        float('inf'), upper range for b = float('inf'); U(a,
                        b), upper range for a = 1, upper range for b = 1; U(a,
                        b), upper range for a = 100, upper range for b = 100;
                        U(a, b), upper range for a = float('inf'), upper range
                        for b = float('inf') Linearized version: Given
                        lowers=[1, float(inf), float(inf), float(inf), 1, 100,
                        float(inf)], uppers= [1, float(inf), float(inf),
                        float(inf), 1, 100, float(inf)] -> 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters up ranges, Tr(a, m, b), upper
                        range for a = 1, upper range for m = 1, upper range
                        for b = 1; Tr(a, m, b), upper range for a =
                        float('inf'), upper range for m = float('inf'), upper
                        range for b = float('inf'); Tr(a, m, b), upper range
                        for a = float('inf'), upper range for m =
                        float('inf'), upper range for b = float('inf'); Tr(a,
                        m, b), upper range for a = float('inf'), upper range
                        for m = float('inf'), upper range for b =
                        float('inf'); Tr(a, m, b), upper range for a = 1,
                        upper range for m = 1, upper range for b = 1; Tr(a, m,
                        b), upper range for a = 100, upper range for m = 100,
                        upper range for b = 100; Tr(a, m, b), upper range for
                        a = float('inf'), upper range for m = float('inf'),
                        upper range for b = float('inf') Linearized version:
                        Given lowers=[1, float(inf), float(inf), float(inf),
                        1, 100, float(inf)], modes= [1, float(inf),
                        float(inf), float(inf), 1, 100, float(inf)], uppers[1,
                        float(inf), float(inf), float(inf), 1, 100,
                        float(inf)] -> 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        up ranges, N(mu, sigma), upper range for mu = 1, upper
                        range for sigma = 0.4; N(mu, sigma), upper range for
                        mu = float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = float('inf'), upper range
                        for sigma = 10; N(mu, sigma), upper range for mu =
                        float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = 1, upper range for sigma
                        = 0.4; N(mu, sigma), upper range for mu = 100, upper
                        range for sigma = 25; N(mu, sigma), upper range for mu
                        = float('inf'), upper range for sigma = 10 Linearized
                        version: Given means=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')], variances=[0.4,
                        10, 10, 10, 0.4, 25, 10] -> 1 float('inf')
                        float('inf') float('inf') 1 100 float('inf') 0.4 10 10
                        10 0.4 25 10 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters up ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), upper range for mu_1 = 1, upper range
                        for mu_2 = float('inf'), upper range for mu_3 =
                        float('inf'), upper range for mu_4 = float('inf'),
                        upper range for mu_5 = 1, upper range for mu_6 = 100,
                        upper range for mu_7 = float('inf'), upper range for
                        sigma_ij = 0.4 if i = j = {1, 5}, upper range for
                        sigma_ij = 10 if i = j = {2, 3, 4, 7}, upper range for
                        sigma_ij = 25 if i = j = 6, upper range for sigma_ij =
                        100 if i != j Linearized version: Given
                        mean_vector=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')],
                        variance_covariance_matrix=[[0.4], [100, 10], [100,
                        100, 10], [100, 100, 100, 10], [100, 100, 100, 100,
                        0.4], [100, 100, 100, 100, 100, 25], [100, 100, 100,
                        100, 100, 100, 10]] -> 1 float('inf') float('inf')
                        float('inf') 1 100 float('inf') 0.4 100 10 100 100 10
                        100 100 100 10 100 100 100 100 0.4 100 100 100 100 100
                        25 100 100 100 100 100 100 10 -- triangular matrix
                        required -- (default: None)
  --adaptive_dom_rand__dist_ranges___univariatenormal____low ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____LOW ...]
                        Adaptive domain randomization univatiate normal
                        distribution parameters low ranges. Use the linearized
                        version (i.e. Parameters list, whose cardinality
                        depends on the number of parameters: single parameters
                        separated by 1 single space. A single parameter can be
                        a scalar, a vector or a a matrix. Inside a vector,
                        elements are separated by 1 single space. Inside a
                        matrix, rows are representd as a (transpose) vector
                        and inserted separated by 1 single space). If None,
                        standard distribution parameters ranges are used.
                        Adaptive domain randomization, standard distribution
                        parameters low ranges. The meaning of the parameter
                        depends on the distribution(s) considered. If
                        univariate distributions are used, the order reflects
                        the corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters low
                        ranges, U(a, b), lower range for a = -1, lower range
                        for b = -1; U(a, b), lower range for a =
                        float('-inf'), lower range for b = float('-inf'); U(a,
                        b), lower range for a = float('-inf'), lower range for
                        b = float('-inf'); U(a, b), lower range for a =
                        float('-inf'), lower range for b = float('-inf'); U(a,
                        b), lower range for a = -1, lower range for b = -1;
                        U(a, b), lower range for a = 0, lower range for b = 0;
                        U(a, b), lower range for a = 0, lower range for b = 0
                        Linearized version: Given lowers=[-1, float(-inf),
                        float(-inf), float(-inf), -1, 0, 0], uppers= [-1,
                        float(-inf), float(-inf), float(-inf), -1, 0, 0] -> -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -
                        Univariate triangular distribution,
                        Tr(lower_brightness, mode_brightness,
                        upper_brightness); Tr(lower_contrast, mode_contrast,
                        upper_contrast); Tr(lower_horizontal_flip,
                        mode_horizontal_flip, upper_horizontal_flip);
                        Tr(lower_vertical_flip, mode_vertical_flip,
                        upper_vertical_flip); Tr(lower_hue, mode_hue,
                        upper_hue); Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters low ranges, Tr(a, m, b), lower
                        range for a = -1, lower range for m = -1, lower range
                        for b = -1; Tr(a, m, b), lower range for a =
                        float('-inf'), lower range for m = float('-inf'),
                        lower range for b = float('-inf'); Tr(a, m, b), lower
                        range for a = float('-inf'), lower range for m =
                        float('-inf'), lower range for b = float('-inf');
                        Tr(a, m, b), lower range for a = float('-inf'), lower
                        range for m = float('-inf'), lower range for b =
                        float('-inf'); Tr(a, m, b), lower range for a = -1,
                        lower range for m = -1, lower range for b = -1; Tr(a,
                        m, b), lower range for a = 0, lower range for m = 0,
                        lower range for b = 0; Tr(a, m, b), lower range for a
                        = 0, lower range for m = 0, lower range for b = 0
                        Linearized version: Given lowers=[-1, float(-inf),
                        float(-inf), float(-inf), -1, 0, 0], modes= [-1,
                        float(-inf), float(-inf), float(-inf), -1, 0, 0],
                        uppers[-1, float(-inf), float(-inf), float(-inf), -1,
                        0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0
                        -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -
                        Univariate normal distribution, N(mean_brightness,
                        variance_brightness); N(mean_contrast,
                        variance_contrast); N(mean_horizontal_flip,
                        variance_horizontal_flip); N(mean_vertical_flip,
                        variance_vertical_flip); N(mean_hue, variance_hue);
                        N(mean_jpeg_quality, variance_jpeg_quality);
                        N(mean_saturation, variance_saturation) standard
                        distribution parameters low ranges, N(mu, sigma),
                        lower range for mu = -1, lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = float('-inf'),
                        lower range for sigma = 0; N(mu, sigma), lower range
                        for mu = float('-inf'), lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = float('-inf'),
                        lower range for sigma = 0; N(mu, sigma), lower range
                        for mu = -1, lower range for sigma = 0; N(mu, sigma),
                        lower range for mu = 0, lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = 0, lower range for
                        sigma = 0 Linearized version: Given means=[-1,
                        float('-inf'), float('-inf'), float('-inf'), -1, 0,
                        0], variances=[0, 0, 0, 0, 0, 0, 0] -> -1
                        float('-inf') float('-inf') float('-inf') -1 0 0 0 0 0
                        0 0 0 0 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters low ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), lower range for mu_1 = -1, lower range
                        for mu_2 = float('-inf'), lower range for mu_3 =
                        float('-inf'), lower range for mu_4 = float('-inf'),
                        lower range for mu_5 = -1, lower range for mu_6 = 0,
                        lower range for mu_7 = 0, lower range for sigma_ij = 0
                        if i = j, lower range for sigma_ij = -100 if i != j
                        Linearized version: Given mean_vector=[-1,
                        float('-inf'), float('-inf'), float('-inf'), -1, 0,
                        0], variance_covariance_matrix=[[0], [-100, 0], [-100,
                        -100, 0], [-100, -100, -100, 0], [-100, -100, -100,
                        -100, 0], [-100, -100, -100, -100, -100, 0], [-100,
                        -100, -100, -100, -100, -100, 0]] -> -1 float('-inf')
                        float('-inf') float('-inf') -1 0 0 0 -100 0 -100 -100
                        0 -100 -100 -100 0 -100 -100 -100 -100 0 -100 -100
                        -100 -100 -100 0 -100 -100 -100 -100 -100 -100 0 --
                        triangular matrix required -- (default: None)
  --adaptive_dom_rand__dist_ranges___univariatenormal____up ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___UNIVARIATENORMAL____UP ...]
                        Adaptive domain randomization univatiate normal
                        distribution parameters up ranges. Use the linearized
                        version (i.e. Parameters list, whose cardinality
                        depends on the number of parameters: single parameters
                        separated by 1 single space. A single parameter can be
                        a scalar, a vector or a a matrix. Inside a vector,
                        elements are separated by 1 single space. Inside a
                        matrix, rows are representd as a (transpose) vector
                        and inserted separated by 1 single space). If None,
                        standard distribution parameters ranges are used.
                        Adaptive domain randomization, standard distribution
                        parameters up ranges. The meaning of the parameter
                        depends on the distribution(s) considered. If
                        univariate distributions are used, the order reflects
                        the corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters up
                        ranges, U(a, b), upper range for a = 1, upper range
                        for b = 1; U(a, b), upper range for a = float('inf'),
                        upper range for b = float('inf'); U(a, b), upper range
                        for a = float('inf'), upper range for b =
                        float('inf'); U(a, b), upper range for a =
                        float('inf'), upper range for b = float('inf'); U(a,
                        b), upper range for a = 1, upper range for b = 1; U(a,
                        b), upper range for a = 100, upper range for b = 100;
                        U(a, b), upper range for a = float('inf'), upper range
                        for b = float('inf') Linearized version: Given
                        lowers=[1, float(inf), float(inf), float(inf), 1, 100,
                        float(inf)], uppers= [1, float(inf), float(inf),
                        float(inf), 1, 100, float(inf)] -> 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters up ranges, Tr(a, m, b), upper
                        range for a = 1, upper range for m = 1, upper range
                        for b = 1; Tr(a, m, b), upper range for a =
                        float('inf'), upper range for m = float('inf'), upper
                        range for b = float('inf'); Tr(a, m, b), upper range
                        for a = float('inf'), upper range for m =
                        float('inf'), upper range for b = float('inf'); Tr(a,
                        m, b), upper range for a = float('inf'), upper range
                        for m = float('inf'), upper range for b =
                        float('inf'); Tr(a, m, b), upper range for a = 1,
                        upper range for m = 1, upper range for b = 1; Tr(a, m,
                        b), upper range for a = 100, upper range for m = 100,
                        upper range for b = 100; Tr(a, m, b), upper range for
                        a = float('inf'), upper range for m = float('inf'),
                        upper range for b = float('inf') Linearized version:
                        Given lowers=[1, float(inf), float(inf), float(inf),
                        1, 100, float(inf)], modes= [1, float(inf),
                        float(inf), float(inf), 1, 100, float(inf)], uppers[1,
                        float(inf), float(inf), float(inf), 1, 100,
                        float(inf)] -> 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        up ranges, N(mu, sigma), upper range for mu = 1, upper
                        range for sigma = 0.4; N(mu, sigma), upper range for
                        mu = float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = float('inf'), upper range
                        for sigma = 10; N(mu, sigma), upper range for mu =
                        float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = 1, upper range for sigma
                        = 0.4; N(mu, sigma), upper range for mu = 100, upper
                        range for sigma = 25; N(mu, sigma), upper range for mu
                        = float('inf'), upper range for sigma = 10 Linearized
                        version: Given means=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')], variances=[0.4,
                        10, 10, 10, 0.4, 25, 10] -> 1 float('inf')
                        float('inf') float('inf') 1 100 float('inf') 0.4 10 10
                        10 0.4 25 10 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters up ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), upper range for mu_1 = 1, upper range
                        for mu_2 = float('inf'), upper range for mu_3 =
                        float('inf'), upper range for mu_4 = float('inf'),
                        upper range for mu_5 = 1, upper range for mu_6 = 100,
                        upper range for mu_7 = float('inf'), upper range for
                        sigma_ij = 0.4 if i = j = {1, 5}, upper range for
                        sigma_ij = 10 if i = j = {2, 3, 4, 7}, upper range for
                        sigma_ij = 25 if i = j = 6, upper range for sigma_ij =
                        100 if i != j Linearized version: Given
                        mean_vector=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')],
                        variance_covariance_matrix=[[0.4], [100, 10], [100,
                        100, 10], [100, 100, 100, 10], [100, 100, 100, 100,
                        0.4], [100, 100, 100, 100, 100, 25], [100, 100, 100,
                        100, 100, 100, 10]] -> 1 float('inf') float('inf')
                        float('inf') 1 100 float('inf') 0.4 100 10 100 100 10
                        100 100 100 10 100 100 100 100 0.4 100 100 100 100 100
                        25 100 100 100 100 100 100 10 -- triangular matrix
                        required -- (default: None)
  --adaptive_dom_rand__dist_ranges___multivariatenormal____low ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____LOW [ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____LOW ...]
                        Adaptive domain randomization multivatiate normal
                        distribution parameters low ranges. Use the linearized
                        version (i.e. Parameters list, whose cardinality
                        depends on the number of parameters: single parameters
                        separated by 1 single space. A single parameter can be
                        a scalar, a vector or a a matrix. Inside a vector,
                        elements are separated by 1 single space. Inside a
                        matrix, rows are representd as a (transpose) vector
                        and inserted separated by 1 single space). If None,
                        standard distribution parameters ranges are used.
                        Adaptive domain randomization, standard distribution
                        parameters low ranges. The meaning of the parameter
                        depends on the distribution(s) considered. If
                        univariate distributions are used, the order reflects
                        the corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters low
                        ranges, U(a, b), lower range for a = -1, lower range
                        for b = -1; U(a, b), lower range for a =
                        float('-inf'), lower range for b = float('-inf'); U(a,
                        b), lower range for a = float('-inf'), lower range for
                        b = float('-inf'); U(a, b), lower range for a =
                        float('-inf'), lower range for b = float('-inf'); U(a,
                        b), lower range for a = -1, lower range for b = -1;
                        U(a, b), lower range for a = 0, lower range for b = 0;
                        U(a, b), lower range for a = 0, lower range for b = 0
                        Linearized version: Given lowers=[-1, float(-inf),
                        float(-inf), float(-inf), -1, 0, 0], uppers= [-1,
                        float(-inf), float(-inf), float(-inf), -1, 0, 0] -> -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -
                        Univariate triangular distribution,
                        Tr(lower_brightness, mode_brightness,
                        upper_brightness); Tr(lower_contrast, mode_contrast,
                        upper_contrast); Tr(lower_horizontal_flip,
                        mode_horizontal_flip, upper_horizontal_flip);
                        Tr(lower_vertical_flip, mode_vertical_flip,
                        upper_vertical_flip); Tr(lower_hue, mode_hue,
                        upper_hue); Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters low ranges, Tr(a, m, b), lower
                        range for a = -1, lower range for m = -1, lower range
                        for b = -1; Tr(a, m, b), lower range for a =
                        float('-inf'), lower range for m = float('-inf'),
                        lower range for b = float('-inf'); Tr(a, m, b), lower
                        range for a = float('-inf'), lower range for m =
                        float('-inf'), lower range for b = float('-inf');
                        Tr(a, m, b), lower range for a = float('-inf'), lower
                        range for m = float('-inf'), lower range for b =
                        float('-inf'); Tr(a, m, b), lower range for a = -1,
                        lower range for m = -1, lower range for b = -1; Tr(a,
                        m, b), lower range for a = 0, lower range for m = 0,
                        lower range for b = 0; Tr(a, m, b), lower range for a
                        = 0, lower range for m = 0, lower range for b = 0
                        Linearized version: Given lowers=[-1, float(-inf),
                        float(-inf), float(-inf), -1, 0, 0], modes= [-1,
                        float(-inf), float(-inf), float(-inf), -1, 0, 0],
                        uppers[-1, float(-inf), float(-inf), float(-inf), -1,
                        0, 0] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0
                        -1 float(-inf) float(-inf) float(-inf) -1 0 0 -1
                        float(-inf) float(-inf) float(-inf) -1 0 0 -
                        Univariate normal distribution, N(mean_brightness,
                        variance_brightness); N(mean_contrast,
                        variance_contrast); N(mean_horizontal_flip,
                        variance_horizontal_flip); N(mean_vertical_flip,
                        variance_vertical_flip); N(mean_hue, variance_hue);
                        N(mean_jpeg_quality, variance_jpeg_quality);
                        N(mean_saturation, variance_saturation) standard
                        distribution parameters low ranges, N(mu, sigma),
                        lower range for mu = -1, lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = float('-inf'),
                        lower range for sigma = 0; N(mu, sigma), lower range
                        for mu = float('-inf'), lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = float('-inf'),
                        lower range for sigma = 0; N(mu, sigma), lower range
                        for mu = -1, lower range for sigma = 0; N(mu, sigma),
                        lower range for mu = 0, lower range for sigma = 0;
                        N(mu, sigma), lower range for mu = 0, lower range for
                        sigma = 0 Linearized version: Given means=[-1,
                        float('-inf'), float('-inf'), float('-inf'), -1, 0,
                        0], variances=[0, 0, 0, 0, 0, 0, 0] -> -1
                        float('-inf') float('-inf') float('-inf') -1 0 0 0 0 0
                        0 0 0 0 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters low ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), lower range for mu_1 = -1, lower range
                        for mu_2 = float('-inf'), lower range for mu_3 =
                        float('-inf'), lower range for mu_4 = float('-inf'),
                        lower range for mu_5 = -1, lower range for mu_6 = 0,
                        lower range for mu_7 = 0, lower range for sigma_ij = 0
                        if i = j, lower range for sigma_ij = -100 if i != j
                        Linearized version: Given mean_vector=[-1,
                        float('-inf'), float('-inf'), float('-inf'), -1, 0,
                        0], variance_covariance_matrix=[[0], [-100, 0], [-100,
                        -100, 0], [-100, -100, -100, 0], [-100, -100, -100,
                        -100, 0], [-100, -100, -100, -100, -100, 0], [-100,
                        -100, -100, -100, -100, -100, 0]] -> -1 float('-inf')
                        float('-inf') float('-inf') -1 0 0 0 -100 0 -100 -100
                        0 -100 -100 -100 0 -100 -100 -100 -100 0 -100 -100
                        -100 -100 -100 0 -100 -100 -100 -100 -100 -100 0 --
                        triangular matrix required -- (default: None)
  --adaptive_dom_rand__dist_ranges___multivariatenormal____up ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____UP [ADAPTIVE_DOM_RAND__DIST_RANGES___MULTIVARIATENORMAL____UP ...]
                        Adaptive domain randomization multivatiate normal
                        distribution parameters up ranges. Use the linearized
                        version (i.e. Parameters list, whose cardinality
                        depends on the number of parameters: single parameters
                        separated by 1 single space. A single parameter can be
                        a scalar, a vector or a a matrix. Inside a vector,
                        elements are separated by 1 single space. Inside a
                        matrix, rows are representd as a (transpose) vector
                        and inserted separated by 1 single space). If None,
                        standard distribution parameters ranges are used.
                        Adaptive domain randomization, standard distribution
                        parameters up ranges. The meaning of the parameter
                        depends on the distribution(s) considered. If
                        univariate distributions are used, the order reflects
                        the corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters up
                        ranges, U(a, b), upper range for a = 1, upper range
                        for b = 1; U(a, b), upper range for a = float('inf'),
                        upper range for b = float('inf'); U(a, b), upper range
                        for a = float('inf'), upper range for b =
                        float('inf'); U(a, b), upper range for a =
                        float('inf'), upper range for b = float('inf'); U(a,
                        b), upper range for a = 1, upper range for b = 1; U(a,
                        b), upper range for a = 100, upper range for b = 100;
                        U(a, b), upper range for a = float('inf'), upper range
                        for b = float('inf') Linearized version: Given
                        lowers=[1, float(inf), float(inf), float(inf), 1, 100,
                        float(inf)], uppers= [1, float(inf), float(inf),
                        float(inf), 1, 100, float(inf)] -> 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) 1 float(inf)
                        float(inf) float(inf) 1 100 float(inf) - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters up ranges, Tr(a, m, b), upper
                        range for a = 1, upper range for m = 1, upper range
                        for b = 1; Tr(a, m, b), upper range for a =
                        float('inf'), upper range for m = float('inf'), upper
                        range for b = float('inf'); Tr(a, m, b), upper range
                        for a = float('inf'), upper range for m =
                        float('inf'), upper range for b = float('inf'); Tr(a,
                        m, b), upper range for a = float('inf'), upper range
                        for m = float('inf'), upper range for b =
                        float('inf'); Tr(a, m, b), upper range for a = 1,
                        upper range for m = 1, upper range for b = 1; Tr(a, m,
                        b), upper range for a = 100, upper range for m = 100,
                        upper range for b = 100; Tr(a, m, b), upper range for
                        a = float('inf'), upper range for m = float('inf'),
                        upper range for b = float('inf') Linearized version:
                        Given lowers=[1, float(inf), float(inf), float(inf),
                        1, 100, float(inf)], modes= [1, float(inf),
                        float(inf), float(inf), 1, 100, float(inf)], uppers[1,
                        float(inf), float(inf), float(inf), 1, 100,
                        float(inf)] -> 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) 1 float(inf) float(inf) float(inf) 1
                        100 float(inf) - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        up ranges, N(mu, sigma), upper range for mu = 1, upper
                        range for sigma = 0.4; N(mu, sigma), upper range for
                        mu = float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = float('inf'), upper range
                        for sigma = 10; N(mu, sigma), upper range for mu =
                        float('inf'), upper range for sigma = 10; N(mu,
                        sigma), upper range for mu = 1, upper range for sigma
                        = 0.4; N(mu, sigma), upper range for mu = 100, upper
                        range for sigma = 25; N(mu, sigma), upper range for mu
                        = float('inf'), upper range for sigma = 10 Linearized
                        version: Given means=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')], variances=[0.4,
                        10, 10, 10, 0.4, 25, 10] -> 1 float('inf')
                        float('inf') float('inf') 1 100 float('inf') 0.4 10 10
                        10 0.4 25 10 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters up ranges, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), upper range for mu_1 = 1, upper range
                        for mu_2 = float('inf'), upper range for mu_3 =
                        float('inf'), upper range for mu_4 = float('inf'),
                        upper range for mu_5 = 1, upper range for mu_6 = 100,
                        upper range for mu_7 = float('inf'), upper range for
                        sigma_ij = 0.4 if i = j = {1, 5}, upper range for
                        sigma_ij = 10 if i = j = {2, 3, 4, 7}, upper range for
                        sigma_ij = 25 if i = j = 6, upper range for sigma_ij =
                        100 if i != j Linearized version: Given
                        mean_vector=[1, float('inf'), float('inf'),
                        float('inf'), 1, 100, float('inf')],
                        variance_covariance_matrix=[[0.4], [100, 10], [100,
                        100, 10], [100, 100, 100, 10], [100, 100, 100, 100,
                        0.4], [100, 100, 100, 100, 100, 25], [100, 100, 100,
                        100, 100, 100, 10]] -> 1 float('inf') float('inf')
                        float('inf') 1 100 float('inf') 0.4 100 10 100 100 10
                        100 100 100 10 100 100 100 100 0.4 100 100 100 100 100
                        25 100 100 100 100 100 100 10 -- triangular matrix
                        required -- (default: None)
  --adaptive_dom_rand__dist_initials ADAPTIVE_DOM_RAND__DIST_INITIALS [ADAPTIVE_DOM_RAND__DIST_INITIALS ...]
                        Adaptive domain randomization distribution parameters
                        initial values. Use the linearized version (i.e.
                        Parameters list, whose cardinality depends on the
                        number of parameters: single parameters separated by 1
                        single space. A single parameter can be a scalar, a
                        vector or a a matrix. Inside a vector, elements are
                        separated by 1 single space. Inside a matrix, rows are
                        representd as a (transpose) vector and inserted
                        separated by 1 single space). If None, standard
                        distribution parameters ranges are used. Adaptive
                        domain randomization, standard distribution parameters
                        initial values. The meaning of the parameter depends
                        on the distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters
                        initial values, U(a, b), initial value for a = 0,
                        initial value for b = 0; U(a, b), initial value for a
                        = 0, initial value for b = 0; U(a, b), initial value
                        for a = 0, initial value for b = 0; U(a, b), initial
                        value for a = 0, initial value for b = 0; U(a, b),
                        initial value for a = 0, initial value for b = 0; U(a,
                        b), initial value for a = 50, initial value for b =
                        50; U(a, b), initial value for a = 1.25, initial value
                        for b = 1.25 Linearized version: Given lowers=[0, 0,
                        0, 0, 0, 50, 1.25], uppers= [0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters initial values, Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 0, initial
                        value for m = 0, initial value for b = 0; Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 50, initial
                        value for m = 50, initial value for b = 50; Tr(a, m,
                        b), initial value for a = 1.25, initial value for m =
                        1.25, initial value for b = 1.25 Linearized version:
                        Given lowers=[0, 0, 0, 0, 0, 50, 1.25], modes= [0, 0,
                        0, 0, 0, 50, 1.25], uppers=[0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 0 0 0 0 0 50
                        1.25 - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        initial values, N(mu, sigma), initial value for mu =
                        0, initial value for sigma = 0.2; N(mu, sigma),
                        initial value for mu = 0, initial value for sigma = 5;
                        N(mu, sigma), initial value for mu = 0, initial value
                        for sigma = 5; N(mu, sigma), initial value for mu = 0,
                        initial value for sigma = 5; N(mu, sigma), initial
                        value for mu = 0, initial value for sigma = 0.2; N(mu,
                        sigma), initial value for mu = 50, initial value for
                        sigma = 12.5; N(mu, sigma), initial value for mu =
                        1.25, initial value for sigma = 5 Linearized version:
                        Given means=[0, 0, 0, 0, 0, 50, 1.25], variances=[0.2,
                        5, 5, 5, 0.2, 12.5, 5] -> 0 0 0 0 0 50 1.25 0.2 5 5 5
                        0.2 12.5 5 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters initial values, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), initial value for mu_1 = 0, initial
                        value for mu_2 = 0, initial value for mu_3 = 0,
                        initial value for mu_4 = 0, initial value for mu_5 =
                        0, initial value for mu_6 = 1.25, initial value for
                        mu_7 = 50, initial value for sigma_ij = 0.2 if i = j =
                        {1,5}, initial value for sigma_ij = 5 if i = j =
                        {2,3,4,7}, initial value for sigma_ij = 12.5 if i = j
                        = 6, initial value for sigma_ij = 0 if i != j
                        Linearized version: Given mean_vector=[0, 0, 0, 0, 0,
                        1.25, 50], variance_covariance_matrix=[[0.2], [0, 5],
                        [0, 0, 5], [0, 0, 0, 5], [0, 0, 0, 0, 0.2], [0, 0, 0,
                        0, 0, 12.5], [0, 0, 0, 0, 0, 0, 5]] -> 0 0 0 0 0 1.25
                        500.2 0 5 0 0 5 0 0 0 5 0 0 0 0 0.2 0 0 0 0 0 12.5 0 0
                        0 0 0 0 5 -- triangular matrix required -- (default:
                        None)
  --adaptive_dom_rand__dist_initials___uniform ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIFORM [ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIFORM ...]
                        Adaptive domain randomization uniform distribution
                        parameters initial values. Use the linearized version
                        (i.e. Parameters list, whose cardinality depends on
                        the number of parameters: single parameters separated
                        by 1 single space. A single parameter can be a scalar,
                        a vector or a a matrix. Inside a vector, elements are
                        separated by 1 single space. Inside a matrix, rows are
                        representd as a (transpose) vector and inserted
                        separated by 1 single space). If None, standard
                        distribution parameters ranges are used. Adaptive
                        domain randomization, standard distribution parameters
                        initial values. The meaning of the parameter depends
                        on the distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters
                        initial values, U(a, b), initial value for a = 0,
                        initial value for b = 0; U(a, b), initial value for a
                        = 0, initial value for b = 0; U(a, b), initial value
                        for a = 0, initial value for b = 0; U(a, b), initial
                        value for a = 0, initial value for b = 0; U(a, b),
                        initial value for a = 0, initial value for b = 0; U(a,
                        b), initial value for a = 50, initial value for b =
                        50; U(a, b), initial value for a = 1.25, initial value
                        for b = 1.25 Linearized version: Given lowers=[0, 0,
                        0, 0, 0, 50, 1.25], uppers= [0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters initial values, Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 0, initial
                        value for m = 0, initial value for b = 0; Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 50, initial
                        value for m = 50, initial value for b = 50; Tr(a, m,
                        b), initial value for a = 1.25, initial value for m =
                        1.25, initial value for b = 1.25 Linearized version:
                        Given lowers=[0, 0, 0, 0, 0, 50, 1.25], modes= [0, 0,
                        0, 0, 0, 50, 1.25], uppers=[0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 0 0 0 0 0 50
                        1.25 - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        initial values, N(mu, sigma), initial value for mu =
                        0, initial value for sigma = 0.2; N(mu, sigma),
                        initial value for mu = 0, initial value for sigma = 5;
                        N(mu, sigma), initial value for mu = 0, initial value
                        for sigma = 5; N(mu, sigma), initial value for mu = 0,
                        initial value for sigma = 5; N(mu, sigma), initial
                        value for mu = 0, initial value for sigma = 0.2; N(mu,
                        sigma), initial value for mu = 50, initial value for
                        sigma = 12.5; N(mu, sigma), initial value for mu =
                        1.25, initial value for sigma = 5 Linearized version:
                        Given means=[0, 0, 0, 0, 0, 50, 1.25], variances=[0.2,
                        5, 5, 5, 0.2, 12.5, 5] -> 0 0 0 0 0 50 1.25 0.2 5 5 5
                        0.2 12.5 5 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters initial values, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), initial value for mu_1 = 0, initial
                        value for mu_2 = 0, initial value for mu_3 = 0,
                        initial value for mu_4 = 0, initial value for mu_5 =
                        0, initial value for mu_6 = 1.25, initial value for
                        mu_7 = 50, initial value for sigma_ij = 0.2 if i = j =
                        {1,5}, initial value for sigma_ij = 5 if i = j =
                        {2,3,4,7}, initial value for sigma_ij = 12.5 if i = j
                        = 6, initial value for sigma_ij = 0 if i != j
                        Linearized version: Given mean_vector=[0, 0, 0, 0, 0,
                        1.25, 50], variance_covariance_matrix=[[0.2], [0, 5],
                        [0, 0, 5], [0, 0, 0, 5], [0, 0, 0, 0, 0.2], [0, 0, 0,
                        0, 0, 12.5], [0, 0, 0, 0, 0, 0, 5]] -> 0 0 0 0 0 1.25
                        500.2 0 5 0 0 5 0 0 0 5 0 0 0 0 0.2 0 0 0 0 0 12.5 0 0
                        0 0 0 0 5 -- triangular matrix required -- (default:
                        None)
  --adaptive_dom_rand__dist_initials___triangular ADAPTIVE_DOM_RAND__DIST_INITIALS___TRIANGULAR [ADAPTIVE_DOM_RAND__DIST_INITIALS___TRIANGULAR ...]
                        Adaptive domain randomization triangular distribution
                        parameters initial values. Use the linearized version
                        (i.e. Parameters list, whose cardinality depends on
                        the number of parameters: single parameters separated
                        by 1 single space. A single parameter can be a scalar,
                        a vector or a a matrix. Inside a vector, elements are
                        separated by 1 single space. Inside a matrix, rows are
                        representd as a (transpose) vector and inserted
                        separated by 1 single space). If None, standard
                        distribution parameters ranges are used. Adaptive
                        domain randomization, standard distribution parameters
                        initial values. The meaning of the parameter depends
                        on the distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters
                        initial values, U(a, b), initial value for a = 0,
                        initial value for b = 0; U(a, b), initial value for a
                        = 0, initial value for b = 0; U(a, b), initial value
                        for a = 0, initial value for b = 0; U(a, b), initial
                        value for a = 0, initial value for b = 0; U(a, b),
                        initial value for a = 0, initial value for b = 0; U(a,
                        b), initial value for a = 50, initial value for b =
                        50; U(a, b), initial value for a = 1.25, initial value
                        for b = 1.25 Linearized version: Given lowers=[0, 0,
                        0, 0, 0, 50, 1.25], uppers= [0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters initial values, Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 0, initial
                        value for m = 0, initial value for b = 0; Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 50, initial
                        value for m = 50, initial value for b = 50; Tr(a, m,
                        b), initial value for a = 1.25, initial value for m =
                        1.25, initial value for b = 1.25 Linearized version:
                        Given lowers=[0, 0, 0, 0, 0, 50, 1.25], modes= [0, 0,
                        0, 0, 0, 50, 1.25], uppers=[0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 0 0 0 0 0 50
                        1.25 - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        initial values, N(mu, sigma), initial value for mu =
                        0, initial value for sigma = 0.2; N(mu, sigma),
                        initial value for mu = 0, initial value for sigma = 5;
                        N(mu, sigma), initial value for mu = 0, initial value
                        for sigma = 5; N(mu, sigma), initial value for mu = 0,
                        initial value for sigma = 5; N(mu, sigma), initial
                        value for mu = 0, initial value for sigma = 0.2; N(mu,
                        sigma), initial value for mu = 50, initial value for
                        sigma = 12.5; N(mu, sigma), initial value for mu =
                        1.25, initial value for sigma = 5 Linearized version:
                        Given means=[0, 0, 0, 0, 0, 50, 1.25], variances=[0.2,
                        5, 5, 5, 0.2, 12.5, 5] -> 0 0 0 0 0 50 1.25 0.2 5 5 5
                        0.2 12.5 5 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters initial values, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), initial value for mu_1 = 0, initial
                        value for mu_2 = 0, initial value for mu_3 = 0,
                        initial value for mu_4 = 0, initial value for mu_5 =
                        0, initial value for mu_6 = 1.25, initial value for
                        mu_7 = 50, initial value for sigma_ij = 0.2 if i = j =
                        {1,5}, initial value for sigma_ij = 5 if i = j =
                        {2,3,4,7}, initial value for sigma_ij = 12.5 if i = j
                        = 6, initial value for sigma_ij = 0 if i != j
                        Linearized version: Given mean_vector=[0, 0, 0, 0, 0,
                        1.25, 50], variance_covariance_matrix=[[0.2], [0, 5],
                        [0, 0, 5], [0, 0, 0, 5], [0, 0, 0, 0, 0.2], [0, 0, 0,
                        0, 0, 12.5], [0, 0, 0, 0, 0, 0, 5]] -> 0 0 0 0 0 1.25
                        500.2 0 5 0 0 5 0 0 0 5 0 0 0 0 0.2 0 0 0 0 0 12.5 0 0
                        0 0 0 0 5 -- triangular matrix required -- (default:
                        None)
  --adaptive_dom_rand__dist_initials___univariatenormal ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIVARIATENORMAL [ADAPTIVE_DOM_RAND__DIST_INITIALS___UNIVARIATENORMAL ...]
                        Adaptive domain randomization univatiate normal
                        distribution parameters initial values. Use the
                        linearized version (i.e. Parameters list, whose
                        cardinality depends on the number of parameters:
                        single parameters separated by 1 single space. A
                        single parameter can be a scalar, a vector or a a
                        matrix. Inside a vector, elements are separated by 1
                        single space. Inside a matrix, rows are representd as
                        a (transpose) vector and inserted separated by 1
                        single space). If None, standard distribution
                        parameters ranges are used. Adaptive domain
                        randomization, standard distribution parameters
                        initial values. The meaning of the parameter depends
                        on the distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters
                        initial values, U(a, b), initial value for a = 0,
                        initial value for b = 0; U(a, b), initial value for a
                        = 0, initial value for b = 0; U(a, b), initial value
                        for a = 0, initial value for b = 0; U(a, b), initial
                        value for a = 0, initial value for b = 0; U(a, b),
                        initial value for a = 0, initial value for b = 0; U(a,
                        b), initial value for a = 50, initial value for b =
                        50; U(a, b), initial value for a = 1.25, initial value
                        for b = 1.25 Linearized version: Given lowers=[0, 0,
                        0, 0, 0, 50, 1.25], uppers= [0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters initial values, Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 0, initial
                        value for m = 0, initial value for b = 0; Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 50, initial
                        value for m = 50, initial value for b = 50; Tr(a, m,
                        b), initial value for a = 1.25, initial value for m =
                        1.25, initial value for b = 1.25 Linearized version:
                        Given lowers=[0, 0, 0, 0, 0, 50, 1.25], modes= [0, 0,
                        0, 0, 0, 50, 1.25], uppers=[0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 0 0 0 0 0 50
                        1.25 - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        initial values, N(mu, sigma), initial value for mu =
                        0, initial value for sigma = 0.2; N(mu, sigma),
                        initial value for mu = 0, initial value for sigma = 5;
                        N(mu, sigma), initial value for mu = 0, initial value
                        for sigma = 5; N(mu, sigma), initial value for mu = 0,
                        initial value for sigma = 5; N(mu, sigma), initial
                        value for mu = 0, initial value for sigma = 0.2; N(mu,
                        sigma), initial value for mu = 50, initial value for
                        sigma = 12.5; N(mu, sigma), initial value for mu =
                        1.25, initial value for sigma = 5 Linearized version:
                        Given means=[0, 0, 0, 0, 0, 50, 1.25], variances=[0.2,
                        5, 5, 5, 0.2, 12.5, 5] -> 0 0 0 0 0 50 1.25 0.2 5 5 5
                        0.2 12.5 5 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters initial values, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), initial value for mu_1 = 0, initial
                        value for mu_2 = 0, initial value for mu_3 = 0,
                        initial value for mu_4 = 0, initial value for mu_5 =
                        0, initial value for mu_6 = 1.25, initial value for
                        mu_7 = 50, initial value for sigma_ij = 0.2 if i = j =
                        {1,5}, initial value for sigma_ij = 5 if i = j =
                        {2,3,4,7}, initial value for sigma_ij = 12.5 if i = j
                        = 6, initial value for sigma_ij = 0 if i != j
                        Linearized version: Given mean_vector=[0, 0, 0, 0, 0,
                        1.25, 50], variance_covariance_matrix=[[0.2], [0, 5],
                        [0, 0, 5], [0, 0, 0, 5], [0, 0, 0, 0, 0.2], [0, 0, 0,
                        0, 0, 12.5], [0, 0, 0, 0, 0, 0, 5]] -> 0 0 0 0 0 1.25
                        500.2 0 5 0 0 5 0 0 0 5 0 0 0 0 0.2 0 0 0 0 0 12.5 0 0
                        0 0 0 0 5 -- triangular matrix required -- (default:
                        None)
  --adaptive_dom_rand__dist_initials___multivariatenormal ADAPTIVE_DOM_RAND__DIST_INITIALS___MULTIVARIATENORMAL [ADAPTIVE_DOM_RAND__DIST_INITIALS___MULTIVARIATENORMAL ...]
                        Adaptive domain randomization multivatiate normal
                        distribution parameters initial values. Use the
                        linearized version (i.e. Parameters list, whose
                        cardinality depends on the number of parameters:
                        single parameters separated by 1 single space. A
                        single parameter can be a scalar, a vector or a a
                        matrix. Inside a vector, elements are separated by 1
                        single space. Inside a matrix, rows are representd as
                        a (transpose) vector and inserted separated by 1
                        single space). If None, standard distribution
                        parameters ranges are used. Adaptive domain
                        randomization, standard distribution parameters
                        initial values. The meaning of the parameter depends
                        on the distribution(s) considered. If univariate
                        distributions are used, the order reflects the
                        corresponding image parameter to be randomized: -
                        Univariate uniform distribution, U(lower_brightness,
                        upper_brightness); U(lower_contrast, upper_contrast);
                        U(lower_horizontal_flip, upper_horizontal_flip);
                        U(lower_vertical_flip, upper_vertical_flip);
                        U(lower_hue, upper_hue); U(lower_jpeg_quality,
                        upper_jpeg_quality); U(lower_saturation,
                        upper_saturation) standard distribution parameters
                        initial values, U(a, b), initial value for a = 0,
                        initial value for b = 0; U(a, b), initial value for a
                        = 0, initial value for b = 0; U(a, b), initial value
                        for a = 0, initial value for b = 0; U(a, b), initial
                        value for a = 0, initial value for b = 0; U(a, b),
                        initial value for a = 0, initial value for b = 0; U(a,
                        b), initial value for a = 50, initial value for b =
                        50; U(a, b), initial value for a = 1.25, initial value
                        for b = 1.25 Linearized version: Given lowers=[0, 0,
                        0, 0, 0, 50, 1.25], uppers= [0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 - Univariate
                        triangular distribution, Tr(lower_brightness,
                        mode_brightness, upper_brightness); Tr(lower_contrast,
                        mode_contrast, upper_contrast);
                        Tr(lower_horizontal_flip, mode_horizontal_flip,
                        upper_horizontal_flip); Tr(lower_vertical_flip,
                        mode_vertical_flip, upper_vertical_flip);
                        Tr(lower_hue, mode_hue, upper_hue);
                        Tr(lower_jpeg_quality, mode_jpeg_quality,
                        upper_jpeg_quality); Tr(lower_saturation,
                        mode_saturation, upper_saturation) standard
                        distribution parameters initial values, Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 0, initial
                        value for m = 0, initial value for b = 0; Tr(a, m, b),
                        initial value for a = 0, initial value for m = 0,
                        initial value for b = 0; Tr(a, m, b), initial value
                        for a = 0, initial value for m = 0, initial value for
                        b = 0; Tr(a, m, b), initial value for a = 50, initial
                        value for m = 50, initial value for b = 50; Tr(a, m,
                        b), initial value for a = 1.25, initial value for m =
                        1.25, initial value for b = 1.25 Linearized version:
                        Given lowers=[0, 0, 0, 0, 0, 50, 1.25], modes= [0, 0,
                        0, 0, 0, 50, 1.25], uppers=[0, 0, 0, 0, 0, 50, 1.25]
                        -> 0 0 0 0 0 50 1.25 0 0 0 0 0 50 1.25 0 0 0 0 0 50
                        1.25 - Univariate normal distribution,
                        N(mean_brightness, variance_brightness);
                        N(mean_contrast, variance_contrast);
                        N(mean_horizontal_flip, variance_horizontal_flip);
                        N(mean_vertical_flip, variance_vertical_flip);
                        N(mean_hue, variance_hue); N(mean_jpeg_quality,
                        variance_jpeg_quality); N(mean_saturation,
                        variance_saturation) standard distribution parameters
                        initial values, N(mu, sigma), initial value for mu =
                        0, initial value for sigma = 0.2; N(mu, sigma),
                        initial value for mu = 0, initial value for sigma = 5;
                        N(mu, sigma), initial value for mu = 0, initial value
                        for sigma = 5; N(mu, sigma), initial value for mu = 0,
                        initial value for sigma = 5; N(mu, sigma), initial
                        value for mu = 0, initial value for sigma = 0.2; N(mu,
                        sigma), initial value for mu = 50, initial value for
                        sigma = 12.5; N(mu, sigma), initial value for mu =
                        1.25, initial value for sigma = 5 Linearized version:
                        Given means=[0, 0, 0, 0, 0, 50, 1.25], variances=[0.2,
                        5, 5, 5, 0.2, 12.5, 5] -> 0 0 0 0 0 50 1.25 0.2 5 5 5
                        0.2 12.5 5 - Multivariate normal distribution,
                        N(mean_vector, variance_covariance_matrix), with
                        mean_vector = [mean_i] and variance_covariance_matrix
                        = [var_ii or covar_ij], with i, j = {brightness,
                        contrast, horizontally flip, vertically flip, hue,
                        jpeg quality, saturation} standard distribution
                        parameters initial values, N( [mu_1, mu_2, mu_3, mu_4,
                        mu_5, mu_6, mu_7], [ [sigma_11, sigma_12, sigma_13,
                        sigma_14, sigma_15, sigma_16, sigma_17], [sigma_21,
                        sigma_22, sigma_23, sigma_24, sigma_25, sigma_26,
                        sigma_27], [sigma_31, sigma_32, sigma_33, sigma_34,
                        sigma_35, sigma_36, sigma_37], [sigma_41, sigma_42,
                        sigma_43, sigma_44, sigma_45, sigma_46, sigma_47],
                        [sigma_51, sigma_52, sigma_53, sigma_54, sigma_55,
                        sigma_56, sigma_57], [sigma_61, sigma_62, sigma_63,
                        sigma_64, sigma_65, sigma_66, sigma_67], [sigma_71,
                        sigma_72, sigma_73, sigma_74, sigma_75, sigma_76,
                        sigma_77] ] ), initial value for mu_1 = 0, initial
                        value for mu_2 = 0, initial value for mu_3 = 0,
                        initial value for mu_4 = 0, initial value for mu_5 =
                        0, initial value for mu_6 = 1.25, initial value for
                        mu_7 = 50, initial value for sigma_ij = 0.2 if i = j =
                        {1,5}, initial value for sigma_ij = 5 if i = j =
                        {2,3,4,7}, initial value for sigma_ij = 12.5 if i = j
                        = 6, initial value for sigma_ij = 0 if i != j
                        Linearized version: Given mean_vector=[0, 0, 0, 0, 0,
                        1.25, 50], variance_covariance_matrix=[[0.2], [0, 5],
                        [0, 0, 5], [0, 0, 0, 5], [0, 0, 0, 0, 0.2], [0, 0, 0,
                        0, 0, 12.5], [0, 0, 0, 0, 0, 0, 5]] -> 0 0 0 0 0 1.25
                        500.2 0 5 0 0 5 0 0 0 5 0 0 0 0 0.2 0 0 0 0 0 12.5 0 0
                        0 0 0 0 5 -- triangular matrix required -- (default:
                        None)
  --neural_network NEURAL_NETWORK
                        Neural Network to be used. Possible choices: [ResNet
                        1, ResNet 2.0, ResNet 2.1, ResNet 2.0.1, ResNet
                        2.1.1]. Neural Networks: - ResNet 1: very fast
                        training, potential low performances. - ResNet 2.0:
                        slowest training, able to reach the best performances.
                        - ResNet 2.1: fast training, potential low
                        performances, generally better than ResNet 1. - ResNet
                        2.0.1: deeper version of ResNet 2.0, useful for large
                        datasets with many classes. - ResNet 2.1.1: deeper
                        version of ResNet 2.1, useful for large datasets with
                        many classes. (default: ResNet 2.0)
  --gradient_based__optimizer GRADIENT_BASED__OPTIMIZER
                        Optimizer to be used for gradient based optimization.
                        Possible choices: SGD, RMSprop, Adagrad, Adadelta,
                        Adafactor, Adam, Adamax, AdamW, Lion, LossScale,
                        Nadam, FTRL, ProximalGradientDescent, ProximalAdagrad,
                        Schedules. (default: Adam)
  --gradient_based__optimizer___learning_rate GRADIENT_BASED__OPTIMIZER___LEARNING_RATE
                        Learning rate of the optimizer to be used for gradient
                        based optimization. It has to be a float. (default:
                        1e-03)
  --gradient_based__loss GRADIENT_BASED__LOSS
                        Loss to be used for gradient based optimization.
                        Possible choices: MeanSquaredError, MeanAbsoluteError,
                        MeanAbsolutePercentageError,
                        MeanSquaredLogarithmicError, SquaredHinge, Hinge,
                        CategoricalHinge, LogCosh, Huber,
                        CategoricalCrossentropy,
                        SparseCategoricalCrossentropy, BinaryCrossentropy,
                        KLDivergence, Poisson, CosineSimilarity, serialize,
                        deserialize. (default: CategoricalCrossentropy)
  --epochs EPOCHS       Training epochs. It has to be an int. (default: 1000)
  --gradient_free__optimizer GRADIENT_FREE__OPTIMIZER
                        Optimizer to be used for gradient free optimization.
                        Possible choices: SqrtMultiBFGS,
                        PortfolioNoisyDiscreteOnePlusOne, LogMultiBFGSPlus,
                        ASCMADEthird, SparseDoubleFastGADiscreteOnePlusOne,
                        TripleCMA, DiscreteDoerrOnePlusOne, QOPSO, EDA,
                        DiagonalCMA, SqrtBFGSCMAPlus, NgIoh3,
                        AlmostRotationInvariantDEAndBigPop, PymooBIPOP,
                        NoisyDiscreteOnePlusOne, RFMetaModelPSO,
                        HammersleySearchPlusMiddlePoint,
                        FastGADiscreteOnePlusOne, QOScrHammersleySearch,
                        DoubleFastGAOptimisticNoisyDiscreteOnePlusOne, LSCMA,
                        TwoPointsDE, Carola5, CSEC, NgDS11, Carola2, F3SQPCMA,
                        SuperSmoothRecombiningDiscreteLognormalOnePlusOne,
                        TEAvgScrHammersleySearch, FCMAs03, DS2, LhsDE,
                        DiscreteOnePlusOneT, Carola9, SDiagonalCMA,
                        ChainCMAwithLHS30, Carola10, ParametrizationDE, CSEC6,
                        MetaModelTwoPointsDE, SADiscreteOnePlusOneExp099,
                        MultiDiscrete, DiscreteNoisy13Splits, CMandAS2,
                        NgIoh2, Carola1, ChoiceBase, SqrtMultiBFGSPlus,
                        RealSpacePSO, ChainMetaModelSQP, MetaModelQODE,
                        HSMetaModel, FCMA, TinySPSA, OldCMA, TinyLhsDE,
                        Carola6, CMAtuning, AlmostRotationInvariantDE,
                        AvgMetaRecenteringNoHull,
                        TEAvgCauchyScrHammersleySearch, CSEC7,
                        FastGANoisyDiscreteOnePlusOne, NgIoh14, NGOptSingle25,
                        FCMAp13, DSproba8, PymooNSGA2, ChainPSOwithRsqrt,
                        CSEC8, BayesOptimBO, ChainPSOwithLHS, LogSQPCMAPlus,
                        BAR2, ChainBOwithMetaTuneRecenteringdim,
                        ChainPSOwithR, ChainPSOwithMetaRecentering30,
                        DoubleFastGADiscreteOnePlusOne, Carola14,
                        SADiscreteLenglerOnePlusOneLinAuto, LBO, LhsHSDE,
                        MultiScaleCMA, SmoothElitistRandRecombiningDiscreteLog
                        normalOnePlusOne,
                        HullCenterHullAvgScrHaltonSearchPlusMiddlePoint,
                        HullCenterHullAvgCauchyScrHammersleySearch,
                        ChainCMAwithRdim, BFGS, NGOptF2, Cobyla, UltraSmoothEl
                        itistRecombiningDiscreteLognormalOnePlusOne,
                        HullCenterHullAvgCauchyLHSSearch, TBPSA, DE, OpoDE,
                        LogBFGSCMA, NLOPT_GN_ISRES,
                        ChainBOwithMetaTuneRecentering30, IsoEMNA,
                        MaxRecombiningDiscreteLenglerOnePlusOne,
                        ChainBOwithMetaTuneRecenteringsqrt, SMAC3,
                        RandomSearchPlusMiddlePoint, NLOPT_LN_NELDERMEAD,
                        RandomScaleRandomSearchPlusMiddlePoint,
                        NLOPT_LN_SBPLX, NLOPT_GN_DIRECT_L, CauchyOnePlusOne,
                        MicroCMA, NgIohRW2, RFMetaModelTwoPointsDE, LargeCMA,
                        NgIoh12, DiscreteLenglerOnePlusOneT,
                        AdaptiveDiscreteOnePlusOne,
                        DiscreteLenglerHalfOnePlusOne,
                        SADiscreteLenglerOnePlusOneExp099, MicroSPSA,
                        MinRecombiningDiscreteLenglerOnePlusOne, MultiBFGS,
                        ChainCMAwithLHS, DiscreteBSOOnePlusOne, MemeticDE,
                        DS8, DiscreteOnePlusOne, ChainBOwithLHS, NgIoh12b,
                        PCABO, NgIoh11, ChainDEwithLHSdim,
                        HullAvgMetaRecentering, BOBYQA, AnisoEMNA,
                        SADiscreteLenglerOnePlusOneLin100,
                        ChainBOwithMetaRecenteringdim, DiscreteDE, DSproba7,
                        ChainDiagonalCMAPowell, NGDSRW,
                        SmoothLognormalDiscreteOnePlusOne, DSproba4,
                        ChainDEwithMetaTuneRecentering30, discretememetic,
                        UltraSmoothDiscreteLenglerOnePlusOne, NLOPT_LN_PRAXIS,
                        NGOptF, QORealSpacePSO, DS6, Carola4, NeuralMetaModel,
                        SADiscreteOnePlusOneLin100, Carola7, NaiveIsoEMNA,
                        SmoothDiscreteOnePlusOne, CSEC9,
                        ChainPSOwithMetaRecentering,
                        RescaleScrHammersleySearch, MetaModelDSproba,
                        MetaNGOpt10, NGOpt, CMAstd, PCarola6, ChainDEwithR,
                        SADiscreteLenglerOnePlusOneExp09Auto,
                        RecombiningPortfolioDiscreteOnePlusOne,
                        RandRecombiningDiscreteLenglerOnePlusOne, HSCMA, ES,
                        NLOPT_GN_ESCH, MilliCMA, ChainBOwithMetaRecentering,
                        ChainDEwithMetaRecentering30, TEAvgLHSSearch,
                        MiniLhsDE, MutDE, RCobyla,
                        ChainCMAwithMetaRecenteringdim,
                        DiscreteLengler3OnePlusOne, NgIoh7,
                        SmoothDiscreteLenglerOnePlusOne,
                        ChainCMAwithMetaRecenteringsqrt, NonNSGAIIES, MixES,
                        pysot, ChainBOwithMetaTuneRecentering, CSEC10,
                        ChainCMAPowell, TEAvgCauchyLHSSearch, NaiveAnisoEMNA,
                        BFGSCMAPlus, LargeDiagCMA, MetaTuneRecentering,
                        SqrtSQPCMA, NgIoh13b, DS5,
                        PortfolioDiscreteOnePlusOne,
                        DiscreteLenglerOnePlusOne, SVMMetaModelTwoPointsDE,
                        NGOptF3, Zero, NoisyRL1,
                        ChainDEwithMetaTuneRecentering, ChainDEwithRsqrt,
                        HullAvgMetaTuneRecentering, AX,
                        SuperSmoothRecombiningDiscreteLanglerOnePlusOne,
                        SpecialRL, RecMixES, SVMMetaModel, Shiwa, NgIoh13,
                        ChainCMAwithR30, HullCenterHullAvgScrHaltonSearch,
                        BFGSCMA, ChainCMAwithR, ChainCMAwithLHSdim, CMandAS3,
                        ChainPSOwithMetaRecenteringsqrt,
                        SmoothRecombiningPortfolioDiscreteOnePlusOne,
                        AnisoEMNATBPSA, VoronoiDE, CMApara,
                        SmoothRecombiningDiscreteLanglerOnePlusOne, NgIoh17,
                        VLPCMA, ChainCMAwithLHSsqrt,
                        HullCenterHullAvgLHSSearch,
                        SmoothElitistRecombiningDiscreteLanglerOnePlusOne,
                        TinyQODE, MetaModelDE, Carola3, Carola15,
                        ChainDEwithLHS30, SQPCMAPlus, FSQPCMA, MiniDE,
                        ChainDEwithRdim, MetaCMA, NoisyBandit, cGA, NGOptBase,
                        MidQRBO, RecombiningOptimisticNoisyDiscreteOnePlusOne,
                        ChainNaiveTBPSAPowell, MetaCauchyRecentering,
                        NGOptSingle9, LPCMA, Wiz,
                        PortfolioDiscreteOnePlusOneT, NLOPT_LN_COBYLA,
                        NgIoh18, RecES, Carola11, DSproba, MultiCMA,
                        AvgHammersleySearch, TinyCMA, NgIoh15, NgIoh8,
                        ChainDEwithMetaTuneRecenteringdim,
                        RecombiningDiscreteLanglerOnePlusOne, PSO,
                        ChainBOwithMetaRecentering30,
                        ChainCMAwithMetaRecentering30, ChainMetaModelDSSQP,
                        QRBO, DS14, UltraSmoothElitistRecombiningDiscreteLangl
                        erOnePlusOne, BAR4, SmoothPortfolioDiscreteOnePlusOne,
                        NGOptDSBase, RecombiningDiscreteLognormalOnePlusOne,
                        ChainDEwithMetaRecenteringsqrt, MetaModelOnePlusOne,
                        ChainMetaModelPowell, Powell, ChainBOwithLHSsqrt,
                        OptimisticNoisyOnePlusOne,
                        SmoothAdaptiveDiscreteOnePlusOne,
                        OnePtRecombiningDiscreteLenglerOnePlusOne,
                        HullCenterHullAvgLargeHammersleySearch, CMAbounded,
                        SOPSO, ChainBOwithRdim,
                        ScrHammersleySearchPlusMiddlePoint, ScrHaltonSearch,
                        MultiSQPPlus, ChainBOwithLHS30, AvgRandomSearch,
                        MultiCobyla, DSproba3, pCarola6, ChainDSPowell,
                        NLOPT_GN_DIRECT, OScrHammersleySearch, ChainDEwithR30,
                        NoisyOnePlusOne, FCarola6, HSSVMCMA,
                        ChainDEwithMetaRecentering, RFMetaModelDE,
                        ChainBOwithMetaRecenteringsqrt, MultiCobylaPlus,
                        CauchyScrHammersleySearch, RPowell,
                        SADiscreteLenglerOnePlusOneExp09, MetaModelFmin2,
                        NgIoh14b, QrDE, ECMA, QODE, ChainPSOwithLHSsqrt,
                        ChainNaiveTBPSACMAPowell, VastLengler, IsoEMNATBPSA,
                        HaltonSearchPlusMiddlePoint, NgIoh4, MultiDS, CM,
                        NoisyInfSplits, NgDS, ORandomSearch, NgIoh10,
                        ChainBOwithR, ChainBOwithLHSdim, OnePlusLambda,
                        OptimisticDiscreteOnePlusOne,
                        SparseDiscreteOnePlusOne, ChainDEwithLHSsqrt,
                        SmoothElitistRandRecombiningDiscreteLanglerOnePlusOne,
                        TinySQP, F2SQPCMA, SVMMetaModelDE, RBFGS,
                        ChainCMAwithRsqrt,
                        TEAvgScrHammersleySearchPlusMiddlePoint, CSEC4,
                        MultiBFGSPlus, SmoothDiscreteLognormalOnePlusOne,
                        LBFGSB, RotatedTwoPointsDE, LogMultiBFGS,
                        AnisotropicAdaptiveDiscreteOnePlusOne, NGOpt10,
                        NLOPT_LN_NEWUOA_BOUND, PolyCMA, HammersleySearch,
                        NeuralMetaModelTwoPointsDE, NGOptRW,
                        SuperSmoothDiscreteLenglerOnePlusOne, NoisyDE,
                        ChainDEwithLHS, BOSplit, ChainBOwithR30,
                        ChainPSOwithR30, NLOPT_GN_CRS2_LM, SQP, NGO, LQODE,
                        CMAsmall, QNDE, NgIoh6, CMA, QOTPDE, ChainDE, NGOpt39,
                        DSproba6, ChainPSOwithLHSdim, SPSA, NgDS3, DS4,
                        NelderMead, RFMetaModelOnePlusOne, DSproba5,
                        ChainPSOwithMetaRecenteringdim, DSsubspace, DS9,
                        HSRFCMA, Carola8, CSEC5, LogSQPCMA,
                        HullCenterHullAvgScrHammersleySearch, NGOpt36,
                        SQORealSpacePSO,
                        HullCenterHullAvgScrHammersleySearchPlusMiddlePoint,
                        Portfolio, NGOptSingle16, LargeHaltonSearch,
                        ChainCMASQP, NgIoh5,
                        TwoPtRecombiningDiscreteLenglerOnePlusOne,
                        HaltonSearch, ScrHaltonSearchPlusMiddlePoint,
                        OnePointDE, DSbase, ForceMultiCobyla, NgIoh20,
                        Noisy13Splits, PymooCMAES, NGOpt8,
                        PortfolioOptimisticNoisyDiscreteOnePlusOne, NGOptF5,
                        MultiSQP, PCABO95DoE20, DS3p, SVMMetaModelPSO, NgIoh9,
                        LogBFGSCMAPlus, StupidRandom, MixDeterministicRL,
                        ScrHammersleySearch, OpoTinyDE,
                        FastGAOptimisticNoisyDiscreteOnePlusOne,
                        DiscreteLengler2OnePlusOne, NLOPT_GN_AGS, NgDS2,
                        RotatedRecombiningGA, PCABO80, NaiveIsoEMNATBPSA, RBO,
                        RLSOnePlusOne, SODE, MetaModelDiagonalCMA,
                        QORandomSearch, SQOPSO, MetaRecentering, BO, VastDE,
                        ParaPortfolio, BAR, ChainCMAwithMetaRecentering,
                        DiscreteLenglerFourthOnePlusOne, MicroSQP,
                        BPRotationInvariantDE, HullCenterHullAvgRandomSearch,
                        NeuralMetaModelDE,
                        RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne,
                        NgIoh15b, CmaFmin2, TEAvgRandomSearch, NgIoh19,
                        RotationInvariantDE, NaiveTBPSA, RescaledCMA,
                        OnePlusOne, SQPCMA, RecMutDE,
                        AvgHammersleySearchPlusMiddlePoint, NGOpt4, NgIoh,
                        NaiveAnisoEMNATBPSA, HSDE, Carola13, NgIoh16,
                        ChainDEwithMetaRecenteringdim, RandomSearch,
                        RFMetaModel, NgLn, SADiscreteOnePlusOneExp09,
                        MetaModelPSO,
                        RandRecombiningDiscreteLognormalOnePlusOne,
                        CauchyLHSSearch, ChainDEwithMetaTuneRecenteringsqrt,
                        HSNeuralCMA, NLOPT_LN_BOBYQA, LQOTPDE,
                        SADiscreteLenglerOnePlusOneLin1, ChainPSOwithLHS30,
                        GeneticDE, NoisyRL3, SPQODE, CauchyRandomSearch,
                        LognormalDiscreteOnePlusOne, MetaModel,
                        RandomScaleRandomSearch, RSQP, DSproba9, NGOpt15,
                        MiniQrDE, BAR3, ChainBOwithRsqrt, NGOpt16, SuperSmooth
                        ElitistRecombiningDiscreteLanglerOnePlusOne, NgIoh21,
                        SqrtBFGSCMA, ChainPSOwithRdim, RecombiningGA,
                        DSproba2,
                        UltraSmoothRecombiningDiscreteLanglerOnePlusOne,
                        LHSSearch, NoisyRL2, SqrtSQPCMAPlus,
                        DiscreteNoisyInfSplits. (default: CMA)
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
   		(Note that the variance covariance matrix is a _positive definite matrix_: only elements of diagonal and elements below diagonal has to be specified.)
	<pre>
	Given mean_vector=[-1, float(-inf), float(-inf), float(-inf), -1, 0, 0], variance_covariance_matrix=[[0], [-100, 0], [-100, -100, 0], [-100, -100, -100, 0], [-100, -100, -100, -100, 0], [-100, -100, -100, -100, -100, 0], [-100, -100, -100, -100, -100, -100, 0]] -> -1 float(-inf) float(-inf) float(-inf) -1 0 0 0 -100 0 -100 -100 0 -100 -100 -100 0 -100 -100 -100 -100 0 -100 -100 -100 -100 -100 0 -100 -100 -100 -100 -100 -100 0 
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

  

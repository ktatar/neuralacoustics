"# neuralacoustics" 

**PREREQS**

_Donwload and install Anaconda

_Create new environment:

    conda create --name YOUR_ENV python=3.7 
  
_Install required libraries in environment:
  
    conda install pytorch torchvision torchaudio cpuonly -c pytorch 
  
    conda install scikit-learn 
   
    conda install scipy 
  
    conda install matplotlib 
  
    conda install h5py 
    
    onda install -c conda-forge tensorboard 
    
(NOTE: missing CUDA installation, this will run code on CPU only)

The argunment "-c" is the channel (i.e., repo). If any library is not available from Anaconda or from the specified channel, try with:

    pip install LIB_NAME


**USAGE**

  Dataset Creation

  _From root dir, activate proper Anaconda environment:

    conda activate YOUR_ENV

  _Run dataset_generation.py and pass config file (optional—if not specified reads default.ini):

    python dataset_generation.py --config /path/to/YOUR_CONFIG_FILE
    
  Only section "dataset_generation" of config file will be parsed. This includes the path to the numerical model that will be used to generate the dataset.

  Note that every numerical model is associated to a config file (MODEL_NAME.ini) too, found in the numerical model's folder. Make sure that the values under section "parameters" are set correctly, for they will determine the behavior of the model as well as the content of the resulting dataset.
    
  The script will generate in the path specified in config file a folder with the new dataset (typically split in chunks—see "chunks" entry in default.ini), along with a log file that contains only info relevant to the generation of the dataset (e.g., number of entries, batch size, details of the numerical model).

  The name of the dataset will be "dataset_N", where N is the number of dataset already present in the target dir.
   
  If in config file "dryrun" is set to 1, then the script will compute a single dataset entry (i.e, a single simulation, instead of N), that will be visualized while computed, but not saved.


"# neuralacoustics" 

**PREREQS**

_Donwload and install Anaconda

_Create new environment:

    conda create --name YOUR_ENV python=3.7 
  
_Install required libraries in environment:
  
    conda install pytorch torchvision torchaudio cpuonly -c pytorch 
  
    conda install jupyter 
  
    conda install scikit-learn 
   
    conda install scipy 
  
    conda install matplotlib 
  
    conda install h5py 
    
(NOTE: missing CUDA installation, this will run code on CPU only)

The argunment "-c" is the channel (i.e., repo). If any library is not available from Anaconda or from the specified channel, try with:

    pip install NAME_OF_LIB


**USAGE**

  Dataset Creation

  _From root dir, activate proper Anaconda environment:

    conda activate YOUR_ENV

  _Run dataset_generation.py and pass config file (optional—if not specified reads default.ini):

    python dataset_generation.py --config /path/to/YOUR_CONFIG_FILE
    
  (Only section "dataset_generation" of config file will be parsed.)
    
  It will generate in the path specified in config file a folder with the new dataset (typically split in checkpoint files—see "checkpoints" entry in default.ini), along with a copy of the config file that contains onlty info relevant to the generation of the dataset (e.g., number of entries, batch size; while model training settings are ignored).
   
  If in config file "dryrun" is set to 1, then the script will compute a single dataset entry (i.e, a single simulation, instead of N), that will be visualized while computed but not saved.


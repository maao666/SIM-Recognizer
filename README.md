# SIM Card ICCID Recognizer
Detect the text on SIM cards with Convolutional Neural Network


## Basic Usage:

    >>> from SIM_OCR import SIM_OCR
    
    >>> sim = SIM_OCR('file_path.jpg')
    
    >>> print(sim.get_serial())

## Required:

    $ pip3 install opencv-contrib-python keras

## Getting Dataset:

    >>> from SIM_OCR import SIM_OCR
    
    >>> sim = SIM_OCR('file_path.jpg')
    
    >>> sim.save_dataset(path='dataset_folder/')
    
    You might want to set file_name_correspondence to
        true so that the file would be automatically
        tagged.

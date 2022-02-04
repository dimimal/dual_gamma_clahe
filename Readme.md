## Automatic Contrast-Limited Adaptive Histogram Equalization With Dual Gamma Correction

This is an attempt to implement in python the algorithm of the paper `Automatic Contrast-Limited Adaptive Histogram
Equalization With Dual Gamma Correction`. This is not an official repository.

## Prerequisites
Create your environment and install the dependencies required for this project.

```commandline
python3 -m venv env
source env/bin/activate
pip install pip --upgrade
pip install -r requirements.txt
```

To execute the algorithm run the `main.py` script. Running the main function with the -h flag 
you can get the list of arguments that you can pass to the main. You pass all the necessary arguments 
noted here in order to run the algorithm. 

``` 
usage: main.py [-h] [--kernel KERNEL [KERNEL ...]] [--alpha ALPHA]
               [--delta DELTA] [--p P] [--show] [--out OUT]
               image

positional arguments:
  image                 Size of the kernel, if int then its (int, int)

optional arguments:
  -h, --help            show this help message and exit
  --kernel KERNEL [KERNEL ...]
                        Size of the kernel, if int then its (height, width)
  --alpha ALPHA         Alpha parameter of the algorithm
  --delta DELTA         The Delta threshold of the algorithm
  --p P                 The factor for the computation of clip limits
  --show                Display the 2 figures with matplotlib, before and
                        after equalization
  --out OUT             Output directory of the equalized image. Default
                        folder is the ./images folder
```


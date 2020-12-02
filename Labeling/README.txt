To Run the Labeleling Utility:

create a anaconda environment using label.yml

activate the anaconda environment then navigate to the Labeling directory
..\WRR_Segmentation_Data\Labeling\

create a params_XXXX.py script within the \WRR_Segmentation_Data\Labeling\modules directory.

Change all of the paths to your specific needs (to be labeled) and create
directories for output. Follow the example of other scripts in the modules directory

Change the lines 

"""from modules.params_RCSandbar import params""
to 
"""from modules.params_XXXX import params""

in the following scripts:
Labeler.py
Crf.py 
CRFLater.py 

Then with the conda environment activate, once you are in the '..\WRR_Segmentation_Data\Labeling\'
directory...
run
>> python Labeler.py 

Do this for as many images as you want...then 
run
>> python CRFLater.py 

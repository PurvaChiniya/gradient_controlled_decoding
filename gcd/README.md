To reproduce the results, setup the environment using 
pip install -r requirements.txt
##Detection Model
 Run the detection model with detection.py 

##Threshold 
To run the threshold on a specific dataset use threshold.py to set t_sure and t_sorry. 

##Controlled Decoding 
For controlled decoding run the model with the generated labels from detection model , using python controlled_decode.py

##Testing 
For end-to-end testing run eval_one.py

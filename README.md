# AI_IIT110_Task_Image_Classification_And_Transfer_Learning
## Basic Code Information
The code has been developped in **python 3.8.2**. Activate the virtual environment following the instructions below. Then, using an IDE (i.e spyder), we can run the code by executing the main.py file.

* Activate the virtual environment :
	- On Windows :: myenv\Scripts\activate
	- On macOS and Linux :: source myenv/bin/activate

## Repository Structure
**dummy_data** - 

**logs** - Folder that is created during the first run of the code and contains log files with useful information for the program execution.

**myenv** - Folder that contains the virtual environment which has been used for the code development. 

**py_imports** - Folder that contains the python scripts (classes, functions) which are used inside the program.

**py_plots** - Folder that contains the python functions that are used in order to create the necessary diagrams.

**results** - Folder that contains the results and is created during the first run of the program.

**main.py** - The main python file.

**.gitignore, README, environment** - Git and virtual environment files.

## Basic Results
Below we can see the results of our final trained model. More details can be found in the report.
| Metric            | UECFOOD256 | FOOD101 (Transfer)     |     
|-------------------|---------|---------|
| Baseline Accuracy | 0.02186 | 0.01386 |
| Accuracy          | 0.76751 | 0.63415 |
| Top‑5 Accuracy    | 0.90764 | 0.82607 | 
| Precision         | 0.78941 | 0.64691 |
| Recall            | 0.76751 | 0.63534 |

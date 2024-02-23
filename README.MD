1. the libraries required to run the project including the full version of each library


Use:
pip install opencv-python
pip install scikit-image

To get:
opencv-python 4.8.1.78
scikit-image 0.22.0
scipy 1.11.4
numpy 1.26.2
and other dependencies


2. how to run each task and where to look for the output file.

Everything is setup in the archive already, install the libraries, then run "ddd.py" to get the output of all tasks, then "evalueaza_solutie.py"

Manual steps:
In the project root, at the same level as the py scripts:
	Create an empty folder named "output"
	Add the "templates" folder from the archive
	Add the "auxiliary" folder from the archive
	Add the input files in a folder named "input" (MUST be the exact same naming scheme as the train data)
	In the "ddd.py" (main) script, change the "game_count" (default: 5) variable to the number of games that will be analysed
	Run "ddd.py", the output txt files will be found in the "output" folder. They contain data for all 3 tasks.
	
	In "evalueaza_solutie.py", change predictions_path_root to "output/" and gt_path_root to the appropiate path.
	Run "evalueaza_solutie.py" for results
	

# Assignment 3

### First, be sure to read the main README.md for general information!
### Due date: 11:55PM of 06 of November. LATE SUBMISSIONS WILL NOT BE ACCEPTED.
#### Notice that there are new libraries in the requirement.txt file. Remember to pip install (or conda install) them!

## How to run the Python Code
In case you are using pycharm, you just need to right click any python file and select `Run 'ScriptName.py'` to run it, however **if you are running the python file from a terminal**, run the following:
* For windows, `cd` into the *assignments1* folder and run `set PYTHONPATH=..\..\ && python ScriptName.py`
* For mac/linux, `cd` into the *assignments1* folder and run `export PYTHONPATH=../../ && python ScriptName.py`

This is due to the normal import pathing python uses. This will ensure that you are able to use code from other folders in your project (e.g. use assignment1 scripts in the assignment2 folder, when time comes).

## General Instructions
* Read the Dalhousie Policy on Plagiarism.
* The assignment must be done individually.
* Properly cite any external sources that you used, for example websites, articles and papers, as comments in your code.
* You are NOT allowed to use or reuse any piece of code from other students.
* For this assignment, all implementation has to be in Python3.8, and using ONLY the libraries listed in the requirements.txt file.
* Use the datasets provided: iris.csv, ratings_Video_Games.csv, life_expectancy_years.csv and geography.csv
* We will run and validate your code, so make sure it runs with no errors or issues! Errors will be considered as a task not being completed.
* Submit your assignment to gitlab as described in the main README.md.
* After submitting your code, double check if it works on your machine. We won't accept corrupted submissions!

Our third assignment will tackle data visualization. We will investigate how to visualize data with two libraries (matplotlib and plotly), but do know there are many others. Your task will be to go through all files of the assignment directory and write the requested code.

At the end, we will also exemplify and ask you to develop a full blown full-stack webapp, which will be very easy by using plotly and dash combined, so only python will be needed. If you know javascript, then many more possibilities are available to you for your project, but for the assignment we will only use python-based visualization libraries.

Each file has directions and at least one example of what is expected of you to complete the tasks of that file. Then, a set number of methods (tasks) will be awaiting your implementation. Some will be simple, some complex. Some have direct solutions, while others will depend on your preferences/opinions on how to tackle the problem.

## Rubric
The outline of the division of points are as follows (totalling 100):
* 20 to a_libraries.py
* 20 to b_simple_usages.py
* 20 to c_interactivity.py
* 40 to d_dash_backend.py

The outline of the division of points for each task are as follows (totalling 100%):
* 15% Well/efficiently implemented code
* 15% Well used methods from the previous tasks and (optionally) the provided libraries (e.g. not reinventing the wheel)
* 15% Well documented/explained code
* 15% Good thought process with correct decisions (e.g. explain correctly why removing nans by removing the entire row in the dataset is a bad idea)
* 40% Right answer (e.g. acceptable answer, including comments of your decisions and/or rationale)

## Asking for assignment review
After receiving your grade, you will have up to 1 week to ask for a review of your assignment. The review will be made by having a different TA mark your assignment. The grade may change to a higher or a lower value, and this new value will be your new official grade. No single-task review will be done, only full assignment review. Keep this in mind when asking for a review of your assignment.

Questions and reasoning for grades (not reviews) may be asked at any time during office hours.
  

## More information about grading

The grading policy for the University is the responsibility of the University Senate and can by found on the University’s website at: https://www.dal.ca/dept/university_secretariat/policies/academic/grading-practices-policy.html. Please note the wording and terminology used to describe the grades. Satisfying assignment requirements without deficiencies may show “good” performance, but not necessarily an excellent performance. Your marker, after reviewing your submission that satisfied the submission requirements will need to determine if it is good or if it is excellent. If there were some deficiencies, although satisfying the requirements, the performance may be viewed as satisfactory, depending on the scale (quality and quantity) of deficiencies. If there are few, or only minor deficiencies, your submission may be viewed as good. If you do not have deficiencies, and your submission is, in your evaluator’s opinion excellent, then you are likely in the A range.

# Assignment 16 of October

### First, be sure to read the main README.md for general information!
### Due date: 11:55PM of 16 of October. LATE SUBMISSIONS WILL NOT BE ACCEPTED.

*Each of the tasks must be able to run in under 10 minutes in a Desktop machine with a Ryzen 5*.
In case this is a problem for you, since tasks in this assignment uses code from assignment 1, feel free to remove some of the logic of your assignment1 code (without any penalty), do more preprocessing (remember the techniques seen in class) and change the parameters of the model used by the task. We won't be grading your exact resulting prediction or model score, but your logic/mindset while performing the tasks required from each task. 

## How to run the Python Code
In case you are using pycharm, you just need to right click any python file and select `Run 'ScriptName.py'` to run it, however **if you are running the python file from a terminal**, run the following:
* For windows, `cd` into the *assignments2* folder and run `set PYTHONPATH=..\..\ && python ScriptName.py`
* For mac/linux, `cd` into the *assignments2* folder and run `export PYTHONPATH=../../ && python ScriptName.py`

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

Our second assignment will tackle machine learning.
Your task will be to go through all files of the assignment directory and write the requested code. The assignment is divided into 5 different steps, listed as 4 files (from a_<file> until d_<file>). Use to your advantage the methods you created during the previous assignment!

Each file has directions and at least one example of what is expected of you to complete the tasks of that file. Then, a set number of methods (tasks) will be awaiting your implementation. Some will be simple, some complex. Some have direct solutions, while others will depend on your preferences/opinions on how to tackle the problem.

## Rubric
The outline of the division of points are as follows (totalling 100):
* 30 to a_classification.py
* 30 to b_regression.py
* 40 to c_clustering.py
* 10 to d_extra.py

Among the files, each of the tasks will be graded linearly depending on how well you considered each step of the dataset transformation. Think well and describe your thoughts with commentaries in the code. A right answer may not be given 100% if wrong suppositions were made or if the code is malformed (badly implemented or not properly explained/commented). DO NOT SUBMIT WITH COMMENTED CODE WITHOUT AN EXPLICIT REASON TO DO SO. You are not required to comment every single line of code, but you are expected to have a code which we can understand why you decided to do what you did. As we usually say, this is not a programming course, therefore we are not grading how well you produce code, but we are grading your mindset and problem solving given the not-exact instructions of the tasks. If a task returns an error when executed instead of an answer, it will be considered wrong and given 0% of its mark.

The outline of the division of points for each task are as follows (totalling 100%):
* 15% Well/efficiently implemented code
* 15% Well used methods from the previous tasks (not reinventing the wheel)
* 15% Well documented/explained code
* 15% Good thought process with correct decisions (e.g. explain correctly why removing nans by removing the entire row in the dataset is a bad idea)
* 40% Right answer (e.g. acceptable answer) regarding how you tackled the problem, how you solved it, and the assumptions you made.

## Asking for assignment review
After receiving your grade, you will have up to 1 week to ask for a review of your assignment. The review will be made by having a different TA mark your assignment. The grade may change to a higher or a lower value, and this new value will be your new official grade. No single-task review will be done, only full assignment review. Keep this in mind when asking for a review of your assignment.

Questions and reasoning for grades (not reviews) may be asked at any time during office hours.
  

## More information about grading

The grading policy for the University is the responsibility of the University Senate and can by found on the University’s website at: https://www.dal.ca/dept/university_secretariat/policies/academic/grading-practices-policy.html. Please note the wording and terminology used to describe the grades. Satisfying assignment requirements without deficiencies may show “good” performance, but not necessarily an excellent performance. Your marker, after reviewing your submission that satisfied the submission requirements will need to determine if it is good or if it is excellent. If there were some deficiencies, although satisfying the requirements, the performance may be viewed as satisfactory, depending on the scale (quality and quantity) of deficiencies. If there are few, or only minor deficiencies, your submission may be viewed as good. If you do not have deficiencies, and your submission is, in your evaluator’s opinion excellent, then you are likely in the A range.

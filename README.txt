Structure of the project folder:

my_project/
├── env/                  ← Conda environment (created with `conda create -p ./env`)
├── data/                 ← (optional) datasets, large files
├── src/                  ← your Python code
├── notebooks/            ← Jupyter notebooks
├── environment.yml       ← environment specification file (for reproducibility)
├── .gitignore
└── README.md


To create environment.yml: 
	conda env export --from-history > environment.yml
	pip freeze --exclude-editable | sed -E 's/ @ file:.*$//' > requirements.txt
#	pip freeze > requirements.txt
#	conda list --export > requirements.txt
	
Then create a .gitingnore to avoid adding the enviornment to git:
	echo "env/" >> .gitignore
	echo "__pycache__/" >> .gitignore
	echo "*.pyc" >> .gitignore
Finally, initialize git:
	git init
	git add .
	git commit -m "Initial project setup"
Publish to git:
	gh repo create XBox_Processing --public --source=. --remote=origin
	Poi
	git push -u origin main


In truth:
git status                         # just to see where you are
git add .                          # stage something (e.g., README.md, .gitignore)
git commit -m "Initial commit"     # create the first commit
git branch -M main                 # rename current branch to 'main'
git remote add origin git@github.com:lsito/XBox_Processing.git  # or use HTTPS URL
git push -u origin main

# Advent of Code

Allows for semi-automated browser advent-of-code submittions. Verified with Brave.

## Running

1. Open Python project.
2. Copy previous day to the next
3. Open powershell terminal in `aoc/solvewatcher`.
4. Execute `./brave.ps1`.

Start with:
```sh
# For awaiting "todays" puzzle:
node .\main.js

# Or a specific other day with:
node .\main.js "2022-12-01"
```  

## Setup
Install with `pip install -r requirements` in root `aoc`-folder in an activated virtual environment. Then install the
browser automation with `npm -i` in `aoc/solutionwatcher`. It awaits the games starting, downloads puzzle input and
then evaluates candidate solutions if they pass a check against `_test == <solve_1>` and `__test == <solve_2>`
respectively.

> It's recommended to use Python 3.9 to 3.10 for now as 3.11 is having some issues with several modules in pip (as of 
2022-12-01).  

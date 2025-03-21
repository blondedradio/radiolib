## radiolib

An assortment of library scripts to complement any features for [RadioRacers](https://github.com/blondedradio/RadioRacers), a DRRR* fork.

<small>*An acryonym for the Kart Racer, [Dr. Robotnik's Ring Racers](https://www.kartkrew.org/).</small>

## How to get started

1. Ensure that [Python](https://www.python.org/) and [`pip`](https://pypi.org/project/pip/) are installed on your system.
2. Install the requirements for this repository, `pip install -r requirements.txt`
   
## Libaries
    
## `emotes.py`

> **TL;DR** Chews up a bunch of emotes, spits out a single PK3.

This script expects a folder of subfolders, each representing an emote in one of three formats:

1. **Static** emote 
   - Single image (preferably PNG). The _folder name_ becomes the emote name.
2. **Animated** emote 
   - Single GIF. The _folder name_ becomes the emote name.
3. **Atlas** emote 
   - Multiple images. Each image's filename becomes its own emote name.

To see a demonstration, execute `run_examples.bat` - this will process the subfolders in the `.\examples` directory and configure any emotes the script finds.

---
To get started, simply drag and drop a folder onto `run.bat`.

For more information, run `emotes.py --help`.
'''
    radiolib - extra tools and utilities to complement features in the "RadioRacers" custom build
    Copyright (C) 2025 $HOME

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, see
    <https://www.gnu.org/licenses/>.

    emotes.py: main functionality for spitting out emotes
'''

import zlib
import asyncio
import aiofiles

import math
import zipfile
import sys
import numpy as np
from scipy.spatial import KDTree 
from pathlib import Path
from PIL import Image, ImageSequence
from concurrent.futures import ThreadPoolExecutor

class RadioLibException(Exception):
    pass

'''
 === 
 CONSTANTS
 ====

'''
REMAP_IMAGE = False
FINAL_PK3_NAME = "emotes"

# Straight from the build src code
EMOTE_NAME_SIZE = 80
MAX_EMOTE_FRAMES = 300

tmp_folder = ".dump"
EMOTE_DEF_NAME = "EMOTEDEF"
EMOTE_DEF_NAME_KEY = "Name"
EMOTE_DEF_TICS_KEY = "Tics"

ATLAS_DEF_NAME = "ATLASDEF"
ATLAS_IMAGE_NAME = "EMOATLAS"

# SRB2/RR palettes should have 256 values
SRB2_PALETTE_SIZE = 256
SRB2_PALETTE_NAME = "PLAYPAL.png"
SRB2_PALETTE = ""
SRB2_RGB_PALETTE = ""
SRB2_FLAT_PALETTE = ""
SRB2_PALETTE_KD = ""

EX_OK = 0
EX_FAIL = 1

'''
 === 
 CONSTANTS
 ====
'''

'''
FILE HANDLING
'''
def create_atlas(images: list, atlas_size: tuple, rows_and_columns: tuple):
    atlas = Image.new("RGBA", atlas_size, (255, 255, 255, 0))

    img_width, img_height = images[0].size

    # Loop through the images, and "paste" them in the atlas image
    for i, img in enumerate(images):
        row = i // rows_and_columns[1]
        column = i % rows_and_columns[1]

        x_pos = column * img_width
        y_pos = row * img_height

        atlas.paste(img, (x_pos, y_pos))

    return atlas


def create_pk3():
    global FINAL_PK3_NAME
    destination_path = Path.cwd() / "build"
    tmp_dir = Path.cwd() / tmp_folder
    pk3_full_path = destination_path / f"{FINAL_PK3_NAME}.pk3"

    if(not destination_path.exists()):
        destination_path.mkdir(parents=True, exist_ok=True)
    
    # if the PK3 already exists, delete it and start from scratch again
    if (pk3_full_path.exists()):
        pk3_full_path.unlink()
        print("\nFound existing emotes.pk3, deleting and building from scratch")
        
    # Create that PK3
    total_zipped_files = 0
    total_emotes = len(list(tmp_dir.glob('*')))
    with zipfile.ZipFile(pk3_full_path, 'w', compression=zipfile.ZIP_DEFLATED) as pk3:
        for item in tmp_dir.iterdir():
            if (item.is_dir()):
                print(end="\033[2K")

                for file in item.iterdir():
                    pk3.write(file, arcname=f"{item.name}/{file.name}")
                total_zipped_files+= 1
                print(f"[{total_zipped_files}/{total_emotes}] Zipping {item.name}", end="\r")

def empty_temp(tmp_path: Path):
    import shutil
    if (tmp_path.is_dir()):
        is_empty = not any(tmp_path.iterdir())
        if (is_empty):
            return
        
        for item in tmp_path.iterdir():
            try:
                if(item.is_file()):
                    item.unlink()
                elif(item.is_dir()):
                    shutil.rmtree(item)
            except PermissionError:
                print(f"Couldn't delete {item} from temp")

def get_frame_delay_in_tics(frame_delay: int):
    '''
    A tic is 1/35th of a second = 1000/35 = 28.57ms = 28 (rounded down)
    So if an animated emote has a frame delay of 30 milliseconds:

    (frame_delay + 14) / 28
    (30 + 14) / 28
    (44) / 28
    1.57 tics, rounded down to 1 tic

    So each frame in the animated emote should last for 1 frame in-game

    '''
    new_delay = math.floor((frame_delay + 14)/ 28)
    if (new_delay <= 0):
        return 1
    
    return min(max(new_delay, 1), 35)

def create_path_for_emote(emote_name: str):
    new_path = Path.cwd() / tmp_folder / emote_name

    if(not new_path.exists()):
        new_path.mkdir()
    return new_path

def validaite_emote_name(emote_name: str):
    if (len(emote_name) > EMOTE_NAME_SIZE):
        print(f"'{emote_name}' is longer than {EMOTE_NAME_SIZE} characters, shorten it.")
        return False
    return True
    
'''
PALETTE HANDLING
'''
def load_palette(palette_file: Path):
    global SRB2_PALETTE
    global SRB2_RGB_PALETTE
    global SRB2_PALETTE_KD
    global SRB2_FLAT_PALETTE

    palette_image = Image.open(palette_file)
    raw_palette = Image.open(palette_file).palette

    if (len(raw_palette.colors) > SRB2_PALETTE_SIZE):
        raise RadioLibException(f'{palette_file} is not a standard SRB2 palette.')

    # Remove any duplicates
    seen = set()
    unique_tuples = []
    for colour in list(palette_image.convert("RGBA").getdata()):
        if (colour not in seen):
            seen.add(colour)
            unique_tuples.append(colour)
    
    SRB2_PALETTE = unique_tuples
    SRB2_PALETTE_KD = KDTree([c[:3] for c in SRB2_PALETTE])
    SRB2_RGB_PALETTE = [c[:3] for c in SRB2_PALETTE]
    SRB2_FLAT_PALETTE = [value for colour in SRB2_RGB_PALETTE for value in colour]

def closest_palette_colour(colour):
    global SRB2_PALETTE
    # Eucildean distance formula (https://www.cuemath.com/euclidean-distance-formula/)

    # colour would be a tuple (r, g, b, a), e.g. (200, 15, 25, 100)
    r, g, b, a = colour

    if a == 0:
        return (0, 0, 0, 0)

    return (
        min(
            SRB2_PALETTE, key=lambda c: (c[0] - r) ** 2 + (c[1] - g) ** 2 + (c[2] - b) ** 2
        )
    )

def closest_palette_colour_numpy(colour):
    global SRB2_PALETTE
    global SRB2_PALETTE_KD

    r, g, b, a = colour

    if a == 0:
        return (0, 0, 0, 0)

    rgb = np.array([r, g, b]).reshape(1, -1)

    _, indices = SRB2_PALETTE_KD.query(rgb)

    # Replace pixels with nearest palette colours
    closest = SRB2_PALETTE[indices[0]]

    return closest

'''
Loop over each pixel in an image and map it to the *closest* colour in the palette.
'''
def map_to_palette(frame: Image, transparency_index:int = None):     
    frame_temp = frame.convert("RGBA")
    pixels = frame_temp.load()
    
    width, height = frame_temp.size
    
    for y in range(height):
        for x in range(width):
            original_pixel = pixels[x, y]
            _,_,_,a = original_pixel

            if(a == 0 or (transparency_index is not None and original_pixel[3] == transparency_index)):
                pixels[x, y] = (0, 0, 0, 0)
                continue

            new_pixel = closest_palette_colour_numpy(original_pixel)
            pixels[x, y] = (new_pixel[0], new_pixel[1], new_pixel[2], a)

    return frame_temp
'''
PALETTE HANDLING
'''

'''
EMOTE HANDLING
'''
def save_config(text: str, path: Path):
    configured_emote = path / EMOTE_DEF_NAME
    configured_emote.write_text(text)

def save_static_config(emote_name: str, path: Path):
    save_config(f"{EMOTE_DEF_NAME_KEY} = {emote_name}\n{EMOTE_DEF_TICS_KEY} = 1", path)

def save_animated_config(frame_delay: int, emote_name: str, path: Path):
    tics_delay = get_frame_delay_in_tics(frame_delay)    
    save_config(f"{EMOTE_DEF_NAME_KEY} = {emote_name}\n{EMOTE_DEF_TICS_KEY} = {tics_delay}", path)

def save_atlas_config(names: list[str], size: tuple, rows_and_columns: tuple, path: Path):
    width, height = size
    rows, columns = rows_and_columns

    # e.g. Emote1 = joy
    emote_names = [f"Emote{i+1} = {name}" for i, name in enumerate(names)]
    emote_names_str = "\n".join(map(str, emote_names))

    configured_atlas = path / ATLAS_DEF_NAME
    configured_atlas.write_text(
        f"Rows = {rows}\nColumns = {columns}\nWidth = {width}\nHeight = {height}\n\n{emote_names_str}"
    )

def map_image(img: Image, transparency_index:int = None):
    global REMAP_IMAGE

    if (REMAP_IMAGE):
        return map_to_palette(img, transparency_index)
    else:
        return img
        

def remap_static_image(png: Path):
    #e.g. joy
    return map_image(Image.open(png))

def _static_image(png: Path):
    emote_name = png.parent.name

    if (not validaite_emote_name(emote_name)):
        return

    new_path = create_path_for_emote(emote_name)

    save_static_config(emote_name, new_path)

    # remap the PNG and save
    mapped_frame = remap_static_image(png)
    mapped_frame.save(f"{str(new_path)}/FRAME001.png", format="PNG")

    print(f"Saved static emote '{emote_name}!")

async def handle_static_image(png: Path):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _static_image, png)
    except Exception as e:
        print(f"Something went wrong processing {png}: {e}")

def _gif(gif: Path):
    gif_img = Image.open(gif)

    #e.g kekw
    emote_name = gif.parent.name

    if(not validaite_emote_name(emote_name)):
        return

    frames = gif_img.n_frames

    if (frames > MAX_EMOTE_FRAMES):
        print(f"{emote_name} has more than {MAX_EMOTE_FRAMES} frames, skipping.")
        return
    
    # GIFs have a transparency index in their metadata
    transparency_index = ('transparency' in gif_img.info) and gif_img.info['transparency'] or None
    
    # Why?
    if (frames == 1):
        # Passively-aggressively inform them
        print(f'{gif} only has one frame..')

        _static_image(gif)
    elif (frames <= 0):
        print(f'{gif} doesn\'t have any frames! Wow! Skipping!')
        return
    else:
        # Save the emote config FIRST, then the frames
        new_path = create_path_for_emote(emote_name)
        
        # 10ms is a magic number, it's just in case a frame doesn't have any delay
        delays = [frame.info.get('duration', 10) for frame in ImageSequence.Iterator(gif_img)]
        
        if not delays:
            frame_delay = 1
        else:
            # Most GIFs don't have consistent frame delays.. get the delay in each frame, and average it
            frame_delay = sum(delays) / len(delays)

        save_animated_config(frame_delay, emote_name, new_path)
        
        for i in range(gif_img.n_frames):
            gif_img.seek(i)
            mapped_frame = map_image(gif_img, transparency_index)

            # SRB2 2.1> supports PNG patches, thank God
            mapped_frame.save(f"{str(new_path)}/FRAME{i:03d}.png", format="PNG")

        print(f"Saved animated emote '{emote_name}', with {frames} frames!")

async def handle_gif(gif: Path):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _gif, gif)
    except Exception as e:
        print(f"Something went wrong processing {gif}: {e}")

def _atlas(images: list[Path]):
    atlas_name = images[0].parent.name
    print(f"Parsing atlas for '{atlas_name}'...")
    
    # Using the first image found as a reference point
    with Image.open(images[0]) as ref_image:
        ref_size = ref_image.size
    
    # In order for this to work, each image has to have the same width x height
    culprits = [img_path for img_path in images if Image.open(img_path).size != ref_size]

    if (len(culprits) > 0):
        print(f"{len(culprits)} images found that aren't {ref_size[0]}x{ref_size[1]}:")
        for culprit in culprits:
            culprit_img = Image.open(culprit)
            print(f"\t{culprit.name}: {culprit_img.width}x{culprit_img.height}")
        print(f"Skipping atlas, '{atlas_name}'")
        return
    
    # loop over each image, remap, and add to the dictionary
    remapped_images = {}
    for img_path in images:
        if (not validaite_emote_name(img_path.stem)):
            continue
        remapped_images[img_path.stem] = remap_static_image(img_path)

    remapped_images_list = list(remapped_images.values())
    num_images = len(remapped_images_list)
    width, height = ref_size

    # Start with a minimum of .. 4 emotes, before the atlas needs more than one row
    if (num_images < 5):
        atlas_size = ((num_images * width), height)
    else:
        # Keep the atlas as square as possible, this isn't a requirement, it's just convenient
        # find the smallest square number, so sqrt the length of images
        # 7 images? sqrt 7 = 3 (rounded up). so 3 rows, 3 columns. 7 images, with 2 empty slots

        # math.ceil adds more space than needed, so manually round up
        columns = int((num_images ** 0.5) + 0.5)
        rows = (num_images + columns - 1) // columns
        atlas_size = (columns * width, rows * height)
    
    print(f"Generating {atlas_size[0]}x{atlas_size[1]} atlas for {num_images} emotes..")
    atlas = create_atlas(remapped_images_list, atlas_size, (rows, columns))

    # alright, atlas is made, create the config to go with it
    atlas_path = create_path_for_emote(atlas_name)

    # and save
    save_atlas_config(list(remapped_images.keys()), ref_size, (rows, columns), atlas_path)
    atlas.save(f"{str(atlas_path)}/{ATLAS_IMAGE_NAME}", format="PNG")

    print(f"Saved atlas emote '{atlas_name}', with {num_images} emotes!")

async def handle_atlas(images: list[Path]):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _atlas, images)
    except Exception as e:
        print(f"Something went wrong processing atlas images in {images[0].parent}: {e}")


'''
Parse each subfolder in the provided directory and check for three scenarios.

1. Animated emote (GIF, ideally)
 - Dump each frame (PNG), translate to palette (if enabled), and configure
2. Static image (PNG, ideally)
 - Translate to palette (if enabled), and configure
3. Multiple static images (PNGs, ideally)
 - Create a new image atlas, translate each image to a palette (if enabled), and configure

The emote configurations will be dynamically built depending on the scenario.
'''
async def parse_folder(folder: Path):
    gifs = []
    images = []

    def is_file_image(file: Path):
        try:
            with Image.open(file) as img:
                format = img.format
                if format == "GIF":
                    gifs.append(file)
                elif format == "PNG" or format == "JPG":
                    images.append(file)
        except Exception:
            pass
    
    # Loop over file in the folder (if it's an image) and append to the appropriate list
    tasks = []
    for file in folder.iterdir():
        if not file.is_file():
            continue

        task = asyncio.to_thread(is_file_image, file)
        tasks.append(task)

    await asyncio.gather(*tasks)

    valid = False
    if (len(gifs) > 1):    
        # Too many GIFs, expecting just one
        print(f"Too many GIFs found in '{folder.name}', there only needs to be one. Skipping.")
        return
    elif len(gifs) == 1:
        # Handle GIF
        valid = True
        await handle_gif(gifs[0])
        return
        
    if (len(images) > 0):
        if (len(images) > 1):
            # Handle emote atlas
            valid = True
            await handle_atlas(images)
        else:
            # Handle static emote
            valid = True
            await handle_static_image(images[0])
    
    if (not valid):
        print(f'{folder} doesn\'t have any valid images, skipping')
        return

'''
EMOTE HANDLING
'''

# Let's start here
async def main():
    import argparse

    parser = argparse.ArgumentParser(epilog='Use the \'examples\' folder as a reference point.',
                                     description='''
Takes a folder full of subfolders, each containing images for emotes, and builds a single PK3.
Intended for use with the custom DRRR build, RadioRacers.
''')
    parser.add_argument('--folder', help='The folder containing the emote image subdirectories')
    parser.add_argument('--palette', type=str, help='Path to SRB2 palette PNG file, will translate each image to SRB2\'s palette as closely as possible.')
    parser.add_argument('--name', type=str, help='Filename for the exported PK3')

    if (len(sys.argv) < 2):
        parser.print_help()
        sys.exit()
    
    args = parser.parse_args()
    directory = args.folder

    try:
        if (args.palette): 
            global REMAP_IMAGE

            # Does the palette exist?
            if (Path(args.palette).exists()):        
                load_palette(Path(args.palette))
                REMAP_IMAGE = True
                print(f"Remapping enabled - image processing can take significantly longer, please be patient!")
            else:
                raise RadioLibException('f{args.remap} missing - is it in the same folder as the script?')
        
        global FINAL_PK3_NAME
        if(args.name):
            FINAL_PK3_NAME = args.name
        print(f"Loading emotes from {Path(directory)}:\n")

        # Loop over each folder in the provided directory and parse each one
        subdirectories = [dir for dir in Path(directory).iterdir() if dir.is_dir()]

        if(len(subdirectories) < 1):
            raise RadioLibException("You need at least one subfolder.")
        
        # Make the tmp directory
        temp_dir = Path.cwd() / tmp_folder
        if(not temp_dir.exists()):
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            ## If it already exists, delete everything inside it
            empty_temp(temp_dir)

        for subdir in subdirectories:
            await parse_folder(subdir)

        # Once we're done, create the PK3
        if (not any(temp_dir.iterdir())):
            print(f"\nNo valid emotes found in {Path(directory)}, not creating PK3.")
        else:
            create_pk3()
            print("\nDone! Check the 'build' folder for your PK3")
    except Exception as e: 
        print(e)
        return EX_FAIL


if __name__ == "__main__":
    asyncio.run(main())
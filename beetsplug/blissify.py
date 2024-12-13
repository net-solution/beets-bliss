"""Beets plugin to analyse music with bliss and create playlists, based off
blissify-rs.
"""

import numpy as np
from beets.library import Library
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, decargs, input_options, input_yn
from bliss_audio import Song
from scipy.spatial import KDTree

# reference:
# album
# analysis
# analysis_dict
# artist
# duration
# genre
# path
# title
# track_number


def analyse_library(lib: Library, opts, args):
    """Analyse beets library with bliss"""

    for item in lib.items():
        song_path = item.path.decode("utf-8")
        song = Song(song_path)

        bliss_data = r"\␀".join(map(str, song.analysis))

        # Check if song has already been analysed
        # TODO: analyse anyway if `--force` is set
        if hasattr(item, "bliss_data"):
            print("Item already analysed, skipping...")
            continue

        item.bliss_data = bliss_data
        item.store()

        print(f"Analysed: {song.title} from {song.album} by {song.artist}")


def select_song(results, page_size=10):
    current_page = 0
    total_pages = (len(results) + page_size - 1) // page_size

    while True:
        start = current_page * page_size
        end = min(start + page_size, len(results))

        print(f"\nPage {current_page+1} of {total_pages}")
        for i, item in enumerate(results[start:end], start=start):
            item_id = f"[ID: {item.id:_>6}]"
            print(f"{i + 1}: {item.artist} - {item.title} {item_id:>18}")

        if len(results) > 1:
            print("\nEnter a number to choose a song, or pick an option")
            choice = input_options(
                "npq",
                numrange=(start + 1, end),
                default=start + 1,
                prompt=f"[N]ext page, [P]revious page, [Q]uit, ({start+1}-{end}):",
            )
        else:
            choice = 1

        if isinstance(choice, int) and start + 1 <= choice <= end:
            result_index = choice - 1

            selection = results[result_index]
            print(f"Selected: {selection.artist} - {selection.title}")
            confirm = input_yn("Are you sure? [Y/n]:")

            if confirm:
                return selection

        elif choice == "p":
            current_page = (current_page - 1) % total_pages
        elif choice == "n":
            current_page = (current_page + 1) % total_pages
        elif choice == "q":
            break

    return None


def save_kdtree(tree, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(tree, f)


def generate_playlist(lib: Library, opts, args):
    """Generate a playlist of similar songs."""
    if opts.compare:
        results = list(lib.items(input("Enter a query: ")))
        song1 = select_song(results, page_size=6)
        results = list(lib.items(input("Enter a query: ")))
        song2 = select_song(results, page_size=6)

        if song1 is not None and song2 is not None:
            song1_data = np.array(song1.bliss_data.split(r"\␀"), dtype=float)
            print(song1_data)
            song2_data = np.array(song2.bliss_data.split(r"\␀"), dtype=float)
            distance = np.linalg.norm(song1_data - song2_data)

            print(f"Distance between songs: {distance}")
            return

    playlist = []

    query = decargs(args)

    results = list(lib.items(query))
    if not results:
        print("No matching songs found!")
        return

    song = select_song(results, page_size=6)
    if song is None:
        print("Good bye")
        return

    def get_bliss_data(item):
        if not hasattr(item, "bliss_data"):
            print("Item has no bliss data! Have you analysed the library?")
            return None
        # return np.array(item.bliss_data.split("\\␀"), dtype=float)
        return item.bliss_data.split(r"\␀")

    song_analysis = [
        np.array(s.bliss_data.split(r"\␀"), dtype=float) for s in lib.items()
    ]

    tree = KDTree(song_analysis)

    distances, indices = tree.query(song.bliss_data.split(r"\␀"), k=10)
    print("Nearest songs:")
    for dist, idx in zip(distances, indices):
        similar_song = lib.items(f"id:{idx}")[0]
        print(f"Song {similar_song.title} with distance {dist:.3f}")

    return
    initial_bliss_data = get_bliss_data(song)

    for item in lib.items():
        print(f"retrieving data for track: {item.title}")
        bliss_data = get_bliss_data(item)

        distance = np.linalg.norm(initial_bliss_data - bliss_data)

        break


class BlissifyPlugin(BeetsPlugin):
    def __init__(self):
        super().__init__()

    def commands(self):
        blissify_scan = Subcommand(
            "blissify_scan",
            help="analyse the music library with bliss",
        )
        blissify_scan.parser.add_option(
            "--force",
            help="re-analyse all songs, overwriting existing data",
        )
        blissify_scan.func = analyse_library

        blissify = Subcommand(
            "blissify",
            help="create playlist with bliss",
        )
        blissify.parser.add_option(
            "--distance",
            default=0.5,
            help="make playlist with closest song to all previous songs",
        )
        blissify.parser.add_option(
            "--seed",
            default=False,
            help="make playlist with closest song to all previous songs",
        )
        blissify.parser.add_option(
            "--count",
            default=32,
            help="size of playlist",
        )
        blissify.parser.add_option(
            "--compare",
            default=False,
            help="get distance between two songs, nothing more or less",
        )
        blissify.func = generate_playlist

        return [blissify_scan, blissify]

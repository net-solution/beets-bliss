"""Beets plugin to analyse music with bliss and create playlists, based off
blissify-rs.
"""

import pickle
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from optparse import OptionParser

import numpy as np
from beets.library import Library
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, UserError, decargs, input_options, input_yn
from bliss_audio import Song
from scipy.spatial import KDTree
from tqdm import tqdm

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


class PassthruParser(OptionParser):
    def parse_args(self, args=None, values=None):
        return self, args


class BlissCommand(Enum):
    SCAN = ("scan", "analyse the beets library with bliss")
    PLAYLIST = ("playlist", "create playlist with bliss")
    COMPARE = ("compare", "calculate distance between two songs, with bliss")

    def __init__(self, cmd, desc):
        self._value_ = cmd
        self.desc = desc

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, BlissCommand):
            return self.value == other.value
        return False

    def __str__(self):
        return self.value


class BlissifyPlugin(BeetsPlugin):
    """Blissify plugin for Beets."""

    def __init__(self):
        super().__init__()

    def commands(self):
        bliss_cmd = Subcommand(
            "bliss", help="run bliss subcommands", parser=PassthruParser()
        )
        bliss_cmd.func = self.bliss_handler

        return [bliss_cmd]

    def bliss_handler(self, lib, opts, args):
        if not args:
            raise UserError(f"""\
    Usage: beet bliss <subcommand> [options]
                
    Available subcommands:
    \t{"\n\t".join(str(cmd) for cmd in BlissCommand)}
""")

        args = decargs(args)
        subcommand = args[0]
        subcommand_args = args[1:]

        match subcommand:
            case BlissCommand.SCAN:
                self.bliss_scan(lib, subcommand_args)
            case BlissCommand.PLAYLIST:
                self.bliss_playlist(lib, subcommand_args)
            case BlissCommand.COMPARE:
                self.bliss_compare(lib, subcommand_args)
            case _:
                raise UserError(f"Unknown subcommand: {subcommand}")

    def bliss_scan(self, lib, args):
        parser = OptionParser(
            usage=f"beet bliss {BlissCommand.SCAN} [options]",
            description="analyse the beets library with bliss",
        )
        parser.add_option(
            "-f",
            "--force",
            action="store_true",
            default=False,
            help="re-analyse all files, overwriting existing data",
        )
        opts, _ = parser.parse_args(args)

        self.analyse_library(lib, opts)

    def bliss_playlist(self, lib, args):
        parser = OptionParser(
            usage=f"beet bliss {BlissCommand.PLAYLIST} [options] <query>",
            description="create playlist with bliss",
        )
        parser.add_option(
            "-d",
            "--distance",
            type="float",
            default=0.5,
            help="make playlist with closest song to all previous songs",
        )
        parser.add_option(
            "-s",
            "--seed",
            action="store_true",
            default=False,
            help="make playlist with closest song to all previous songs",
        )
        parser.add_option(
            "-n",
            "--count",
            type="int",
            default=32,
            help="number of songs to make the playlist",
        )
        parser.add_option(
            "--query",
            default="",
            help='query the music library before making playlist (e.g. "^christmas")',
        )
        opts, sub_args = parser.parse_args(args)
        self.generate_playlist(lib, opts, sub_args)

    def bliss_compare(self, lib, args):
        parser = OptionParser(
            usage=f"beet bliss {BlissCommand.COMPARE} [options] <query>",
            description="calculate distance between two songs, with bliss",
        )
        opts, sub_args = parser.parse_args(args)
        self.compare_songs(lib, opts, sub_args)

    def _compute_bliss_data(self, args):
        """Compute bliss data for given path."""
        song_id, song_path = args
        song = Song(song_path)
        bliss_data = r"\␀".join(map(str, song.analysis))

        return song_id, bliss_data

    def _store_bliss_data(self, lib: Library, item_id: int, bliss_data: str):
        """Store computed bliss data to the beets database."""
        item = lib.get_item(item_id)
        item.bliss_data = bliss_data
        item.store()

    def analyse_library(self, lib: Library, opts):
        """Analyse beets library with bliss."""

        print("Analysing library...")

        tasks = [
            (item.id, item.path.decode("utf-8"))
            for item in lib.items()
            if opts.force
            or not hasattr(item, "bliss_data")
            or len(item.bliss_data.split(r"\␀")) != 20
        ]

        if not tasks:
            print("Library up-to-date!")
            return

        with ProcessPoolExecutor() as executor:
            for item_id, bliss_data in tqdm(
                executor.map(self._compute_bliss_data, tasks), total=len(tasks)
            ):
                self._store_bliss_data(lib, item_id, bliss_data)

        print("Analysis complete!")

    def select_song(self, results, page_size=10):
        current_page = 0
        total_pages = (len(results) + page_size - 1) // page_size

        while True:
            start = current_page * page_size
            end = min(start + page_size, len(results))

            print(f"\nPage {current_page + 1} of {total_pages}")
            for i, item in enumerate(results[start:end], start=start):
                item_id = f"[ID: {item.id:_>6}]"
                print(
                    f"{i + 1}: {item.artist:40} - {item.title:50} {item_id:>18}"
                )

            if len(results) > 1:
                print("\nEnter a number to choose a song, or pick an option")
                choice = input_options(
                    "npq",
                    numrange=(start + 1, end),
                    default=start + 1,
                    prompt=f"[N]ext page, [P]revious page, [Q]uit, ({start + 1}-{end}):",
                )
            else:
                choice = 1

            if isinstance(choice, int) and start + 1 <= choice <= end:
                result_index = choice - 1

                selection = results[result_index]
                print(f"\nSelected: {selection.artist} - {selection.title}")
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

    def save_kdtree(self, tree, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(tree, f)

    def get_nearest_songs(self, tree, query_vector, song_ids, k, threshold):
        nearest_songs = []
        offset = 0  # offset for already processed indices
        max_requested = k

        if threshold:  # get a few more results if using threshold
            k = int(k * 1.5)

        # but still using while loop, in case we didnt get enough extra
        while len(nearest_songs) < max_requested:
            distances, indices = tree.query(
                query_vector, k=k + offset, workers=-1
            )

            # skip the already processed songs
            indices = indices[offset:]
            distances = distances[offset:]

            new_songs = [
                (song_ids[idx], distances[i])
                for i, idx in enumerate(indices)
                if distances[i] > threshold
            ]

            nearest_songs.extend(new_songs)
            offset += k

            # exit if we exhaust all neighbors
            if len(indices) == 0:
                break

        return nearest_songs[:max_requested]

    def compare_songs(self, lib, opts, args):
        results = list(lib.items(input("Enter a query: ")))
        song1 = self.select_song(results, page_size=6)
        results = list(lib.items(input("Enter a query: ")))
        song2 = self.select_song(results, page_size=6)

        if song1 is not None and song2 is not None:
            song1_data = np.array(song1.bliss_data.split(r"\␀"), dtype=float)
            song2_data = np.array(song2.bliss_data.split(r"\␀"), dtype=float)
            distance = np.linalg.norm(song1_data - song2_data)

            print(f"Distance between songs: {distance}")

    def generate_playlist(self, lib: Library, opts, args):
        """Generate a playlist of similar songs."""
        query = decargs(args)

        results = list(lib.items(query))
        if not results:
            print("No matching songs found!")
            return

        seed_song = self.select_song(results, page_size=6)
        if seed_song is None:
            print("Good bye")
            return

        music_library = lib.items()
        try:
            seed_analysis = seed_song.bliss_data.split(r"\␀")
            song_analysis = [
                np.array(s.bliss_data.split(r"\␀"), dtype=float)
                for s in music_library
            ]
        except AttributeError:
            print("ERROR: Missing bliss data, please (re)analyse the library!")
            return None

        # Store song ids in same order as analysis vectors
        song_ids = [s.id for s in music_library]

        # Build KDTree
        tree = KDTree(song_analysis)

        nearest_songs = self.get_nearest_songs(
            tree, seed_analysis, song_ids, opts.count, opts.threshold
        )

        print("\nNearest songs:")
        for song_id, distance in nearest_songs:
            song = lib.get_item(song_id)
            print(f"  * {song.artist:40} - {song.title:50} (dist: {distance})")

        return

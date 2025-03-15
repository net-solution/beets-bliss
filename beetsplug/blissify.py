"""Beets plugin to analyse music with bliss and create playlists, based off
blissify-rs.
"""

from binascii import hexlify
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from optparse import OptionParser

import numpy as np
import requests
from beets.library import Library
from beets.plugins import BeetsPlugin
from beets.ui import (
    Subcommand,
    UserError,
    config,
    decargs,
    input_options,
    input_yn,
)
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
        help_message = ""
        help_message += "Usage: beet bliss <subcommand> [options]\n\n"
        help_message += "Available subcommands:\n"
        help_message += (
            f"\t{'\n\t'.join(f'{cmd:12} - {cmd.desc}' for cmd in BlissCommand)}"
        )

        if not args:
            raise UserError(help_message)

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
                raise UserError(help_message)

    def bliss_scan(self, lib, args):
        parser = OptionParser(
            usage=f"beet bliss {BlissCommand.SCAN} [options]",
            description=BlissCommand.SCAN.desc,
        )
        parser.add_option(
            "-f",
            "--force",
            action="store_true",
            default=False,
            help="re-analyse all files, overwriting existing data",
        )
        opts, _ = parser.parse_args(args)

        self._analyse_library(lib, opts)

    def bliss_playlist(self, lib, args):
        parser = OptionParser(
            usage=f"beet bliss {BlissCommand.PLAYLIST} [options] <query>",
            description=BlissCommand.PLAYLIST.desc,
        )
        parser.add_option(
            "-q",
            "--quiet",
            action="store_true",
            default=False,
            help="quiet mode, don't ask for any input",
        )
        parser.add_option(
            "-w",
            "--walk",
            action="store_true",
            default=False,
            help="find nearest song for each song we add\
            (walk through the similar songs)",
        )
        parser.add_option(
            "-s",
            "--shuffle",
            action="store_true",
            default=False,
            help="use weighted randomness to 'shuffle' the mix. helps give unique playlist each time while still being similar songs",
        )
        parser.add_option(
            "-r",
            "--randomness",
            type="float",
            default=0,
            help="add randomness",
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
            help=r'query the music library before making playlist (e.g. "^christmas")',
        )
        parser.add_option(
            "-v",
            "--verbose",
            action="store_true",
            default=False,
            help="verbose output",
        )
        opts, sub_args = parser.parse_args(args)
        self._generate_playlist(lib, opts, sub_args)

    def bliss_compare(self, lib, args):
        parser = OptionParser(
            usage=f"beet bliss {BlissCommand.COMPARE} [options] <query>",
            description=BlissCommand.COMPARE.desc,
        )
        opts, sub_args = parser.parse_args(args)
        self._compare_songs(lib, opts, sub_args)

    def _compare_songs(self, lib, opts, args):
        song1 = self._select_song(lib, page_size=6)
        song2 = self._select_song(lib, page_size=6)

        if song1 is not None and song2 is not None:
            song1_data = np.array(song1.bliss_data.split(r"\␀"), dtype=float)
            song2_data = np.array(song2.bliss_data.split(r"\␀"), dtype=float)
            distance = np.linalg.norm(song1_data - song2_data)

            print(f"Distance between songs: {distance}")

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

    def _analyse_library(self, lib: Library, opts):
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

    def _select_song(
        self, lib: Library, results=None, page_size=10, first=False
    ):
        if first:
            print("Enter a number to choose a song, or pick an option:")
            print("[M]ore songs, [S]earch, [H]elp, [Q]uit, [#] song number\n")
            print(f"{'#':>4} | {'Artist':24} | {'Album':16} | Title")
            print(f"{'=' * 4}=|={'=' * 24}=|={'=' * 16}=|={'=' * 32}=")

        if results is None:
            results = list(lib.items(input("Enter search query: ")))

        current_page = 0
        total_pages = (len(results) + page_size - 1) // page_size

        while True:
            start = current_page * page_size
            end = min(start + page_size, len(results))

            for i, item in enumerate(results[start:end], start=start):
                print(
                    f"{(i + 1):>4} | {item.artist[:24]:24} | {item.album[:16]:16} | {item.title[:32]:32}"
                )

            choice = input_options(
                "mshq",
                numrange=(1, end),
                default=start + 1,
                prompt="\nChoice:",
            )

            if isinstance(choice, int) and 1 <= choice <= end:
                result_index = choice - 1

                selection = results[result_index]
                print(f"\nSelected: {selection.artist} - {selection.title}")

                confirm = input_yn("Are you sure? [Y/n]:")

                if confirm:
                    return selection
            elif choice == "m":
                current_page += 1
            elif choice == "h":
                print("===================================================")
                print("Options:")
                print()
                print("[M]ore songs - print next batch of songs")
                print("[S]earch     - enter a new search query for songs")
                print("[H]elp       - print available options")
                print("[Q]uit       - quit beets")
                print("===================================================")
                print()
            elif choice == "q":
                break

            elif choice == "s":
                return self._select_song(lib, page_size=page_size)

        return None

    def _deduplicate_mask(self, distances, indices, threshold=1e-9):
        """Return indices of unique distances (within threshold)."""
        # sort distances/indices arrays by distance
        sorted_order = np.argsort(distances)
        sorted_distances = distances[sorted_order]
        sorted_indices = indices[sorted_order]

        # find unique distances, aka compare each distance to the one before
        # it, and if the difference is super tiny, then it is False
        unique_mask = np.diff(sorted_distances, prepend=-np.inf) > threshold
        # this will also only keep distances above 0 (the seed song). maybe
        # 0.0somethingsmall  to also count songs that are basically    the
        # same
        unique_mask &= sorted_distances > 0.005

        unique_indices = sorted_indices[unique_mask]
        unique_distances = sorted_distances[unique_mask]

        return unique_indices, unique_distances

    def _get_nearest_songs(self, tree, seed_vector, k, rng=False):
        """Return k indices of nearest songs to seed song."""
        k_multiplier = 10 if rng else 1

        distances, indices = tree.query(
            seed_vector, k=k * k_multiplier, workers=-1
        )

        unique_indices, unique_distances = self._deduplicate_mask(
            distances, indices
        )

        nearest_songs = np.column_stack((
            unique_indices,
            unique_distances,
        ))

        if rng:
            # Weighted randomness:
            #   - first invert the distance, (with tolerance to prevent
            #   division by zero);
            #   - then normalize the distances so they sum to 1 (allows them to
            #   become probability weights)
            weights = 1 / (unique_distances + 1e-5)
            weights /= weights.sum()

            # choose K songs from our song indices
            weighted_indices = np.random.choice(
                unique_indices.shape[0], size=k, replace=False, p=weights
            )

            nearest_songs = nearest_songs[weighted_indices]

        return nearest_songs

    def _random_walk_playlist(self, tree: KDTree, seed_vector, k, explore, rng):
        playlist = []
        current_vector = seed_vector

        def jump_to_song(tree):
            random_index = np.random.choice(tree.n)
            random_distance = np.linalg.norm(
                current_vector - tree.data[random_index]
            )
            playlist.append((random_index, random_distance))

            return tree.data[random_index]

        for _ in range(k):
            if np.random.rand() < explore:
                # Jump to a completely random song
                current_vector = jump_to_song(tree)
            else:
                distances, indices = tree.query(
                    current_vector, k=k * 3, workers=-1
                )
                unique_indices, unique_distances = self._deduplicate_mask(
                    distances, indices
                )
                valid_pairs = [
                    (idx, dist)
                    for idx, dist in zip(unique_indices, unique_distances)
                    if idx not in [p[0] for p in playlist]
                ]
                if not valid_pairs:
                    current_vector = jump_to_song(tree)
                    continue

                valid_indices, valid_distances = zip(*valid_pairs)

                if rng:
                    weights = 1 / (np.array(valid_distances) + 1e-5)
                    weights /= weights.sum()
                    weighted_index = np.random.choice(
                        len(valid_indices), p=weights
                    )
                else:
                    # take second index, as first is the query vector
                    weighted_index = 1

                chosen_index = valid_indices[weighted_index]
                chosen_distance = valid_distances[weighted_index]
                current_vector = tree.data[chosen_index]

                playlist.append((chosen_index, chosen_distance))

        return playlist

    def _generate_playlist(self, lib: Library, opts, args):
        """Generate a playlist of similar songs."""
        query = decargs(args)

        results = list(lib.items(query))
        if not results:
            print("No matching songs found!")
            return

        if opts.quiet:
            seed_song = results[0]
        else:
            seed_song = self._select_song(lib, results, page_size=6, first=True)
            if seed_song is None:
                return

        music_library = lib.items(opts.query)
        try:
            song_analysis = [
                np.array(s.bliss_data.split(r"\␀"), dtype=float)
                for s in music_library
            ]
        except AttributeError:
            print("ERROR: Missing bliss data, please (re)analyse the library!")
            return None

        seed_analysis = np.array(seed_song.bliss_data.split(r"\␀"), dtype=float)

        # perturb seed data
        # rng = np.random.default_rng()
        # sigma = opts.randomness
        # seed_analysis = seed_analysis + np.random.normal(
        #     0, sigma, size=seed_analysis.shape
        # )

        # Store song ids in same order as analysis vectors
        song_ids = np.array([s.id for s in music_library])

        # Build KDTree
        tree = KDTree(song_analysis)

        if opts.walk:
            nearest_songs = self._random_walk_playlist(
                tree,
                seed_analysis,
                opts.count,
                opts.randomness,
                opts.shuffle,
            )
        else:
            nearest_songs = self._get_nearest_songs(
                tree, seed_analysis, opts.count, opts.shuffle
            )

        # Open file for saving
        if self.config["save_playlist"]:
            filename = "nearest_songs.m3u"
            file = open(filename, "w")

        subsonic_url = config["subsonic"]["url"]
        subsonic_user = config["subsonic"]["user"].as_str()
        subsonic_pass = config["subsonic"]["pass"].as_str()

        encpass = hexlify(subsonic_pass.encode()).decode()
        payload = {
            "u": subsonic_user,
            "p": f"enc:{encpass}",
            "v": "1.15.0",
            "c": "beets",
            "f": "json",
        }
        subsonic_endpoint = f"{subsonic_url}/rest/savePlayQueue"

        # clear queue
        response = self.send_request(subsonic_endpoint, payload)

        subsonic_ids = []

        print("\nNearest songs:\n")
        print(f"{'Artist':35} | {'Song':35} | Distance")
        print(f"{'=' * 35}=|={'=' * 35}=|==========")

        # Process nearest songs (print/save/queue)
        # seed_song_id = int(song_ids[int()])
        for idx, distance in nearest_songs:
            song_id = int(song_ids[int(idx)])
            song = lib.get_item(song_id)

            output = (
                f"{song.artist[:35]:35} | {song.title[:35]:35} | {distance:.3f}"
            )

            subsonic_ids.append(song.subsonic_id)

            if self.config["save_playlist"]:
                file.write(output)

            print(output)

        payload["id"] = subsonic_ids

        response = self.send_request(subsonic_endpoint, payload)
        if response:
            print("Queue sent to subsonic server")
        else:
            print("Failed to send queue!")

        if self.config["save_playlist"]:
            file.close()
            print(f"Songs saved to {filename}")

        return

    def send_request(self, url, payload):
        try:
            response = requests.get(url, params=payload, timeout=5.0)
            response.raise_for_status()

            json_response = response.json()

            if "subsonic-response" not in json_response:
                self._log.error("Invalid response: missing 'subsonic-response'")
                return None

            subsonic_response = json_response["subsonic-response"]
            if subsonic_response["status"] == "ok":
                return subsonic_response
            else:
                error = subsonic_response.get("error", {})
                error_msg = error.get("message", "Unknown error")
                error_code = error.get("code", "Unknown code")
                self._log.error(f"Subsonic error {error_code}: {error_msg}")
                return None

        except requests.exceptions.RequestException as error:
            self._log.error(f"Request failed: {error}")
            return None
        except ValueError as error:
            self._log.error(f"Failed to parse JSON: {error}")
            return None

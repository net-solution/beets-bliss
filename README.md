# beets bliss plugin

beets-bliss is a plugin for [beets](https://github.com/beetbox/beets) that
generates playlists of similar songs (diy spotify radio!). It uses the python
bindings for the [bliss](https://pypi.org/project/bliss-audio/) library to
determine song similarity.

inspired by and based on [blissify-rs](https://github.com/polochon-street/blissify-rs)

# installation

Install the plugin with:

```
$ pip install -U --force-reinstall --no-deps git+https://github.com/net-solution/beets-bliss.git
```

This should also install its dependencies (`tqdm`, `bliss-audio`)

# usage

To analyse the music library, run

```
$ beet bliss scan
```

This scans the database for new songs to analyse, and stores the computed data
into a `bliss_data` field per song. It could take a few hours depending on your
library, for me it was about an hour for >9k songs.

You can run the command again to analyse any new songs you add, and it'll skip
everything it has already analysed. If you need to re-analyse the library for
some reason, use the `-f` (or `--force`) option:

```
$ beet bliss scan -f
```

## make playlists

Generate a playlist with the `playlist` subcommand:

```
$ beet bliss playlist [options] <query>
```

This will prompt you to pick a song, from which it will find the top similar
songs. You can enter any beets query to filter the results, or use the `--quiet`
(`-q`) flag to select the first result without asking.

By default, it will create a playlist of 32 songs, saved to a `.m3u` playlist
with a unique name. You can specify the name with `--output <name>` (or `-o
<name>`).

You can also set the number of songs for the playlist with `--count NUM` (`-n
NUM`).

## subsonic integration

The plugin is designed to work seamlessly with a Subsonic compatible server,
when using the [beets-subsonic](https://github.com/arsaboo/beets-subsonic)
plugin by arsaboo.

Simply configure that plugin with your Subsonic server details, and run the
plugin to fetch the Subsonic IDs for each song (`beet subsonic_getids`).

`beet bliss` can then optionally use that configuration, to send the generated
playlist directly to the Subsonic queue, by setting `subsonic: yes` in your
beets config or using the `--subsonic` flag.

## configuration

Here are the available config options, set to their default values:

```
bliss:
    playlist_count: 32
    playlist_name: beets-radio
    walk: False
    shuffle: True
    randomness: 0
    subsonic: no

```

Note, most of these have a CLI equivalent flag, which will override whatever is
in the config file.

- **playlist-count**: The number of songs to use in the playlist. Equivalent to
  `--count NUM` or `-n NUM`

- **playlist_name**: The name of the saved `.m3a` playlist file. If the name
  already exists it will append the current date

- **walk**: 'Walk' through the similar songs, by getting each similar song to the
  last similar song

- **shuffle**: Randomise the song selection, weighted by their similarity.
  Equivalent to `--shuffle` or `-s`

- **randomness**: Amount of randomness to use. If non-zero, will set `shuffle:
yes`. In regular mode, it will influence the shuffle weighting (lower value=
  similar songs have most weighting, higher value= all songs have equal
  weighting). In walk mode, will influence probability of jumping to a
  completely random song. `--randomness PROB` or `-r PROB`

- **subsonic**: whether to attempt sending the songs to your Subsonic server, as
  configured for the [beets-subsonic](https://github.com/arsaboo/beets-subsonic)
  plugin. If it fails, it will save a playlist

- **something**:

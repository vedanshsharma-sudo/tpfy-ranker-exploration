import csv
from typing import Mapping, TypeVar, Generic, Dict
import dataclasses
import math
import os
import shutil
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass

from common.cms3.cms3_utils import get_cms_store
from common.config import TENANT
from common.config.constants import DataPath
from common.config.utils import data_path
from common.s3_utils import download_single_file, download_multiple_files
from common.time_utils import timestamp, get_diff_date
from model.content_meta_utils import enrich_metadata

MOVIE_FIELDS = [
    "content_type_id",
    "production_house_id",
    "content_provider_id",
    "studio_id",
    "studio_name",
    "year",
    "is_disneyplus",
    "language_ids",
    "genre_ids",
    "vip",
    "premium",
    "start_dt",
    "end_date",
    "duration",
    "title",
    "deleted",
    "sub_tagged",
    "hidden",
    "is_search_blocked",
    "priority",
    "parental_rating",
    "parental_rating_name",
    "status",
    "effective_start_dt",
    "s2avod_ts",
    "ett_ts_history",
]
TV_SHOW_FIELDS = [
    "content_type_id",
    "production_house_id",
    "studio_id",
    "studio_name",
    "channel_id",
    "channel_name",
    "is_disneyplus",
    "language_ids",
    "genre_ids",
    "vip",
    "premium",
    "start_dt",
    "end_date",
    "title",
    "deleted",
    "sub_tagged",
    "hidden",
    "is_search_blocked",
    "season_cnt",
    "episode_cnt",
    "priority",
    "parental_rating",
    "parental_rating_name",
    "season_start_dts",
    "ep_start_dts",
    "season_broadcast_dts",
    "s2avod_ts",
    "ett_ts_history",
    "show_type",
    "is_btv",
    "archived",
]
MATCH_FIELDS = [
    "content_type_id",
    "production_house_id",
    "content_provider_id",
    "studio_id",
    "studio_name",
    "language_ids",
    "genre_ids",
    "start_dt",
    "end_date",
    "title",
    "deleted",
    "hidden",
    "is_search_blocked",
    "priority",
    "parental_rating",
    "parental_rating_name",
    "simatch_id",
    "avs_season_id",
    "sports_season_id",
    "tournament_id",
    "game_id",
    "clip_type",
    "status",
    "vip",
    "premium",
    "duration",
    "teams",
    "live_start_dt",
]
EPISODE_FIELDS = [
    "content_type_id",
    "language_ids",
    "vip",
    "premium",
    "start_dt",
    "end_date",
    "deleted",
    "hidden",
    "is_search_blocked",
    "broadcast_date",
    "show_content_id",
    "season_id",
    "status",
]


def _coalesce(value, default):
    return value if value is not None else default


def _safe_get(d, a, b, log=False):
    if a not in d:
        return 0.0
    if b not in d[a]:
        return 0.0
    if log:
        return math.log(d[a][b] + 1)
    return d[a][b]


def _retry_if_use_hive(func, hive_connector_factory=None, **kwargs):
    if hive_connector_factory is not None:
        retry = 10
        while retry > 0:
            try:
                return func(hive_connector=hive_connector_factory(), **kwargs)
            except Exception as e:
                if retry > 0:
                    retry -= 1
                    print(e)
                    print("retry in 60 seconds")
                    time.sleep(60)
                else:
                    raise e
    else:
        return func(hive_connector=None, **kwargs)


@dataclass
class PopularityValue:
    dc: float = 0
    dv: float = 0
    wc: float = 0
    wv: float = 0
    mc: float = 0
    mv: float = 0


def load_discover_popularity(country, date_str):
    print(f"load popularity {country} {date_str}")
    s3_prefix = data_path(DataPath.S3_DISCOVER_POPULARITY_AGG, country) % date_str
    local_path = "discover_pop"
    os.makedirs(local_path, exist_ok=True)
    download_multiple_files(s3_prefix, local_path, ".csv", "pop")
    popularity = {}
    valid_tag_names = {f.name for f in dataclasses.fields(PopularityValue)}
    num_values = 0
    for filename in os.listdir(local_path):
        if not filename.endswith(".csv"):
            continue
        with open(os.path.join(local_path, filename), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                day, content_id, language_id, tag, value = row
                if tag not in valid_tag_names:
                    continue
                value = float(value)
                language_id = int(language_id)
                key = (content_id, language_id)
                if key not in popularity:
                    popularity[key] = PopularityValue()
                setattr(popularity[key], tag, value)
                num_values += 1

    print(
        f"loaded {len(popularity)} content-language popularity for country {country} date {date_str}; entries {num_values}"
    )
    shutil.rmtree(local_path)
    return popularity


def load_state_nolang_popularity_subtable(
    table, country, table_template, date_str: str, day
):
    s3_prefix = data_path(table_template, country) % date_str
    local_path = "state_pop"
    os.makedirs(local_path, exist_ok=True)
    download_multiple_files(s3_prefix, local_path, ".csv", "pop")
    num_values = 0
    if day not in {1, 7, 30}:
        raise Exception(f"illegal day {day}")
    for filename in os.listdir(local_path):
        if not filename.endswith(".csv"):
            continue
        with open(os.path.join(local_path, filename), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                state, content_id, count, percentile = row
                key = (state, content_id)
                count = float(count)
                percentile = float(percentile)
                if key not in table:
                    table[key] = PopularityValue()
                pv = table[key]
                if day == 1:
                    pv.dc = count
                    pv.dv = percentile
                elif day == 7:
                    pv.wc = count
                    pv.wv = percentile
                elif day == 30:
                    pv.mc = count
                    pv.mv = percentile
                num_values += 1
    print(
        f"loaded {num_values} values for state nolang pop day {day}; country {country} date {date_str}"
    )
    shutil.rmtree(local_path)


def load_state_nolang_popularity(country, date_str):
    print(f"load state nolang popularity {date_str}")
    popularity = {}
    load_state_nolang_popularity_subtable(
        popularity,
        country,
        DataPath.S3_DISCOVER_POPULARITY_STATE_NO_LANG_DAILY,
        date_str,
        1,
    )
    load_state_nolang_popularity_subtable(
        popularity,
        country,
        DataPath.S3_DISCOVER_POPULARITY_STATE_NO_LANG_WEEKLY,
        date_str,
        7,
    )
    load_state_nolang_popularity_subtable(
        popularity,
        country,
        DataPath.S3_DISCOVER_POPULARITY_STATE_NO_LANG_MONTHLY,
        date_str,
        30,
    )

    print(
        f"loaded {len(popularity)} state nolang pop content for country {country} {date_str}"
    )
    return popularity


T = TypeVar("T")


class MultiVersionData(Generic[T]):
    def __init__(self):
        self.versions_map = {}

    def add_version(self, version, data: T):
        self.versions_map[version] = data

    def get_version(self, version) -> T:
        return self.versions_map[version]

    def get_or_close_version(self, version) -> T:
        target_version = None
        if version in self.versions_map:
            target_version = version
        elif version > self.versions[-1]:
            target_version = self.versions[-1]
        elif version < self.versions[0]:
            target_version = self.versions[0]
        else:
            raise Exception("no matched version")
        return self.versions_map[target_version]

    def build(self):
        self.versions = sorted(self.versions_map.keys())
        return self


class Metadata:
    def __init__(
        self,
        date,
        countries,
        base_path="tmp/tpfy/%s",
    ):
        self.base_path = base_path
        self.countries = countries

        self.date = date
        self.end_ts = 86400 + timestamp(date)

        self.discover_popularity: Dict[
            str, MultiVersionData[Dict[(str, int), PopularityValue]]
        ] = {}  # keyed by country
        self.state_nolang_popularity: Dict[
            str, MultiVersionData[Dict[(str, str), PopularityValue]]
        ] = {}  # keyed by country

        self.use_cms3 = True

    def _load_show_values(self, override_path=None):
        self.wl_values = {}
        s3_path = _coalesce(
            override_path, data_path(DataPath.S3_WATCH_VALUE, TENANT) % self.date
        )
        local_path = self.base_path % "watch_value.csv"
        download_single_file(s3_path, local_path, ".csv")

        with open(local_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for cid, language, value_str in reader:
                cid = int(cid)
                tks = value_str.split(",")
                thres, values = [], []
                for i in range(0, len(tks), 2):
                    thres.append(float(tks[i]))
                    values.append(float(tks[i + 1]))
                self.wl_values[(int(cid), int(language))] = (thres, values)
        print(
            "Loaded WL Value Function",
            self.wv(1000044084, 7, 7200),
            self.wv(1000044084, 7, 1200),
            self.wv(8795, 8, 30000),
            self.wv(8795, 8, 300),
        )

    def load(self, hive_connector_factory=None, sql_context=None):
        self.movies = defaultdict(dict)
        self.tv_shows = defaultdict(dict)
        self.sports = defaultdict(dict)

        cms_store = _retry_if_use_hive(
            get_cms_store,
            hive_connector_factory=hive_connector_factory,
            sql_context=sql_context,
            country_list=self.countries,
            movie_fields=MOVIE_FIELDS,
            tv_show_fields=TV_SHOW_FIELDS,
            match_fields=MATCH_FIELDS,
            episode_fields=EPISODE_FIELDS,
            return_tuples=False,
            use_mock_table=False,
        )

        movies, tv_shows, matches, episodes = (
            cms_store.movies,
            cms_store.tv_shows,
            cms_store.matches,
            cms_store.episodes,
        )
        self.episode_repo = cms_store.episodes

        for country in self.countries:
            enrich_metadata(
                movies.all_entities_of_country(country),
                tv_shows.all_entities_of_country(country),
                episodes.all_entities_of_country(country),
                matches.all_entities_of_country(country),
            )

            for content_id, tv_show in tv_shows.get_items(country):
                if self.end_ts >= tv_show.episode_start_dt > 0:
                    self.tv_shows[country][content_id] = tv_show
            for content_id, movie in movies.get_items(country):
                if movie.start_dt <= self.end_ts:
                    self.movies[country][content_id] = movie
            for content_id, match in matches.get_items(country):
                self.sports[country][content_id] = match

            print(
                f"country: {country}, valid shows: {len(self.tv_shows[country])}, valid movies: {len(self.movies[country])}"
            )

        self._load_show_values()
        self.load_popularity()

    def load_popularity(self):
        self.discover_popularity = {}
        self.state_nolang_popularity = {}

        for country in self.countries:
            country_popularity_versions = MultiVersionData()
            country_state_nolang_popularity_versions = MultiVersionData()
            for version_diff in range(1, 4):
                history_version = get_diff_date(self.date, -version_diff)
                country_popularity_versions.add_version(
                    history_version, load_discover_popularity(country, history_version)
                )
                if country == "in":
                    country_state_nolang_popularity_versions.add_version(
                        history_version,
                        load_state_nolang_popularity(country, history_version),
                    )
            self.discover_popularity[country] = country_popularity_versions.build()
            self.state_nolang_popularity[
                country
            ] = country_state_nolang_popularity_versions.build()

    def wv(self, cid, lang, seconds):
        if (cid, lang) not in self.wl_values:
            return 1.0
        thres, values = self.wl_values[(cid, lang)]
        pos = bisect_right(thres, seconds / 60) - 1
        return values[pos]

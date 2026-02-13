from tpfy.tf_dataset.metadata import MOVIE_FIELDS, TV_SHOW_FIELDS
from common.cms3.cms3_utils import get_cms_store


class CmsData:
    def __init__(self, countries, sql_context):
        self.cms3 = True
        cms_store = get_cms_store(
            country_list=countries,
            sql_context=sql_context,
            use_mock_table=False,
            movie_fields=MOVIE_FIELDS,
            tv_show_fields=TV_SHOW_FIELDS,
            return_tuples=False,
        )
        self.movies = {}
        self.tv_shows = {}
        for country in countries:
            self.movies[country] = dict(cms_store.movies.get_items(country))
            self.tv_shows[country] = dict(cms_store.tv_shows.get_items(country))

            movie_content_ids = list(self.movies[country].keys())
            show_content_ids = list(self.tv_shows[country].keys())
            for cid in movie_content_ids[:10]:
                assert isinstance(cid, int)
                print("movie cid", cid)
            for cid in show_content_ids[:10]:
                assert isinstance(cid, int)
                print("show cid", cid)


class CumWatchMetadata:
    def __init__(self, date, countries, sql_context, cms_data: CmsData):
        self.cms_data = cms_data

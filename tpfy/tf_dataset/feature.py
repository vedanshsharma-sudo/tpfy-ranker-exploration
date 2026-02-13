import numpy as np

from model.fid import slot_bits, feature_bits, StringSlot, UIntSlot
from model.schema import FeatureSchemaMeta, BaseFeatureSchema

ONE_DAY_SECS = 86400
JOINED_MAX_MONTH = 24
PUBLISHED_YEAR_BUCKETS = 7


def _coalesce(value, default):
    return value if value else default


def _sports_wv(watch, video):
    wv = watch.watch_len / 3600
    if video.duration > 0:
        wv = max(watch.watch_len / video.duration, wv)
    wv = min(1.0, wv)
    if video.content_type_id != 301:
        wv = wv * 0.1
    return wv


def get_slot_id(fid):
    return fid >> feature_bits


# slot id must not be changed after release
class FeatureSchema(BaseFeatureSchema, metaclass=FeatureSchemaMeta):
    # user slots
    dw_pid = StringSlot(1)  # not used

    raw_plan_types = StringSlot(2)

    gender = StringSlot(3)
    age = StringSlot(4)

    platform = StringSlot(5)
    country = StringSlot(6)
    state = StringSlot(7)
    asn_number = StringSlot(8)
    carrier = StringSlot(9)
    manufacturer = StringSlot(10)
    model = StringSlot(11)
    screen_height = StringSlot(12)

    joined_bucket = StringSlot(13)
    request_country = StringSlot(14)
    is_paid_user = UIntSlot(15)
    is_honeypot_user = UIntSlot(16)  # free timer status
    is_honeypot_enabled = UIntSlot(17)  # global setting

    watched_content = StringSlot(20)
    watched_entitlements = StringSlot(21)
    watched_language_id = StringSlot(22)
    watched_studio_id = StringSlot(23)
    watched_genre_id = StringSlot(24)
    watched_production_house_id = StringSlot(25)
    watched_year_bucket = StringSlot(26)
    watched_content_type = StringSlot(27)
    watched_parental_rating_id = StringSlot(28)

    watched_sports_language_id = StringSlot(30)
    watched_sports_tournament_id = StringSlot(32)
    watched_sports_team = StringSlot(33)
    watched_sports_game_id = StringSlot(34)

    target_id = StringSlot(50)
    target_content_type_id = StringSlot(51)
    target_genre_id = StringSlot(52)
    target_studio_id = StringSlot(53)
    target_production_house_id = StringSlot(54)
    target_entitlement = StringSlot(55)
    target_parental_rating_id = StringSlot(56)
    target_year_bucket = StringSlot(57)

    target_priority = UIntSlot(58)

    # additional features for cum watch
    target_channel_id = StringSlot(59)
    target_release_time_bucket = StringSlot(60)
    target_latest_episode_time_bucket = StringSlot(61)
    target_alive_episode_count_bucket = StringSlot(62)
    target_language_id = StringSlot(63)
    target_show_type = StringSlot(64)
    target_show_btv = StringSlot(65)
    target_past_watched_time_cumwt = UIntSlot(67)
    target_last_watch_to_now_cumwt = UIntSlot(68)

    watched_channel_weight = StringSlot(107)
    watched_release_time_bucket = StringSlot(108)

    total_watch_hours_bucket = StringSlot(109)
    days_from_first_watch_bucket = StringSlot(110)

    adsGender = StringSlot(111)
    adsAge = StringSlot(112)
    infrequentUser = UIntSlot(113)
    subPropensity = UIntSlot(114)
    l30Active = StringSlot(115)
    netAcuityGeoState = StringSlot(116)
    netAcuityGeoCity = StringSlot(117)
    watchBehaviorCohortId = StringSlot(124)
    subsBehaviorCohortId = StringSlot(125)

    online_platform = StringSlot(199)

    # additional features for conversion
    device_price_tag = StringSlot(200)
    tray_type = StringSlot(201)
    tray_id = StringSlot(202)
    tray_position = StringSlot(203)
    tile_position = UIntSlot(204)
    target_last_watch_to_now = StringSlot(205)
    target_watched_time_bucket = StringSlot(206)
    sub_plan_expired_to_now = StringSlot(207)

    hour = UIntSlot(208)
    weekday = UIntSlot(209)


class ScalarIndexMeta(type):
    def __new__(mcs, name, bases, dct):
        max_index = 0
        for name, index in dct.items():
            if isinstance(index, int):
                max_index = max(max_index, index)
        dct["max_index"] = max_index
        return type.__new__(mcs, name, bases, dct)


class ScalarIndex(metaclass=ScalarIndexMeta):
    # 0 is reserved for invalid
    genre_weight = 1
    studio_weight = 2
    parental_rating_weight = 3
    content_type_weight = 4
    production_house_weight = 5
    year_bucket_weight = 6

    log_recency_weight = 7
    release_3d = 8
    release_7d = 9
    release_30d = 10

    stateNoLangPopularityDV = 11
    stateNoLangPopularityWV = 12
    primaryLangPopularityDV = 13
    primaryLangPopularityWV = 14


schema = FeatureSchema

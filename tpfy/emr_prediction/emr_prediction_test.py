from common.data.behavior.watch import Watch2


def print_watches(p_id, watches, contents, genres, languages):
    print("=========", p_id, "=========")
    print("ent_watches:", len(contents))
    for watch in Watch2.grouping(watches[0]):
        if watch.content_id in contents:
            print(
                "\t",
                contents.get(watch.content_id).to_rich_str(genres, languages),
                watch,
            )
        else:
            print("\t", watch.content_id, next(iter(contents.keys())), watch)
    print("ns_watches:")
    for watch in Watch2.grouping(watches[1]):
        if watch.content_id in contents:
            print(
                "\t",
                contents.get(watch.content_id).to_rich_str(genres, languages),
                watch,
            )
        else:
            print("\t", watch.content_id, next(iter(contents.keys())), watch)


def print_recommendations(title, results, contents, genres, languages):
    print(".........", title, ".........", len(contents))
    for result in results:
        if result[0] in contents:
            print("\t", result, contents.get(result[0]).to_rich_str(genres, languages))
        else:
            print(
                "\t",
                result,
                next(iter(contents.keys())),
                type(result[0]),
                type(next(iter(contents.keys()))),
            )


def test_prediction(
    model_path, contents_bc, data_bc, common_candidates_bc, prediction_start, args
):
    watch_ts = prediction_start - 86400 * 10
    joined_on = prediction_start - 86400 * 30
    sub_info = "HotstarPremium,%s,%s" % (
        prediction_start - 86400 * 30,
        prediction_start + 86400 * 30,
    )
    device_info = None
    clicks = None
    subs_tray_impressions = None

    meta = [joined_on, sub_info, device_info, clicks, subs_tray_impressions, None, None]

    watches1 = (
        [
            1660000038,
            3,
            120,
            watch_ts,
            watch_ts,
            1000217977,
            9,
            120,
            watch_ts,
            watch_ts,
        ],
        [],
        [],
        # [
        #     {'content_id': 1660000038, 'tray_id': '5187', 'count': 12}
        # ],
        "26",
        "m",
        *meta,
    )
    watches2 = (
        [
            1660000038,
            3,
            120,
            watch_ts,
            watch_ts,
            1000217977,
            9,
            120,
            watch_ts,
            watch_ts,
        ],
        [2001707650, 3, 12000, watch_ts, watch_ts],
        [],
        # None,
        "26",
        "m",
        *meta,
    )
    watches3 = (
        [
            1660000038,
            3,
            12000,
            watch_ts,
            watch_ts,
            1000217977,
            9,
            12000,
            watch_ts,
            watch_ts,
        ],
        [2001707650, 3, 12000, watch_ts, watch_ts],
        [],
        # [
        #     {'content_id': 1770000948, 'tray_id': '5187', 'count': 12},
        #     {'content_id': 1260014811, 'tray_id': '5187', 'count': 2},
        #     {'content_id': 1260017860, 'tray_id': '5187', 'count': 5}
        # ],
        "26",
        "m",
        *meta,
    )
    watches4 = (
        [],
        [2001707650, 3, 12000, watch_ts, watch_ts],
        [],
        # [
        #     {'content_id': 1660000038, 'tray_id': '5187', 'count': 12}
        # ],
        "26",
        "m",
        *meta,
    )
    watches5 = (
        [
            3941,
            7,
            12000,
            watch_ts,
            watch_ts,
            18310,
            7,
            12000,
            watch_ts,
            watch_ts,
            1000044065,
            7,
            12000,
            watch_ts,
            watch_ts,
        ],
        [2001707650, 3, 12000, watch_ts, watch_ts],
        [],
        # [
        #     {'content_id': 1100005285, 'tray_id': '5187', 'count': 12}
        # ],
        "26",
        "m",
        *meta,
    )
    user_watches = [
        ("user1", watches1),
        ("user9", watches2),
        ("user3", watches3),
        ("usera", watches4),
        ("userb", watches5),
    ]

    from tpfy.emr_prediction.emr_prediction import predict

    results = list(
        predict(
            user_watches,
            model_path,
            contents_bc,
            data_bc,
            common_candidates_bc,
            prediction_start,
            args,
            watch_len_thres=-1,
        )
    )

    movies, shows, clips, matches, genres, languages = contents_bc.value
    contents = {}
    for cid, cnt in movies.items():
        contents[cid] = cnt
    for cid, cnt in shows.items():
        contents[cid] = cnt

    for (p_id, watches), r in zip(user_watches, results):
        print_watches(p_id, watches, contents, genres, languages)
        print_recommendations("home", r[1], contents, genres, languages)
        print_recommendations("tv", r[2], contents, genres, languages)
        print_recommendations("movie", r[3], contents, genres, languages)
        print_recommendations("disney", r[4], contents, genres, languages)

import math


def log_behaviors(out, section_name, watches, now, metadata):
    section_header = "## %s\n" % section_name
    out.write(section_header)
    data = []
    for watch in sorted(watches, key=lambda t: t.last_watch, reverse=True):
        cid = watch.content_id
        if cid in metadata.tv_shows:
            content = metadata.tv_shows[cid]
        elif cid in metadata.movies:
            content = metadata.movies[cid]
        elif cid in metadata.sports:
            content = metadata.sports[cid]
        else:
            content = None

        age = "{0:>.1f}d".format((now - watch.last_watch) / 86400)
        watch_len = "{0:>.1f}m".format(watch.watch_len / 60)
        if content is None:
            data.append(
                (
                    cid,
                    None,
                    watch.language,
                    None,
                    watch_len,
                    age,
                    watch.last_watch,
                    watch.first_watch,
                )
            )
        else:
            data.append(
                (
                    cid,
                    content.title,
                    watch.language,
                    content.content_type_id,
                    watch_len,
                    age,
                    watch.last_watch,
                    watch.first_watch,
                )
            )
    header = [
        "ID",
        "Title",
        "Lang",
        "Type",
        "Watched",
        "Recency",
        "Last At",
        "First At",
    ]
    adjustable = {"Title"}
    write_tables(out, header, data, adjustable=adjustable)


def log_results(out, candidates, scores, sources, metadata):
    out.write("## Final Results\n")
    data = []
    for (cid, lang), score in sorted(
        zip(candidates, scores), key=lambda t: t[1], reverse=True
    ):
        if cid in metadata.tv_shows:
            content = metadata.tv_shows[cid]
        elif cid in metadata.movies:
            content = metadata.movies[cid]
        else:
            content = None

        title, cnt_type = (
            (content.title, content.content_type_id) if content else (None, None)
        )
        data.append(
            (
                cid,
                title,
                "{0:>.3f}".format(score),
                lang,
                cnt_type,
                ", ".join(sources[(cid, lang)]),
            )
        )
    header = ["ID", "Title", "Score", "Lang", "Type", "Sources"]
    adjustable = {"Sources"}
    write_tables(out, header, data, adjustable=adjustable)


def log_gradients(out, name, var_gradients):
    out.write("## %s\n" % name)
    header = ["Var", "Abs Gradient", "Gradient"]
    data = [
        (var, "{0:>.3f}".format(math.fabs(gradient)), "{0:>.3f}".format(gradient))
        for var, gradient in sorted(
            var_gradients.items(), key=lambda t: math.fabs(t[1]), reverse=True
        )
    ]
    write_tables(out, header, data)


def write_tables(out, header, data, adjustable=None, max_width=160):
    data = [[str(v) for v in row] for row in data]
    column_width = [len(name) for name in header]
    for row in data:
        for i, v in enumerate(row):
            column_width[i] = max(column_width[i], min(len(v), 50))
    value_max_width = max_width - len(header) - 1
    value_width = sum(column_width)

    if value_max_width < value_width:
        if adjustable is None:
            adjustable = set()
        fixed_width = sum(
            column_width[i] for i, name in enumerate(header) if name not in adjustable
        )
        adjustable_width = value_max_width - fixed_width
        value_width = value_width - fixed_width

        for i, name in enumerate(header):
            if name in adjustable:
                column_width[i] = int(column_width[i] * adjustable_width / value_width)

    table_width = sum(column_width) + len(header) + 1

    out.write("+")
    out.write("-" * (table_width - 2))
    out.write("+\n")

    out.write("|")
    for value, width in zip(header, column_width):
        if len(value) > width:
            out.write(value[0:width])
        else:
            form = "{0:>%ss}" % width
            out.write(form.format(value))
        out.write("|")
    out.write("\n")

    out.write("+")
    out.write("-" * (table_width - 2))
    out.write("+\n")

    for row in data:
        for value, width in zip(row, column_width):
            out.write("|")
            if len(value) > width:
                out.write(value[0:width])
            else:
                form = "{0:>%ss}" % width
                out.write(form.format(value))
        out.write("|\n")

    out.write("+")
    out.write("-" * (table_width - 2))
    out.write("+\n")

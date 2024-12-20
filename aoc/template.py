import codecs
import os
import re
from collections import OrderedDict
from string import Template
from pathlib import Path

from aoc.helpers import copy_tree


def _script_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")


def mkdir_p(path: str):
    joined = os.path.join(_script_dir(), path)
    dir_path = joined if os.path.isdir(joined) else os.path.dirname(joined)
    dir_path = dir_path if dir_path.endswith("/") else f"{dir_path}/"
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def copy_from_source():
    folders = []
    for walker in os.walk("../../aoc/year_2024"):
        for folder in walker[1]:
            if re.search(r"day_\d", folder):
                folders.append(os.path.join(walker[0], folder))

    source_targets = []
    for source in folders:
        tail_name = re.findall(r"year_\d.*$", source)[0]
        target = os.path.join(_script_dir(), tail_name)
        source_targets.append((source, target))
        print(f"Copy folder '{source}' to '{target}'")

    print("")
    for source, target in source_targets:
        copy_tree(source, target)


def template_readme():
    folders = []
    advent_folder = os.path.join(_script_dir(), "year_2024")
    for walker in os.walk(advent_folder):
        for folder in walker[1]:
            if re.search(r"day_\d", folder):
                sub_folder = os.path.join(walker[0], folder)
                folders.append(os.path.join(walker[0], sub_folder))

    advents_codes = []
    folders.sort(key=lambda kk: kk)
    for folder in folders:
        for files_walker in os.walk(folder):
            if "sink" in files_walker[0]:
                continue
            for file in files_walker[2]:
                if re.search(r"solve(_\d+)?\.[a-zA-z]{1,15}", file):
                    advents_codes.append(os.path.abspath(os.path.join(files_walker[0], file)))

    advent_of_codes = OrderedDict()
    for advents_code in advents_codes:
        import_rows = []
        body = []
        body_found = False

        if not advents_code.endswith("py"):
            continue

        with codecs.open(advents_code, mode='r', encoding="utf-8", errors='strict', buffering=-1) as fp:
            if not advents_code.endswith("py"):
                text = ""
                for line in fp.readlines():
                    text += line
                body.append(text)
            else:
                last_line = None
                for line in fp.readlines():
                    if line.startswith("from") or line.startswith("import"):
                        import_rows.append(line)
                        continue

                    helper_line = re.search("^(_?default|puzzle|test|challenge)", line, re.IGNORECASE)
                    body_started = not body_found and (not line.strip() or line.startswith("#"))
                    if body_started or helper_line:
                        continue

                    if line.startswith('if __name__ == "__main__":'):
                        break

                    if not body_started and re.search(r"^\w", line):
                        body_found = True

                    if re.search(r"^\s*$", line) and re.search(r"^\s*$", last_line):
                        continue

                    body.append(line)
                    last_line = line

        text = ""
        for line in import_rows:
            text += line

        if import_rows:
            for line in ["\n", "\n"]:
                text += line

        for n, line in enumerate(body):
            if "".join(body[n:]):
                text += line

        advent_of_codes[advents_code] = text

    codes = ""
    readme_template = os.path.join(_script_dir(), "aoc/README.template")
    with open(readme_template, 'r') as fp:
        for k, text in reversed(advent_of_codes.items()):
            file_type = "py" if k.endswith("py") else "javascript"
            title = re.findall("(year_.*)", k)[0]
            t = Template("\n## $title\n\n```$type\n$code```")
            codes += t.substitute(dict(title=title, code=text, type=file_type))

        readme = Template("".join(fp.readlines())).substitute(dict(code=codes))

        readme_file = os.path.join(_script_dir(), "README.md")
        with open(readme_file, 'w') as out_fp:
            out_fp.write(readme)


if __name__ == "__main__":
    copy_from_source()
    template_readme()


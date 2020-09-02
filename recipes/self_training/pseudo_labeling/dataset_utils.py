from __future__ import absolute_import, division, print_function, unicode_literals


class Transcript(object):
    def __init__(self, sid, path, duration, transcript=""):
        self.sid = sid
        self.path = path
        self.duration = duration
        self.transcript = transcript

    def __hash__(self):
        return hash(self.sid)

    def __eq__(self, other):
        return self.sid == other.sid


def create_transcript_dict_from_listfile(listpath):
    res = {}
    with open(listpath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            els = line.split()
            sid = els[0]
            res[sid] = Transcript(sid, els[1], els[2], " ".join(els[3:]))
    return res


listfileformat = """{sid} {path} {duration} {transcript}\n"""


def write_transcript_list_to_file(transcript_dict, outfilename):
    with open(outfilename, "w") as f:
        for sid in transcript_dict.keys():
            transcript = transcript_dict[sid]
            f.write(
                listfileformat.format(
                    sid=transcript.sid,
                    path=transcript.path,
                    duration=transcript.duration,
                    transcript=transcript.transcript.rstrip(),
                )
            )


def zip_datasets(first, second):
    """
    Zips together two list dataset representations. Preserves the sample ID for the
    """
    output = first
    for sample_key in second.keys():
        sample = second[sample_key]
        if sample.sid in output.keys():
            raise Exception("Attempted to write duplicate sample ID: " + sample.sid)
        output[sample.sid] = sample
    return output
